import os
import json
import asyncio
from openai import AsyncOpenAI, BadRequestError
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from tqdm.asyncio import tqdm_asyncio

# Config
# Models
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct" 
PORT = os.getenv("VLLM_PORT", "8001") 
API_URL = f"http://localhost:{PORT}/v1"

# File Paths
CACHE_DIR = "/project2/robinjia_875/wangzhu/eric_huang/.cache/huggingface"
INPUT_FILES = "wildchat_preprocessed_*.jsonl"

# Array / Sharding Config from slurm
SHARD_ID = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
NUM_SHARDS = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))

# File paths
OLD_PROGRESS_FILE = "wildchat_scores_progress.jsonl" 
NEW_OUTPUT_FILE = f"wildchat_scores_part_{SHARD_ID}.jsonl" 

CONCURRENCY_LIMIT = 64

# prompt
PROMPT_FILE = "judge_prompt_anthropomorphism.txt"

# Client setup
client = AsyncOpenAI(base_url=API_URL, api_key="EMPTY")

def load_system_prompt():
    """Loads the system prompt from an external file."""
    with open(PROMPT_FILE, "r") as f:
        return f.read().strip()

def setup_dataset():
    "Load JSONL file"

    dataset = load_dataset("json", data_files=INPUT_FILES, split="train", cache_dir=CACHE_DIR)

    if NUM_SHARDS > 1:
        print(f"[Shard {SHARD_ID}] Slicing dataset: Part {SHARD_ID+1}/{NUM_SHARDS}")
        dataset = dataset.shard(num_shards=NUM_SHARDS, index=SHARD_ID)
    
    return dataset


async def process_single_example(sem, example, f_out, system_prompt):
    async with sem:
        user_msg = example['conversation'][0]['content']
        asst_msg = example['conversation'][1]['content']

        # skip the API calls that may fail for A6000 GPU
        total_len = len(user_msg or "") + len(asst_msg or "")
        if total_len > 24000:
            return

        conversation_text = f"USER: {user_msg}\nASSISTANT: {asst_msg}"

        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation_text}
            ],
            temperature=0,
            max_tokens=512,
        )
        raw_output = response.choices[0].message.content.strip()

        if "```" in raw_output:
            raw_output = raw_output.split("```json")[-1].split("```")[0].strip()    
        
        # note that I put try-except b/c it kept on crashing due to 1 bad example...
        try:    
            data = json.loads(raw_output)
        
            if not isinstance(data, dict):
                raise ValueError("JSON parsed but output not a dictionary.")
            
        except (json.JSONDecodeError, ValueError, TypeError):
            data = {"error": "parsing_failed", "raw_output": raw_output}


        result = {
                "hash": example['conversation_hash'],
                "user_input": user_msg,
                "assistant_response": asst_msg,
                "timestamp": str(example['timestamp']), 
                **data
            }
        
        f_out.write(json.dumps(result) + "\n")
        
async def main():
    SYSTEM_PROMPT = load_system_prompt()
    dataset = setup_dataset()
    print(f"[Shard {SHARD_ID}] Dataset ready. Rows to process: {len(dataset)}")

    processed_hashes = set()

    # Load Checkpoint
    if os.path.exists(OLD_PROGRESS_FILE):
        print(f"[Shard {SHARD_ID}] Reading previous global progress...")
        with open(OLD_PROGRESS_FILE, "r") as f:
            for line in f:
                processed_hashes.add(json.loads(line)['hash'])

    if os.path.exists(NEW_OUTPUT_FILE):
        print(f"[Shard {SHARD_ID}] Reading local shard progress...")
        with open(NEW_OUTPUT_FILE, "r") as f:
            for line in f:
                processed_hashes.add(json.loads(line)['hash'])
    
    print(f"[Shard {SHARD_ID}] Found {len(processed_hashes)} total skipped records.")

    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = []
    
    with open(NEW_OUTPUT_FILE, "a") as f_out:
        for example in dataset:
            if example['conversation_hash'] in processed_hashes:
                continue
            tasks.append(process_single_example(sem, example, f_out, SYSTEM_PROMPT))

        print(f"[Shard {SHARD_ID}] Starting processing of {len(tasks)} items...")
        if tasks:
            await tqdm_asyncio.gather(*tasks)
        else:
            print(f"[Shard {SHARD_ID}] All items in this shard completed!")

if __name__ == "__main__":
    asyncio.run(main())