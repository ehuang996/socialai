# this is the preprocessing script; this will run first before llm_as_a_judge
# this filter script will run on 5 high-prio GPU on nlp-cluster

# imports
import os
import json
import asyncio
from openai import AsyncOpenAI
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio

# configuration
# DATASET_NAME = "yuntian-deng/WildChat-4.8M-Full"
DATASET_NAME = "allenai/WildChat-1M"
CACHE_DIR = "/project2/robinjia_875/wangzhu/eric_huang/.cache/huggingface"

# model + server
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
PORT = os.getenv("VLLM_PORT", "8001")
API_URL = f"http://localhost:{PORT}/v1"

# sharding
SHARD_ID = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
NUM_SHARDS = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))

# paths
CACHE_DIR = "/project2/robinjia_875/wangzhu/eric_huang/.cache/huggingface"
OUTPUT_FILE = f"wildchat_preprocessed_part_{SHARD_ID}.jsonl"
CONCURRENCY_LIMIT = 100

FILTER_PROMPT = """You are a dataset curator. Analyze the USER's prompt.
    
Determine if this conversation should be KEPT or DISCARDED based on these rules:

KEEP IF:
- Simple chitchat (e.g., "Hi", "How are you?").

DISCARD IF:
- The user asks the AI to act as a specific persona (Roleplay).
- Basic creative writing tasks (e.g., "Write a poem about trees", "Write a letter to my boss").
- Programming tasks (e.g., "Fix this bug", "How to center a div").
- Remove online threads and posts, or story completion.
- A large corpus of random text but no prompt or instructions.

Return ONLY valid JSON:
{
  "reasoning": "...",
  "keep": true/false
}"""


async def main():
    # client setup
    client = AsyncOpenAI(base_url=API_URL, api_key="EMPTY")
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    # load data
    print(f"[Shard {SHARD_ID}] Loading {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split="train", cache_dir=CACHE_DIR)

    # English filter
    dataset = dataset.filter(lambda x: x['language'] == 'English')

    # Deduplication
    if NUM_SHARDS > 1 and SHARD_ID == 0:
        print(f"[Shard {SHARD_ID}] Deduplicating global dataset...")

    seen_hashes = set()
    unique_indices = []

    # Fast iteration over the hash column
    for i, h in enumerate(dataset['conversation_hash']):
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_indices.append(i)
            
    dataset = dataset.select(unique_indices)
    print(f"[Shard {SHARD_ID}] Rows after English+Dedup: {len(dataset)}")

    # Shard Dataset
    if NUM_SHARDS > 1:
        print(f"[Shard {SHARD_ID}] Slicing dataset: Part {SHARD_ID+1}/{NUM_SHARDS}")
        dataset = dataset.shard(num_shards=NUM_SHARDS, index=SHARD_ID)
    
    print(f"[Shard {SHARD_ID}] Rows to process: {len(dataset)}")

    # define worker
    async def process_row(row, f_out):
        async with sem:
            # input checks
            if not row.get('conversation'): return 
            user_msg = row['conversation'][0]['content']

            if not user_msg or len(user_msg.strip()) < 5: return 

            # Truncate to 4k characters
            if len(user_msg) > 4000:
                user_msg = user_msg[:4000]
                
            # API Call 
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": FILTER_PROMPT},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0,
                max_tokens=512,
            )

            # parse output
            raw = response.choices[0].message.content.strip()
            if "```" in raw:
                raw = raw.split("```json")[-1].split("```")[0].strip()
            
            try:
                decision = json.loads(raw)

                if not isinstance(decision, dict):
                    return 0
            except:
                return 0

            # save to disk
            if decision.get("keep"):
                row['filter_reasoning'] = decision['reasoning']

                f_out.write(json.dumps(row, default=str) + "\n")
                f_out.flush()
                return 1
            return 0

    # run
    with open(OUTPUT_FILE, "a") as f_out:
        tasks = [process_row(row, f_out) for row in dataset]
        await tqdm_asyncio.gather(*tasks)

    print(f"[Shard {SHARD_ID}] Done.")

if __name__ == "__main__":
    asyncio.run(main())