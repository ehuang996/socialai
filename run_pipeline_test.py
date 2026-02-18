#!/usr/bin/env python3
"""Run actual pipeline test with real model inference."""

import json
import os
import sys
import tempfile
import asyncio

# Test conversation
test_conversation = {
    "conversation_hash": "test_hash_123",
    "conversation": [
        {"role": "user", "content": "Hello how are you"},
        {"role": "assistant", "content": "I am doing well sir"}
    ],
    "language": "English",
    "timestamp": "2024-01-15 10:30:00"
}

async def test_stage1():
    """Test Stage 1: ChitChatJudge"""
    print("=" * 70)
    print("STAGE 1: CHIT-CHAT FILTER")
    print("=" * 70)

    from openai import AsyncOpenAI

    # Create test input file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        input_file = f.name
        f.write(json.dumps(test_conversation) + "\n")

    output_file = tempfile.mktemp(suffix='.jsonl')

    # Use the running server
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY"
    )

    # Stage 1 prompt
    system_prompt = """You are a dataset curator. Analyze the USER's prompt.

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

    user_msg = test_conversation["conversation"][0]["content"]

    print(f"\n📤 INPUT:")
    print(f"User message: {user_msg}")

    # Call the model
    response = await client.chat.completions.create(
        model="Qwen3-32B",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.0,
        max_tokens=512
    )

    raw_output = response.choices[0].message.content.strip()

    print(f"\n📥 RAW MODEL OUTPUT:")
    print(raw_output)

    # Parse JSON
    if "```" in raw_output:
        raw_output = raw_output.split("```json")[-1].split("```")[0].strip()

    try:
        parsed = json.loads(raw_output)
        print(f"\n✅ PARSED OUTPUT:")
        print(json.dumps(parsed, indent=2))

        if parsed.get("keep"):
            # Save full row + filter_reasoning
            output_row = {**test_conversation, "filter_reasoning": parsed.get("reasoning", "")}
            print(f"\n💾 SAVED TO FILE (because keep=true):")
            print(json.dumps(output_row, indent=2, default=str))

            # Save for stage 2
            with open(output_file, 'w') as f:
                f.write(json.dumps(output_row) + "\n")
        else:
            print("\n❌ NOT SAVED (keep=false)")
            output_file = None

    except json.JSONDecodeError as e:
        print(f"\n❌ Failed to parse JSON: {e}")
        output_file = None

    # Cleanup
    os.unlink(input_file)

    return output_file

async def test_stage2(stage1_output):
    """Test Stage 2: AnthropomorphicJudge"""
    print("\n" + "=" * 70)
    print("STAGE 2: ANTHROPOMORPHISM JUDGE")
    print("=" * 70)

    from openai import AsyncOpenAI

    # Use stage 1 output or create test input
    if stage1_output and os.path.exists(stage1_output):
        with open(stage1_output, 'r') as f:
            row = json.loads(f.readline())
    else:
        row = test_conversation

    # Use the running server
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY"
    )

    # Load the actual prompt file
    with open("prompts/judge_prompt_anthropomorphism.txt", "r") as f:
        system_prompt = f.read().strip()

    user_msg = row["conversation"][0]["content"]
    asst_msg = row["conversation"][1]["content"]

    conversation_text = f"USER: {user_msg}\nASSISTANT: {asst_msg}"

    print(f"\n📤 INPUT:")
    print(conversation_text)

    # Call the model
    response = await client.chat.completions.create(
        model="Qwen3-32B",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": conversation_text}
        ],
        temperature=0.0,
        max_tokens=512
    )

    raw_output = response.choices[0].message.content.strip()

    print(f"\n📥 RAW MODEL OUTPUT:")
    print(raw_output)

    # Parse JSON
    if "```" in raw_output:
        raw_output = raw_output.split("```json")[-1].split("```")[0].strip()

    try:
        parsed = json.loads(raw_output)
        print(f"\n✅ PARSED OUTPUT:")
        print(json.dumps(parsed, indent=2))

        # Create output in old format
        result = {
            "hash": row["conversation_hash"],
            "user_input": user_msg,
            "assistant_response": asst_msg,
            "timestamp": str(row.get("timestamp", "")),
            **parsed
        }

        print(f"\n💾 SAVED TO FILE:")
        print(json.dumps(result, indent=2))

    except json.JSONDecodeError as e:
        print(f"\n❌ Failed to parse JSON: {e}")

    # Cleanup
    if stage1_output and os.path.exists(stage1_output):
        os.unlink(stage1_output)

async def main():
    print("\n🧪 RUNNING ACTUAL PIPELINE TEST")
    print(f"Test input:")
    print(f"  User: {test_conversation['conversation'][0]['content']}")
    print(f"  Assistant: {test_conversation['conversation'][1]['content']}\n")

    # Run Stage 1
    stage1_output = await test_stage1()

    # Run Stage 2
    await test_stage2(stage1_output)

    print("\n✅ Test complete!")

if __name__ == "__main__":
    asyncio.run(main())
