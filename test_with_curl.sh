#!/bin/bash
# Test the pipeline using curl to call the vLLM server directly

echo "==============================================="
echo "STAGE 1: CHIT-CHAT FILTER"
echo "==============================================="

STAGE1_PROMPT='You are a dataset curator. Analyze the USER'"'"'s prompt.

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
}'

USER_MSG="Hello how are you"

echo ""
echo "📤 INPUT:"
echo "User message: $USER_MSG"
echo ""

# Call Stage 1
STAGE1_RESPONSE=$(curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-VL-8B-Instruct",
    "messages": [
      {"role": "system", "content": "'"${STAGE1_PROMPT}"'"},
      {"role": "user", "content": "'"${USER_MSG}"'"}
    ],
    "temperature": 0.0,
    "max_tokens": 512
  }')

echo "📥 MODEL OUTPUT:"
echo "$STAGE1_RESPONSE" | python3 -c "import sys, json; resp=json.load(sys.stdin); print(resp['choices'][0]['message']['content'])"
echo ""

# Extract the actual response
FILTER_RESULT=$(echo "$STAGE1_RESPONSE" | python3 -c "
import sys, json
resp = json.load(sys.stdin)
content = resp['choices'][0]['message']['content']
# Strip markdown code blocks if present
if '\`\`\`' in content:
    content = content.split('\`\`\`json')[-1].split('\`\`\`')[0].strip()
print(content)
")

echo "✅ PARSED FILTER OUTPUT:"
echo "$FILTER_RESULT" | python3 -m json.tool

echo ""
echo "==============================================="
echo "STAGE 2: ANTHROPOMORPHISM JUDGE"
echo "==============================================="

STAGE2_PROMPT=$(cat prompts/judge_prompt_anthropomorphism.txt)

ASST_MSG="I am doing well sir"
CONVERSATION="USER: ${USER_MSG}\nASSISTANT: ${ASST_MSG}"

echo ""
echo "📤 INPUT:"
echo -e "$CONVERSATION"
echo ""

# Call Stage 2
STAGE2_RESPONSE=$(curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-VL-8B-Instruct",
    "messages": [
      {"role": "system", "content": "'"${STAGE2_PROMPT}"'"},
      {"role": "user", "content": "'"${CONVERSATION}"'"}
    ],
    "temperature": 0.0,
    "max_tokens": 512
  }')

echo "📥 MODEL OUTPUT:"
echo "$STAGE2_RESPONSE" | python3 -c "import sys, json; resp=json.load(sys.stdin); print(resp['choices'][0]['message']['content'])"
echo ""

# Extract and parse the judge result
JUDGE_RESULT=$(echo "$STAGE2_RESPONSE" | python3 -c "
import sys, json
resp = json.load(sys.stdin)
content = resp['choices'][0]['message']['content']
# Strip markdown code blocks if present
if '\`\`\`' in content:
    content = content.split('\`\`\`json')[-1].split('\`\`\`')[0].strip()
print(content)
")

echo "✅ PARSED JUDGE OUTPUT:"
echo "$JUDGE_RESULT" | python3 -m json.tool

echo ""
echo "💾 FINAL OUTPUT FORMAT (Stage 2):"
python3 << EOF
import json
judge_result = json.loads('''${JUDGE_RESULT}''')
final_output = {
    "hash": "test_hash_123",
    "user_input": "${USER_MSG}",
    "assistant_response": "${ASST_MSG}",
    "timestamp": "2024-01-15 10:30:00",
    **judge_result
}
print(json.dumps(final_output, indent=2))
EOF

echo ""
echo "✅ Test complete!"
