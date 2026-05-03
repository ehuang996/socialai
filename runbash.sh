{
echo "=== Pipeline Results: 104 input rows, Stage 1 -> 94 ==="
echo ""
printf "%-35s %6s %6s %6s %6s\n" "Measure" "S1" "S2" "S3" "S4"
printf "%-35s %6s %6s %6s %6s\n" "---" "---" "---" "---" "---"

for m in 1B_intentional_human_speech 1B_human_pronoun 1C_identity_transparency 2A_fabricated_personal_details 2B_explicit_emotions 2B_implicit_emotions 2B_romantic_bonding 2C_sycophancy 2D_human_relationship_encouragement 3A_engagement_hooks; do
    ROOT="/project2/robinjia_875/ehuang97/socialai/experiments/verify/verify_${m}/results"
    
    s2=$(python3 -c "import json; print(sum(1 for l in open('${ROOT}/${m}_scores.jsonl') if json.loads(l).get('keep')))" 2>/dev/null || echo "0")
    
    s3f="${ROOT}/${m}_high_quality.jsonl"
    if [[ -f "$s3f" && -s "$s3f" ]]; then
        s3_chitchat=$(python3 -c "
import json
rows = [json.loads(l) for l in open('$s3f')]
both = sum(1 for r in rows if r.get('model_responses',{}).get('gpt_4o_mini',{}).get('chitchat_keep') and r.get('model_responses',{}).get('gpt_4o_mini',{}).get('category_keep'))
print(both)
")
    else
        s3_chitchat="0"
    fi
    
    s4f="${ROOT}/${m}_final.jsonl"
    if [[ -f "$s4f" && -s "$s4f" ]]; then
        s4=$(python3 -c "
import json
rows = [json.loads(l) for l in open('$s4f')]
both = sum(1 for r in rows if r.get('model_responses',{}).get('claude_opus_4_6',{}).get('chitchat_keep') and r.get('model_responses',{}).get('claude_opus_4_6',{}).get('category_keep'))
print(both)
")
    else
        s4="0"
    fi
    
    printf "%-35s %6s %6s %6s %6s\n" "$m" "94" "$s2" "$s3_chitchat" "$s4"
done
} > outputbash