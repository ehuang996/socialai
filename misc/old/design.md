# Design Notes

This file collects design decisions for the socialai pipeline that aren't obvious from the code alone.

## Stage 5 — Global chitchat veto

### Rule

In Stage 5 Phase 1 ([`collect_and_dedupe`](../src/data_preprocessing/data_preprocessing.py)), a `user_input` is dropped from the entire output if **any** measure's Stage 4 judge (Opus 4.6) returned `chitchat_keep=false` for it — even if every other measure for that same `user_input` returned both `chitchat_keep=true` and `category_keep=true`.

### Motivation

Each measure's `final_filter` row records two judgments from Opus:

- `chitchat_keep` — "is this conversation casual chitchat at all?"
- `category_keep` — "does this conversation exhibit the specific behavior this measure targets?"

`category_keep` is measure-specific by design: a conversation may exhibit sycophancy but not engagement hooks, so the same conversation can have `category_keep=true` for one measure and `false` for another. That is expected and correct.

`chitchat_keep`, however, is **a property of the conversation itself**, independent of which measure is being evaluated. If Opus says "this isn't chitchat" while judging `1B_intentional_human_speech`, it would say the same thing while judging `1C_identity_transparency` — the input is the same conversation, the question is the same question. The only reason different measures can disagree on `chitchat_keep` is judge noise.

When the judges disagree, the conservative interpretation is to trust the `false`. Including a borderline-non-chitchat conversation pollutes downstream model evaluation (Stages 6–8): if the input isn't really chitchat, the behaviors we measure aren't really chitchat behaviors.

### Example

Suppose conversation `C` has Stage 4 results across two measures:

| Measure                      | chitchat_keep | category_keep |
| ---------------------------- | ------------- | ------------- |
| `1B_intentional_human_speech`      | **false**     | false         |
| `1C_identity_transparency`   | true          | true          |

Without the veto, `C` would be kept as a `1C_identity_transparency` row. With the veto, `C` is dropped from the output entirely, because `1B`'s judge said it isn't chitchat.

### Behavior

`collect_and_dedupe` does a two-pass scan over the same set of `<measure>_final.jsonl` files:

1. **Pass 1 — build veto set.** Iterate every measure file; collect `user_input`s where `model_responses[model].chitchat_keep is False`. Errors and missing fields do **not** veto (we don't know what the judge would have said).
2. **Pass 2 — collect & dedupe (existing logic).** For each row, skip if `user_input` is in the veto set, otherwise apply the existing `chitchat_keep AND category_keep` test.

The summary print includes `Veto set: N user_inputs ...` and `Dropped by chitchat veto: M` so the impact is visible in the run log.

### Why "errors are not vetoes"

Phase 1 already treats Stage 4 errors as skip-not-veto everywhere else. Treating an error as a veto would let API hiccups silently shrink the dataset in non-reproducible ways. If you want stricter behavior (every kept conversation must have been judged-as-chitchat by every measure with no errors), it should be a separate explicit flag.
