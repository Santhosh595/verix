# Multi-Modal Evidence Review — Solution

## How to run

```bash
cd code
pip install -r requirements.txt
cp .env.example .env   # fill in API key(s)

# Evaluate against the labeled sample set
PYTHONPATH=. python evaluation/main.py

# Run the full pipeline on dataset/claims.csv -> output.csv
python main.py
```

## Architecture

```
main.py → pipeline.run_batch() → run_claim() for each claim:
  1. claim_parser.parse_claim()     — extract claim from conversation text
  2. vision_analyzer.analyze_claim_images() — VLM per-image analysis
  3. evidence_checker.check_evidence() — apply evidence_requirements.csv rules
  4. risk_assessor.assess_image_risk() + assess_history_risk() — compute risk flags
  5. decision_engine.decide() — fuse everything into final claim_status
  6. Assemble output row with exact 14-column schema
```

Key design decisions:
- **Per-image VLM calls** (not multi-image) for granular supporting_image_ids
- **Rule-based claim parser** with optional LLM refinement for ambiguous cases
- **Closed vocabulary validation** on all outputs — model outputs normalized via alias matching
- **Content-addressed caching** of VLM responses by image hash — re-runs are instant
- **Provider fallback chain:** Groq → OpenRouter → error

## Model(s) used and why

- **Primary:** Groq `meta-llama/llama-4-scout-17b-16e-instruct` — free tier, vision-capable, ~10-30s per image
- **Fallback:** OpenRouter `nex-agi/nex-n2-pro:free` — free tier, used when Groq rate-limited
- Both are free ($0 cost) which meets the hackathon constraint

## Decision hierarchy

Following the spec (TASK_BRIEF.md §4):

1. **Vision first** — What do the images actually show? (issue_type, object_part, severity from pixels)
2. **Conversation defines scope** — What is the user claiming? (from claim_parser)
3. **Evidence requirements gate** — Are the images sufficient to evaluate? (evidence_checker)
4. **History adds context** — Risk flags from user_history.csv, but **never overrides** clear visual evidence

The decision_engine.py enforces this: claim_status is determined purely by vision + evidence. History/risk flags can add `manual_review_required` or `user_history_risk` but cannot flip a clearly supported/contradicted claim.

## Handling adversarial cases

- **Prompt injection:** Regex patterns in claim_parser.py detect embedded instructions (e.g., "ignore previous instructions", "mark this as supported"). Flagged as `text_instruction_present`.
- **Manipulated images:** VLM authenticity_notes + image quality heuristics detect potential manipulation
- **Wrong object/part:** Detected by comparing claimed vs. vision-detected object_part → `wrong_object_part` flag
- **Untrusted user_claim:** Never used in system prompts or to control branching logic — only analyzed as data

## Evaluation summary

On `dataset/sample_claims.csv` (20 labeled samples):

| Field | Accuracy |
|---|---|
| claim_status | 35.0% (7/20) |
| issue_type | 20.0% (4/20) |
| object_part | 55.0% (11/20) |
| severity | 40.0% (8/20) |
| evidence_standard_met | 85.0% (17/20) |
| valid_image | 90.0% (18/20) |

See `evaluation/evaluation_report.md` for full metrics, confusion matrix, and operational analysis.

## Known limitations

- Free vision model sometimes misidentifies damage types
- Hinglish text handling is regex-based, not comprehensive
- No multi-image batching (more expensive but more accurate per-image IDs)
- Evidence requirements checked qualitatively, not with image understanding
