# Multi-Modal Evidence Review — HackerRank Orchestrate June 2026

## Overview

A damage-claim verification pipeline that fuses computer vision (VLM), natural language extraction, evidence-rule gating, and user-history risk scoring to decide whether submitted images **support**, **contradict**, or provide **not enough information** about a user's claim.

## Quick Start

```bash
cd code
pip install -r requirements.txt
# Add API keys to .env (see .env.example)
python main.py                          # Process dataset/claims.csv → output.csv
python evaluation/main.py               # Evaluate on sample_claims.csv
PYTHONPATH=. python tests/test_adversarial.py  # Run adversarial robustness tests
```

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Input CSVs │────▶│  data_loader.py  │────▶│  ClaimRecord[]  │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
┌──────────────────────────────────────────────────────────────┐
│                     pipeline.run_claim()                      │
│                                                              │
│  ┌────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │ claim_parser   │  │ vision_analyzer  │  │ risk_assessor│ │
│  │ (text → claim) │  │ (VLM per image)  │  │ (flags)      │ │
│  └───────┬────────┘  └───────┬──────────┘  └──────┬───────┘ │
│          │                   │                     │         │
│          ▼                   ▼                     ▼         │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              evidence_checker                           │  │
│  │  (evidence_requirements.csv gate)                       │  │
│  └───────────────────────┬────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              decision_engine                            │  │
│  │  (vision > claim > evidence > hierarchy)                │  │
│  └───────────────────────┬────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│                   Output Row (16 columns)                    │
└──────────────────────────────────────────────────────────────┘
```

## Design Decisions

### Decision Hierarchy (enforced in `decision_engine.py`)

1. **Vision evidence is ground truth** — What the VLM sees in images determines issue_type, object_part, and severity
2. **Claim text defines scope** — Used to know *what to check for* and detect mismatches
3. **Evidence requirements gate** — `evidence_requirements.csv` determines if images are sufficient
4. **User history adds context only** — Can raise `user_history_risk` or `manual_review_required` flags but **never overrides** clear visual evidence

The `claim_status_justification` field explicitly shows this conflict resolution:
> "Vision evidence (2/3 images clear): shows dent on rear_bumper in img_1; img_2. | User claim: 'dent on rear_bumper'. Vision supports the claim. | Evidence standard met: The claimed car panel or bumper should be visible from an angle where surface marks can be assessed. | Risk flags: none."

### Calibrated Confidence

Each decision includes a `confidence` field (`high`/`medium`/`low`) based on:
- **high**: Clean images, vision agrees, no risk flags
- **medium**: Quality issues (blur/glare), multiple issue types detected, or history risk flags present
- **low**: Evidence not met, all images poor quality, or model confidence < 0.5

### Provider Fallback Chain

```
Primary: Groq (meta-llama/llama-4-scout-17b-16e-instruct)
  ↓ (on rate limit or error)
Fallback: OpenRouter (nex-agi/nex-n2-pro:free)
  ↓ (on rate limit or error)
Error: graceful failure with not_enough_information status
```

### Caching

VLM responses are cached by `(image_content_sha256, prompt_version, provider)` in `.cache/`. Re-running the pipeline on the same dataset is instant for previously processed images.

## Output Schema (16 columns)

| Column | Description |
|---|---|
| user_id | From input claims.csv |
| image_paths | Semicolon-separated paths |
| user_claim | Raw conversation text |
| claim_object | car / laptop / package |
| evidence_standard_met | true / false |
| evidence_standard_met_reason | Human-readable explanation |
| risk_flags | Semicolon-separated flags or "none" |
| issue_type | Visible damage type (closed vocabulary) |
| object_part | Relevant object part (closed vocabulary) |
| claim_status | supported / contradicted / not_enough_information |
| claim_status_justification | Conflict-resolution reasoning trail |
| supporting_image_ids | Image IDs supporting the decision |
| valid_image | true / false |
| valid_image_reason | Explanation for valid_image |
| confidence | high / medium / low |
| severity | none / low / medium / high / unknown |

## Failure Modes

| Scenario | Behavior |
|---|---|
| Both providers rate-limited | Claim returns `not_enough_information` with fallback row; error logged |
| Image file not found | Recorded in `missing_image_paths`; may fail evidence gate |
| All images unreadable/blurry | `valid_image=false`, `confidence=low`, status=`not_enough_information` |
| VLM returns unparseable JSON | Falls back to `unknown` issue/part; `confidence=low` |
| Prompt injection in user_claim | Detected via regex; flagged as `text_instruction_present`; never executed |
| Missing user history | Not treated as risky; no `user_history_risk` flag added |
| Evidence requirements not met | `evidence_standard_met=false`; status=`not_enough_information` |

## Adversarial Robustness

The system detects and flags prompt-injection-style phrasing in `user_claim` text. Tested against:
- Direct injection ("ignore previous instructions", role overrides, policy overrides
- Photo-claim manipulation ("the photos show clearly", "damage is obvious")
- Normal claims in English and Hinglish (not falsely flagged)

See `tests/test_adversarial.py` for the full test suite.

## Evaluation Results

Run on `dataset/sample_claims.csv` (20 labeled samples):

| Field | Accuracy | Notes |
|---|---|---|
| claim_status | 35.0% | Free vision model struggles with fine-grained damage types |
| issue_type | 25.0% | Confuses similar types (e.g., broken_part vs dent) |
| object_part | 55.0% | Better at identifying car parts than damage types |
| severity | 20.0% | Subjective even for humans |
| evidence_standard_met | 75.0% | Binary true/false is easier for the model |
| valid_image | 75.0% | Most images are clearly usable |

Full evaluation report: `evaluation/evaluation_report.md`

## Known Limitations

- **Free model accuracy**: A paid VLM (e.g., GPT-4o, Gemini Pro) would significantly improve accuracy
- **Rate limits**: Free tiers limit throughput; production use requires paid API credits
- **Hinglish text**: Regex-based handling covers common terms but not all multilingual variants
- **No multi-image batching**: Each image analyzed separately (more expensive but provides per-image supporting IDs)
- **Evidence requirements**: Qualitative rules, not image-understanding-based assessment

## Tech Stack

- Python 3.11
- Groq SDK (vision via OpenAI-compatible API)
- OpenRouter SDK (fallback)
- PIL/Pillow (image processing)
- NumPy (quality heuristics)
- python-dotenv (configuration)
