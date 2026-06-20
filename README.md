# Verix

<div align="center">

### Multi-Modal Evidence Review for Insurance Damage Claims

Evidence-first claim adjudication for **car**, **laptop**, and **package** damage.  
Built for the **HackerRank Orchestrate Hackathon**.

</div>

---

## Overview

**Verix** is a claim adjudication pipeline that treats **photographic evidence as the primary source of truth**.  
It evaluates images against:

- claim conversation,
- evidence requirements, and
- user history,

to produce a **structured, explainable decision** with a transparent reasoning trail.

---

## Problem Statement

Insurance claim adjudication often relies on three inconsistent signals:

1. what the customer says,
2. what photos show,
3. what customer history suggests.

If claim text is trusted naively, systems become vulnerable to:
- honest misdescription, and
- adversarial manipulation (e.g., prompt-injection-like instructions in user text).

Verix addresses this with a strict evidence hierarchy:

> **Visual evidence > claim text**  
> Claim text is never treated as factual evidence and never used to control branching logic.

User history can raise risk flags and nudge a case toward manual review, but it **cannot overturn** a decision already supported by visual evidence.

---

## How It Works

```text
claim record + images + user history + evidence requirements
        │
        ▼
1. Claim parsing         — extracts issue_type / object_part from claim text;
                           regex + LLM-based prompt-injection detection
        │
        ▼
2. Vision analysis       — VLM inspects each image independently
                           (Gemini → Groq → OpenRouter fallback chain)
        │
        ▼
3. Evidence checking     — image count/quality vs. evidence_requirements.csv
        │
        ▼
4. Risk assessment       — user_history signals → risk_flags (advisory only)
        │
        ▼
5. Decision engine       — applies hierarchy, assigns claim_status,
                           confidence, severity, and builds the
                           justification trail
        │
        ▼
16-column output.csv
```

### Security / Robustness Behavior

- User claim text is **never concatenated into system prompts**.
- User claim text is **never used as a branching signal**.
- Prompt-injection-like text (e.g., *“ignore previous instructions”*) is detected in `claim_parser.py` and surfaced as `text_instruction_present` — **logged, never acted on**.

✅ **Adversarial tests:** 10/10 pass with zero false positives on genuine claims (including Hinglish input).

---

## Output Schema

`output.csv` has **16 columns** (one row per claim):

| Column | Purpose |
|---|---|
| `user_id`, `image_paths`, `user_claim`, `claim_object` | Input identifiers |
| `evidence_standard_met`, `evidence_standard_met_reason` | Whether enough usable evidence was submitted |
| `risk_flags` | History/quality signals (advisory only) |
| `issue_type`, `object_part` | Extracted/confirmed claim details |
| `claim_status` | `supported` / `contradicted` / `not_enough_information` |
| `claim_status_justification` | Full reasoning trail: vision finding → claim comparison → evidence status → risk flags → which signal won and why |
| `supporting_image_ids` | Images that support the decision |
| `valid_image`, `valid_image_reason` | Per-claim image usability assessment |
| `confidence` | `high` / `medium` / `low` based on signal quality and agreement |
| `severity` | Estimated damage severity |

`claim_status_justification` is the transparency layer: every decision explains what vision detected, how it compared to claim text, whether evidence standards were met, and why the final signal won.

---

## Model Strategy

| Provider | Role | Notes |
|---|---|---|
| Groq (`llama-4-scout-17b`) | Primary | Free tier, ~10–30s/claim |
| OpenRouter (`nex-n2-pro:free`) | Fallback | 50 req/day free tier |

- On rate limit/error from Groq, Verix automatically falls back to OpenRouter.
- If both providers fail, the claim is returned as `not_enough_information` with a logged error (batch continues safely).
- Calls are cached by `(image_content_sha256, prompt_version, provider)` to avoid re-billing identical images on reruns.

---

## Evaluation (Real Metrics)

Note: this is a separate, ground-truth-labeled scoring run on `dataset/sample_claims.csv` (20 claims), distinct from the full `output.csv` (44/44 claims, complete). The eval run happened after the 44-claim batch and ran into leftover free-tier rate limits — see below.

Run on `dataset/sample_claims.csv` (20 claims; 5 failed on 429s from exhausted free-tier quota after the full 44-claim run — reported on the 15 that completed):

- Total claims: **20**
- After a larger 44-claim run, free-tier quota exhaustion caused 5 claims to fail (429s)
- Results are reported transparently from attempted/completed runs

| Field | Accuracy |
|---|---|
| `claim_status` | 35.0% (7/20 attempted) |
| `issue_type` | 25.0% |
| `object_part` | 55.0% |
| `severity` | 20.0% |
| `evidence_standard_met` | 75.0% |
| `valid_image` | 75.0% |

The `claim_status` confusion matrix shows spread across adjacent classes (e.g., `supported` misread as `not_enough_information`) rather than one-directional collapse, which is consistent with free-tier vision limitations on fine-grained damage discrimination.

> We intentionally report real-world performance and operational attrition instead of cherry-picking cleaner runs.

For full breakdown, strategy comparison, latency/cost estimates, and limitations, see:  
`code/evaluation/evaluation_report.md`

---

## Project Structure

```text
verix/
├── AGENTS.md, README.md, problem_statement.md
├── prompt/TASK_BRIEF.md
├── dataset/
│   ├── claims.csv, sample_claims.csv, user_history.csv
│   ├── evidence_requirements.csv
│   └── images/{sample,test}/
└── code/
    ├── main.py
    ├── config/         — schema.py (column + vocab definitions), settings.py
    ├── core/
    │   ├── data_loader.py, claim_parser.py, vision_analyzer.py
    │   ├── evidence_checker.py, risk_assessor.py
    │   ├── decision_engine.py   — hierarchy enforcement + justification builder
    │   └── pipeline.py
    ├── prompts/        — system / claim-extraction / vision prompts
    ├── utils/          — csv_io, image_utils, caching, logging
    ├── evaluation/     — main.py, metrics.py, compare_strategies.py, evaluation_report.md
    └── tests/          — test_pipeline.py, test_adversarial.py
```

---

## Running Verix

```bash
cd code
cp .env.example .env   # fill in GROQ_API_KEY / OPENROUTER_API_KEY / GEMINI_API_KEY
pip install -r requirements.txt
python main.py
```

Output is written to `code/output.csv` and validated against the exact 16-column schema in `config/schema.py`.

### Run tests

```bash
pytest tests/
```

### Run strategy comparison

```bash
python evaluation/compare_strategies.py
```

---

## Known Limitations

- Free-tier vision models misidentify fine-grained damage types more often than paid alternatives.
- Free-tier rate limits (Groq: 500K tokens/day, OpenRouter: 50 req/day) caused real evaluation attrition.
- Hinglish claim-text handling is regex-based, not a full NLU pipeline.
- Images are scored independently per claim (no cross-image batching yet).
- Evidence requirement rules are qualitative thresholds, not learned directly from image content.

---

## Status

**Prototype** built for hackathon submission.

Verix is a **decision-support system** intended to assist human reviewers, not to auto-adjudicate claims without oversight.
