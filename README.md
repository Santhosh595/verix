# Verix

<div align="center">

# **Verix**
### Multi-Modal Evidence Review for Insurance Damage Claims

**Built for the HackerRank Orchestrate Hackathon**

</div>

---

## 🚀 Overview

**Verix** is an evidence-first adjudication pipeline for **car, laptop, and package damage claims**.

It treats **photographic evidence as ground truth**, then evaluates it against:
- claim conversation,
- evidence requirements, and
- user history,

to produce a **structured, explainable decision** with a transparent reasoning trail.

---

## 🎯 The Core Problem

Claims adjudication typically depends on three noisy signals:
1. what the customer says,
2. what the photos show,
3. what customer history suggests.

Naively trusting claim text creates two risks:
- honest misdescription,
- adversarial manipulation (including prompt-injection style text).

Verix enforces a strict evidence hierarchy:

> **Visual evidence > claim text**
>
> Claim text is never treated as fact and never used to control branching logic.

User history can raise risk flags and nudge a claim to manual review — but it **cannot override** a decision already established by visual evidence.

---

## 🧠 How Verix Works

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
5. Decision engine       — enforces hierarchy, assigns claim_status,
                           confidence, severity, and builds justification trail
        │
        ▼
16-column output.csv
```

User claim text is never concatenated into system prompts and never used as a branching signal.  
Prompt-injection-like phrases (for example, *"ignore previous instructions"*) are detected by `claim_parser.py` and surfaced as `text_instruction_present` — logged, never acted on.

✅ **Adversarial robustness:** 10/10 adversarial test cases pass, with zero false positives on genuine claims (including Hinglish input).

---

## 📤 Output Schema

`output.csv` contains **16 columns** (one row per claim):

| Column | Purpose |
|---|---|
| `user_id`, `image_paths`, `user_claim`, `claim_object` | Input identifiers |
| `evidence_standard_met`, `evidence_standard_met_reason` | Whether enough usable evidence was submitted |
| `risk_flags` | History/quality signals (advisory only) |
| `issue_type`, `object_part` | Extracted/confirmed claim details |
| `claim_status` | `supported` / `contradicted` / `not_enough_information` |
| `claim_status_justification` | Full reasoning trail: vision finding → claim comparison → evidence status → risk flags → which signal won and why |
| `supporting_image_ids` | Which images back the decision |
| `valid_image`, `valid_image_reason` | Per-claim image usability assessment |
| `confidence` | `high` / `medium` / `low` based on signal quality and agreement |
| `severity` | Estimated damage severity |

`claim_status_justification` is the transparency layer: each row explains what the model saw, how it compared with the claim text, whether evidence standards were met, and why the final decision was made.

---

## 🧩 Model Strategy

| Provider | Role | Notes |
|---|---|---|
| Gemini | Originally intended primary | Hit persistent 429s on free tier |
| Groq (`llama-4-scout-17b`) | Primary | Free tier, ~10–30s/claim |
| OpenRouter (`nex-n2-pro:free`) | Fallback | 50 req/day free tier |

- Calls are cached by content hash to avoid re-billing identical images.
- Includes 2 retries with automatic provider fallback.

---

## 📊 Evaluation (Real, Unfiltered Metrics)

Evaluated on `dataset/sample_claims.csv`.

- Total claims in sample: **20**
- Due to free-tier quota exhaustion after a broader 44-claim run, only **15 completed**.
- Metrics are reported exactly as observed (no cherry-picking).

| Field | Accuracy |
|---|---|
| `claim_status` | 35.0% (7/20 attempted) |
| `issue_type` | 25.0% |
| `object_part` | 55.0% |
| `severity` | 20.0% |
| `evidence_standard_met` | 75.0% |
| `valid_image` | 75.0% |

The `claim_status` confusion matrix shows dispersion across adjacent classes (e.g., `supported` vs `not_enough_information`) rather than one-directional collapse — consistent with free-tier vision limits on fine-grained damage discrimination.

> We intentionally report real performance and operational failures (including rate-limit attrition) for transparency.

For full breakdowns, strategy comparisons, latency/cost estimates, and limitations:  
`code/evaluation/evaluation_report.md`

---

## 🗂️ Project Structure

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
    │   ├── decision_engine.py    — hierarchy enforcement + justification builder
    │   └── pipeline.py
    ├── prompts/        — system / claim-extraction / vision prompts
    ├── utils/          — csv_io, image_utils, caching, logging
    ├── evaluation/     — main.py, metrics.py, compare_strategies.py, evaluation_report.md
    └── tests/          — test_pipeline.py, test_adversarial.py
```

---

## ⚙️ Running Verix

```bash
cd code
cp .env.example .env   # fill GROQ_API_KEY / OPENROUTER_API_KEY / GEMINI_API_KEY
pip install -r requirements.txt
python main.py
```

Output is written to `code/output.csv` and validated against the exact 16-column schema in `config/schema.py`.

### Run tests

```bash
pytest tests/
```

### Compare provider strategies

```bash
python evaluation/compare_strategies.py
```

---

## ⚠️ Known Limitations

- Free-tier vision models struggle with fine-grained damage categorization.
- Free-tier rate limits (Groq: 500K tokens/day; OpenRouter: 50 requests/day) caused real evaluation attrition.
- Hinglish claim-text handling is regex-based, not a full multilingual NLU stack.
- Images are processed independently per claim (no cross-image reasoning/batching yet).
- Evidence standards are rule-based thresholds, not learned from image content.

---

## 🧪 Project Status

**Prototype (Hackathon Submission)**

Verix is a **decision-support system** intended to assist human reviewers — not replace human oversight in final claim adjudication.
