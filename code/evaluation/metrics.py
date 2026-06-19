"""
Scoring logic for evaluation/main.py. Pure functions -- no model calls,
no I/O beyond what's passed in.
"""

from config.schema import OUTPUT_COLUMNS

# Categorical fields to score individually
CATEGORICAL_FIELDS = [
    "claim_status",
    "issue_type",
    "object_part",
    "severity",
    "evidence_standard_met",
    "valid_image",
]

# Multi-value fields (semicolon-separated)
MULTI_VALUE_FIELDS = [
    "risk_flags",
    "supporting_image_ids",
]


def score(predictions: list[dict], expected: list[dict]) -> dict:
    """
    Score predictions against expected outputs.
    Returns a dict with per-field accuracy, claim_status confusion matrix,
    and multi-value field overlap metrics.
    """
    n = len(predictions)
    if n == 0:
        return {"error": "no predictions"}

    results = {
        "n_samples": n,
        "per_field_accuracy": {},
        "claim_status_confusion": {},
        "multi_value_overlap": {},
    }

    # Per-field accuracy for categorical columns
    for field in CATEGORICAL_FIELDS:
        correct = sum(
            1 for p, e in zip(predictions, expected)
            if p.get(field, "").lower() == e.get(field, "").lower()
        )
        results["per_field_accuracy"][field] = {
            "correct": correct,
            "total": n,
            "accuracy": round(correct / n, 4),
        }

    # Confusion matrix for claim_status
    statuses = ["supported", "contradicted", "not_enough_information"]
    confusion = {s: {t: 0 for t in statuses} for s in statuses}
    for p, e in zip(predictions, expected):
        actual = p.get("claim_status", "").lower()
        exp = e.get("claim_status", "").lower()
        if actual in confusion and exp in confusion[actual]:
            confusion[actual][exp] += 1
    results["claim_status_confusion"] = confusion

    # Multi-value field overlap (Jaccard similarity)
    for field in MULTI_VALUE_FIELDS:
        jaccard_scores = []
        for p, e in zip(predictions, expected):
            pred_set = set(s.strip().lower() for s in p.get(field, "").split(";") if s.strip() and s.strip().lower() != "none")
            exp_set = set(s.strip().lower() for s in e.get(field, "").split(";") if s.strip() and s.strip().lower() != "none")
            if not pred_set and not exp_set:
                jaccard = 1.0  # both empty = perfect match
            elif not pred_set or not exp_set:
                jaccard = 0.0
            else:
                intersection = pred_set & exp_set
                union = pred_set | exp_set
                jaccard = len(intersection) / len(union)
            jaccard_scores.append(jaccard)
        avg_jaccard = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0
        results["multi_value_overlap"][field] = {
            "avg_jaccard": round(avg_jaccard, 4),
            "scores": [round(s, 4) for s in jaccard_scores],
        }

    return results


def format_report_section(results: dict) -> str:
    """Render results as a markdown section for evaluation_report.md."""
    lines = []
    lines.append(f"## Metrics on `dataset/sample_claims.csv` ({results.get('n_samples', '?')} samples)")
    lines.append("")

    # Per-field accuracy
    lines.append("### Per-field accuracy")
    lines.append("")
    lines.append("| Field | Correct | Total | Accuracy |")
    lines.append("|---|---|---|---|")
    for field, acc in results.get("per_field_accuracy", {}).items():
        lines.append(f"| {field} | {acc['correct']} | {acc['total']} | {acc['accuracy']:.1%} |")
    lines.append("")

    # Confusion matrix
    lines.append("### `claim_status` confusion matrix")
    lines.append("")
    confusion = results.get("claim_status_confusion", {})
    statuses = ["supported", "contradicted", "not_enough_information"]
    header = "| Predicted → | " + " | ".join(statuses) + " |"
    separator = "|---|---|---|---|"
    lines.append(header)
    lines.append(separator)
    for actual in statuses:
        row = f"| **{actual}** |"
        for exp in statuses:
            row += f" {confusion.get(actual, {}).get(exp, 0)} |"
        lines.append(row)
    lines.append("")

    # Multi-value overlap
    lines.append("### Multi-value field overlap (Jaccard similarity)")
    lines.append("")
    lines.append("| Field | Avg Jaccard |")
    lines.append("|---|---|")
    for field, overlap in results.get("multi_value_overlap", {}).items():
        lines.append(f"| {field} | {overlap['avg_jaccard']:.4f} |")
    lines.append("")

    return "\n".join(lines)
