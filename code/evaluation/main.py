"""
Evaluation entry point. Runs the pipeline on dataset/sample_claims.csv
(which has expected outputs) and scores the predictions, then writes
evaluation/evaluation_report.md.
"""

import argparse
import json
import time
from pathlib import Path

# Load environment variables BEFORE any config imports
from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

from config import settings
from core import data_loader, pipeline
from evaluation import metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the pipeline against sample_claims.csv.")
    parser.add_argument("--input", type=str, default=str(settings.SAMPLE_CLAIMS_CSV))
    parser.add_argument("--output", type=str, default=str(Path("../dataset/sample_output.csv")))
    return parser.parse_args()


def main():
    args = parse_args()
    start = time.time()

    print(f"Loading data from {args.input}...")
    records = data_loader.load_claims_csv(Path(args.input))
    evidence_requirements = data_loader.load_evidence_requirements(settings.EVIDENCE_REQUIREMENTS_CSV)
    user_history = data_loader.load_user_history(settings.USER_HISTORY_CSV)

    print(f"  {len(records)} claims, {len(user_history)} user histories, {len(evidence_requirements)} evidence rules")

    # Run pipeline
    print("Running pipeline...")
    predictions = pipeline.run_batch(records, evidence_requirements, user_history)

    # Extract expected outputs from records
    expected = []
    for record in records:
        if record.expected:
            expected.append(record.expected)
        else:
            expected.append({})

    # Score
    print("Scoring predictions...")
    results = metrics.score(predictions, expected)

    # Print summary
    print(f"\nResults ({results.get('n_samples', '?')} samples):")
    for field, acc in results.get("per_field_accuracy", {}).items():
        print(f"  {field}: {acc['accuracy']:.1%} ({acc['correct']}/{acc['total']})")

    # Write evaluation report
    report_path = settings.EVAL_REPORT_MD
    report_section = metrics.format_report_section(results)

    # Read existing report template
    if report_path.exists():
        existing = report_path.read_text(encoding="utf-8")
    else:
        existing = "# Evaluation Report\n\n"

    # Replace the metrics section
    lines = existing.split("\n")
    new_lines = []
    in_metrics = False
    for line in lines:
        if line.startswith("## 2. Metrics on"):
            in_metrics = True
            new_lines.append(report_section)
            continue
        if in_metrics and line.startswith("## "):
            in_metrics = False
        if not in_metrics:
            new_lines.append(line)

    report_path.write_text("\n".join(new_lines), encoding="utf-8")
    print(f"\nEvaluation report written to {report_path}")

    elapsed = time.time() - start
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
