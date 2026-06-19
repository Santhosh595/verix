"""
CLI entry point. Reads dataset/claims.csv (or --input), runs the
pipeline on every row, writes output.csv (or --output).

Usage:
    python main.py
    python main.py --input ../dataset/sample_claims.csv --output sample_output.csv --limit 5
"""

import argparse
import logging
import time
from pathlib import Path

# Load environment variables BEFORE any config imports
from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

from config import settings
from config.schema import OUTPUT_COLUMNS
from core import data_loader, pipeline
from utils import csv_io


def parse_args():
    parser = argparse.ArgumentParser(description="Run the multi-modal evidence review pipeline.")
    parser.add_argument("--input", type=str, default=str(settings.CLAIMS_CSV))
    parser.add_argument("--output", type=str, default=str(settings.OUTPUT_CSV))
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows (debugging).")
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main():
    setup_logging()
    args = parse_args()
    start = time.time()

    print(f"Loading data from {args.input}...")

    # 1. Load inputs
    records = data_loader.load_claims_csv(Path(args.input))
    evidence_requirements = data_loader.load_evidence_requirements(settings.EVIDENCE_REQUIREMENTS_CSV)
    user_history = data_loader.load_user_history(settings.USER_HISTORY_CSV)

    if args.limit:
        records = records[:args.limit]
        print(f"  (limited to first {args.limit} rows)")

    print(f"  {len(records)} claims, {len(user_history)} user histories, {len(evidence_requirements)} evidence rules")

    # 2. Run pipeline
    print(f"Running pipeline (provider: {settings.VLM_PROVIDER})...")
    rows = pipeline.run_batch(records, evidence_requirements, user_history)

    # 3. Write output
    output_path = Path(args.output)
    csv_io.write_output_csv(rows, output_path)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Processed: {len(rows)} claims")
    print(f"  Output: {output_path}")

    # Quick summary
    statuses = {}
    for row in rows:
        s = row.get("claim_status", "unknown")
        statuses[s] = statuses.get(s, 0) + 1
    print(f"  Breakdown: {statuses}")


if __name__ == "__main__":
    main()
