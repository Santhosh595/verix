"""
Reading/writing CSVs with the exact required schema and column order.
Keep this dumb and reusable -- no business logic here.
"""

import csv
import logging
from pathlib import Path

from config.schema import OUTPUT_COLUMNS, validate_row

logger = logging.getLogger(__name__)


def read_csv_rows(path: Path) -> list[dict]:
    """
    Read a CSV file and return a list of plain dicts (one per row).
    Uses csv.DictReader so column order in the file does not matter --
    we only care about the header names. Strips whitespace from keys
    and values. Raises FileNotFoundError if the path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, raw_row in enumerate(reader, start=2):  # start=2: row 1 is header
            # Strip whitespace from keys and values
            row = {
                k.strip(): v.strip() if v is not None else ""
                for k, v in raw_row.items()
                if k is not None
            }
            row["_source_row"] = i
            rows.append(row)

    logger.info("Read %d rows from %s", len(rows), path)
    return rows


def write_output_csv(rows: list[dict], path: Path) -> None:
    """
    Write rows to a CSV file using the exact OUTPUT_COLUMNS order.
    Validates every row before writing -- raises ValueError with all
    validation errors collected, so nothing half-written gets committed
    to disk.
    """
    # Validate all rows first
    all_errors: list[str] = []
    for i, row in enumerate(rows):
        # Strip internal tracking field before validation
        clean_row = {k: v for k, v in row.items() if not k.startswith("_")}
        errors = validate_row(clean_row)
        if errors:
            all_errors.extend([f"Row {i + 1}: {e}" for e in errors])

    if all_errors:
        error_block = "\n".join(all_errors)
        raise ValueError(
            f"CSV validation failed with {len(all_errors)} error(s):\n{error_block}"
        )

    # Write with exact column order
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=OUTPUT_COLUMNS,
            extrasaction="ignore",  # we already validated; this is defense-in-depth
        )
        writer.writeheader()
        for row in rows:
            clean_row = {k: v for k, v in row.items() if not k.startswith("_")}
            writer.writerow(clean_row)

    logger.info("Wrote %d rows to %s", len(rows), path)
