"""
Loads and joins all input CSVs, resolving image_paths to real files on
disk. This is the single place that knows how to turn a raw claims.csv
row into everything the rest of the pipeline needs.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ClaimRecord:
    user_id: str
    image_paths: list[Path]          # resolved, existing paths only
    missing_image_paths: list[str]   # paths listed but not found on disk
    user_claim: str
    claim_object: str
    expected: dict | None = None     # populated only for sample_claims.csv rows
    raw_row: dict = field(default_factory=dict)


def _is_sample_csv(path: Path) -> bool:
    """True if the CSV file is sample_claims.csv (has expected output columns)."""
    return "sample" in path.name.lower()


def load_claims_csv(path: Path) -> list[ClaimRecord]:
    """
    Read claims.csv or sample_claims.csv and return a list of ClaimRecord.
    - Splits image_paths on ';' and resolves each relative path against the
      dataset root (parent of the images/ folder).
    - Records any path that doesn't resolve to an existing file in
      missing_image_paths (this matters for valid_image / risk_flags).
    - If reading sample_claims.csv, populate `expected` with the labeled
      output columns for that row.
    """
    from utils.csv_io import read_csv_rows

    dataset_dir = path.parent
    rows = read_csv_rows(path)
    is_sample = _is_sample_csv(path)

    # Output columns that represent expected labels in sample_claims.csv
    expected_columns = [
        "evidence_standard_met",
        "evidence_standard_met_reason",
        "risk_flags",
        "issue_type",
        "object_part",
        "claim_status",
        "claim_status_justification",
        "supporting_image_ids",
        "valid_image",
        "severity",
    ]

    records: list[ClaimRecord] = []
    for row in rows:
        # Split and resolve image paths
        raw_paths = row.get("image_paths", "")
        image_paths: list[Path] = []
        missing_paths: list[str] = []

        if raw_paths.strip():
            for rel in raw_paths.split(";"):
                rel = rel.strip()
                if not rel:
                    continue
                resolved = dataset_dir / rel
                if resolved.exists():
                    image_paths.append(resolved)
                else:
                    missing_paths.append(rel)

        # Build expected dict for sample rows
        expected: dict | None = None
        if is_sample:
            expected = {}
            for col in expected_columns:
                if col in row:
                    expected[col] = row[col]

        record = ClaimRecord(
            user_id=row.get("user_id", "").strip(),
            image_paths=image_paths,
            missing_image_paths=missing_paths,
            user_claim=row.get("user_claim", "").strip(),
            claim_object=row.get("claim_object", "").strip().lower(),
            expected=expected,
            raw_row=row,
        )
        records.append(record)

    return records


def load_user_history(path: Path) -> dict[str, dict]:
    """
    Return {user_id: {past_claim_count, accept_claim, ...}} keyed by user_id.
    Numeric fields are parsed to int; history_flags left as string.
    """
    from utils.csv_io import read_csv_rows

    rows = read_csv_rows(path)
    history: dict[str, dict] = {}

    numeric_fields = {
        "past_claim_count",
        "accept_claim",
        "manual_review_claim",
        "rejected_claim",
        "last_90_days_claim_count",
    }

    for row in rows:
        uid = row.get("user_id", "").strip()
        if not uid:
            continue
        parsed: dict = {}
        for k, v in row.items():
            if k == "user_id":
                continue
            if k in numeric_fields:
                try:
                    parsed[k] = int(v)
                except (ValueError, TypeError):
                    parsed[k] = 0
            else:
                parsed[k] = v
        history[uid] = parsed

    return history


# Module-level cache for evidence requirements
_evidence_reqs_cache: list[dict] | None = None


def load_evidence_requirements(path: Path) -> list[dict]:
    """
    Return parsed rows of evidence_requirements.csv. Results are cached at
    module level since this file is static during a run.
    """
    global _evidence_reqs_cache
    if _evidence_reqs_cache is not None:
        return _evidence_reqs_cache

    from utils.csv_io import read_csv_rows

    rows = read_csv_rows(path)
    _evidence_reqs_cache = rows
    return rows


def lookup_evidence_requirement(
    claim_object: str,
    issue_family: str,
) -> dict | None:
    """
    Look up the evidence requirement matching a (claim_object, issue_family).
    Prefers an object-specific rule over one with claim_object == "all".
    Returns the matching row dict or None if nothing matches.

    Matching is case-insensitive and tolerant of the issue_family string
    (e.g. "dent or scratch" matches a query of "dent").
    """
    reqs = load_evidence_requirements(
        Path(__file__).resolve().parents[2] / "dataset" / "evidence_requirements.csv"
    )

    claim_object = claim_object.lower().strip()
    issue_family = issue_family.lower().strip()

    # First pass: look for object-specific rule (not "all")
    best_match: dict | None = None
    for req in reqs:
        req_obj = req.get("claim_object", "").lower().strip()
        req_applies = req.get("applies_to", "").lower().strip()

        if req_obj == claim_object or req_obj == "all":
            # Check issue family overlap (substring match is enough)
            if (
                issue_family in req_applies
                or req_applies in issue_family
                or _issue_family_overlap(issue_family, req_applies)
            ):
                if req_obj != "all":
                    return req  # exact object match wins
                if best_match is None:
                    best_match = req  # "all" fallback

    return best_match


def _issue_family_overlap(a: str, b: str) -> bool:
    """
    True if two issue-family strings share any significant word.
    e.g. "dent or scratch" and "dent" -> True
    """
    words_a = {w.strip() for w in a.replace("or", ",").split(",") if w.strip()}
    words_b = {w.strip() for w in b.replace("or", ",").split(",") if w.strip()}
    return bool(words_a & words_b)
