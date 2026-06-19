"""
Single source of truth for the output schema and closed vocabularies.

Every other module should import from here rather than re-typing column
names or allowed values. Keep this in sync with problem_statement.md if
it is ever clarified/updated.
"""

# Exact output column order required by the spec.
OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
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

CLAIM_OBJECTS = ["car", "laptop", "package"]

CLAIM_STATUSES = ["supported", "contradicted", "not_enough_information"]

ISSUE_TYPES = [
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown",
]

OBJECT_PARTS = {
    "car": [
        "front_bumper",
        "rear_bumper",
        "door",
        "hood",
        "windshield",
        "side_mirror",
        "headlight",
        "taillight",
        "fender",
        "quarter_panel",
        "body",
        "unknown",
    ],
    "laptop": [
        "screen",
        "keyboard",
        "trackpad",
        "hinge",
        "lid",
        "corner",
        "port",
        "base",
        "body",
        "unknown",
    ],
    "package": [
        "box",
        "package_corner",
        "package_side",
        "seal",
        "label",
        "contents",
        "item",
        "unknown",
    ],
}

RISK_FLAGS = [
    "none",
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
]

SEVERITIES = ["none", "low", "medium", "high", "unknown"]

BOOL_STRINGS = ["true", "false"]


def closest_allowed_value(value: str, allowed: list[str], default: str = "unknown") -> str:
    """
    Normalize a value against an allowed list. Tries exact match first
    (case-insensitive, whitespace-stripped), then substring/alias match.
    Falls back to `default` if nothing matches closely enough -- never
    invents a new label.

    Examples:
        closest_allowed_value("Dent", ISSUE_TYPES) -> "dent"
        closest_allowed_value("scratched", ISSUE_TYPES) -> "scratch"
        closest_allowed_value("totally_wrong", ISSUE_TYPES) -> "unknown"
    """
    if not value or not isinstance(value, str):
        return default

    cleaned = value.strip().lower().replace("-", "_").replace(" ", "_")

    # Exact match (case-insensitive)
    for item in allowed:
        if item.lower() == cleaned:
            return item

    # Substring / alias matching
    aliases = {
        "dent": "dent",
        "dented": "dent",
        "ding": "dent",
        "scratch": "scratch",
        "scratched": "scratch",
        "scuff": "scratch",
        "scuffed": "scratch",
        "crack": "crack",
        "cracked": "crack",
        "shatter": "glass_shatter",
        "shattered": "glass_shatter",
        "broken": "broken_part",
        "break": "broken_part",
        "missing": "missing_part",
        "torn": "torn_packaging",
        "rip": "torn_packaging",
        "ripped": "torn_packaging",
        "crushed": "crushed_packaging",
        "smash": "crushed_packaging",
        "water": "water_damage",
        "wet": "water_damage",
        "moisture": "water_damage",
        "stain": "stain",
        "stained": "stain",
        "discolor": "stain",
        "none": "none",
        "no_damage": "none",
        "no issue": "none",
        "unknown": "unknown",
        "unidentified": "unknown",
        "front_bumper": "front_bumper",
        "front bumper": "front_bumper",
        "rear_bumper": "rear_bumper",
        "rear bumper": "rear_bumper",
        "door": "door",
        "hood": "hood",
        "bonnet": "hood",
        "windshield": "windshield",
        "windscreen": "windshield",
        "side_mirror": "side_mirror",
        "side mirror": "side_mirror",
        "mirror": "side_mirror",
        "headlight": "headlight",
        "head light": "headlight",
        "taillight": "taillight",
        "tail light": "taillight",
        "fender": "fender",
        "quarter_panel": "quarter_panel",
        "quarter panel": "quarter_panel",
        "body": "body",
        "screen": "screen",
        "display": "screen",
        "keyboard": "keyboard",
        "trackpad": "trackpad",
        "touchpad": "trackpad",
        "hinge": "hinge",
        "lid": "lid",
        "corner": "corner",
        "port": "port",
        "base": "base",
        "box": "box",
        "package_corner": "package_corner",
        "package_side": "package_side",
        "seal": "seal",
        "label": "label",
        "contents": "contents",
        "item": "item",
        "supported": "supported",
        "confirm": "supported",
        "confirmed": "supported",
        "contradicted": "contradicted",
        "reject": "contradicted",
        "rejected": "contradicted",
        "denied": "contradicted",
        "not_enough_information": "not_enough_information",
        "not enough": "not_enough_information",
        "insufficient": "not_enough_information",
        "inconclusive": "not_enough_information",
        "low": "low",
        "medium": "medium",
        "moderate": "medium",
        "high": "high",
        "severe": "high",
        "critical": "high",
        "true": "true",
        "yes": "true",
        "false": "false",
        "no": "false",
    }

    if cleaned in aliases:
        mapped = aliases[cleaned]
        # Verify the mapped value is actually in the allowed list
        if mapped in allowed:
            return mapped

    # Substring fallback: check if any allowed value is a substring of input or vice versa
    for item in allowed:
        item_lower = item.lower()
        if item_lower in cleaned or cleaned in item_lower:
            return item

    return default


def validate_claim_object(value: str) -> bool:
    """True if value is a valid claim_object."""
    if not value:
        return False
    return value.strip().lower() in CLAIM_OBJECTS


def validate_claim_status(value: str) -> bool:
    """True if value is a valid claim_status."""
    if not value:
        return False
    return value.strip().lower() in CLAIM_STATUSES


def validate_issue_type(value: str) -> bool:
    """True if value is a valid issue_type."""
    if not value:
        return False
    return value.strip().lower() in ISSUE_TYPES


def validate_object_part(claim_object: str, object_part: str) -> bool:
    """True if object_part is valid for the given claim_object."""
    if not claim_object or not object_part:
        return False
    allowed_parts = OBJECT_PARTS.get(claim_object.lower())
    if not allowed_parts:
        return False
    return object_part.strip().lower() in allowed_parts


def validate_risk_flags(flags: str) -> bool:
    """
    True if every semicolon-separated flag is valid, or if the value is 'none'.
    Empty string is treated as invalid.
    """
    if not flags or not flags.strip():
        return False
    if flags.strip().lower() == "none":
        return True
    for flag in flags.split(";"):
        if flag.strip().lower() not in RISK_FLAGS:
            return False
    return True


def validate_severity(value: str) -> bool:
    """True if value is a valid severity."""
    if not value:
        return False
    return value.strip().lower() in SEVERITIES


def validate_bool_string(value: str) -> bool:
    """True if value is 'true' or 'false' (case-insensitive)."""
    if not value:
        return False
    return value.strip().lower() in BOOL_STRINGS


def validate_row(row: dict) -> list[str]:
    """
    Validate a single output row against the schema. Returns a list of
    error messages (empty list = valid). Checks:
    - All required columns present, no extra columns
    - claim_object, claim_status, issue_type, severity in allowed vocab
    - object_part valid for claim_object
    - risk_flags parseable
    - evidence_standard_met and valid_image are 'true'/'false'
    """
    errors = []
    row_keys = set(row.keys())
    expected_keys = set(OUTPUT_COLUMNS)

    missing = expected_keys - row_keys
    if missing:
        errors.append(f"Missing columns: {sorted(missing)}")

    extra = row_keys - expected_keys
    if extra:
        errors.append(f"Extra columns: {sorted(extra)}")

    # Validate closed-vocabulary fields
    if "claim_object" in row and not validate_claim_object(row.get("claim_object", "")):
        errors.append(f"Invalid claim_object: {row.get('claim_object')!r}")

    if "claim_status" in row and not validate_claim_status(row.get("claim_status", "")):
        errors.append(f"Invalid claim_status: {row.get('claim_status')!r}")

    if "issue_type" in row and not validate_issue_type(row.get("issue_type", "")):
        errors.append(f"Invalid issue_type: {row.get('issue_type')!r}")

    if "severity" in row and not validate_severity(row.get("severity", "")):
        errors.append(f"Invalid severity: {row.get('severity')!r}")

    if "object_part" in row and "claim_object" in row:
        if not validate_object_part(row["claim_object"], row.get("object_part", "")):
            errors.append(
                f"Invalid object_part {row.get('object_part')!r} for claim_object {row['claim_object']!r}"
            )

    if "risk_flags" in row and not validate_risk_flags(row.get("risk_flags", "")):
        errors.append(f"Invalid risk_flags: {row.get('risk_flags')!r}")

    if "evidence_standard_met" in row and not validate_bool_string(
        row.get("evidence_standard_met", "")
    ):
        errors.append(f"Invalid evidence_standard_met: {row.get('evidence_standard_met')!r} (must be 'true' or 'false')")

    if "valid_image" in row and not validate_bool_string(row.get("valid_image", "")):
        errors.append(f"Invalid valid_image: {row.get('valid_image')!r} (must be 'true' or 'false')")

    return errors
