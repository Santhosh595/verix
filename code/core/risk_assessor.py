"""
Computes risk_flags from two sources:
  1. Image-level signals (quality, authenticity, mismatch vs. claimed
     object/part, embedded-instruction text).
  2. User history signals (dataset/user_history.csv).

Per the spec, history must only ADD context/flags -- it must never be the
sole reason a clearly-supported or clearly-contradicted claim_status gets
flipped. decision_engine.py enforces that hierarchy; this module just
surfaces the flags.
"""

import logging
from config.schema import RISK_FLAGS

logger = logging.getLogger(__name__)

# Quality note -> risk flag mapping
_QUALITY_TO_FLAG = {
    "blurry_image": "blurry_image",
    "blurry": "blurry_image",
    "cropped_or_obstructed": "cropped_or_obstructed",
    "cropped": "cropped_or_obstructed",
    "obstructed": "cropped_or_obstructed",
    "low_light_or_glare": "low_light_or_glare",
    "low light": "low_light_or_glare",
    "glare": "low_light_or_glare",
    "wrong_angle": "wrong_angle",
    "wrong angle": "wrong_angle",
    "wrong_object": "wrong_object",
    "wrong object": "wrong_object",
    "wrong_object_part": "wrong_object_part",
    "wrong object part": "wrong_object_part",
    "damage_not_visible": "damage_not_visible",
    "damage not visible": "damage_not_visible",
    "possible_manipulation": "possible_manipulation",
    "manipulation": "possible_manipulation",
    "non_original_image": "non_original_image",
    "screenshot": "non_original_image",
    "stock photo": "non_original_image",
    "text_instruction_present": "text_instruction_present",
}


def assess_image_risk(image_findings: list, claim_object: str, parsed_claim) -> set[str]:
    """
    Map ImageFinding.quality_notes / authenticity_notes into the closed
    risk_flags vocabulary. Also detects object/part mismatch between
    what was claimed and what the vision model saw.
    """
    flags: set[str] = set()

    if not image_findings:
        return flags

    # Collect all quality notes across findings
    all_quality_notes: set[str] = set()
    all_authenticity_notes: set[str] = set()
    all_object_parts_seen: set[str] = set()

    for finding in image_findings:
        for note in finding.quality_notes:
            all_quality_notes.add(note.lower().strip())
        for note in finding.authenticity_notes:
            all_authenticity_notes.add(note.lower().strip())
        if finding.object_part and finding.object_part != "unknown":
            all_object_parts_seen.add(finding.object_part)

    # Map quality notes to flags
    for note in all_quality_notes:
        flag = _QUALITY_TO_FLAG.get(note)
        if flag:
            flags.add(flag)

    # Map authenticity notes to flags
    for note in all_authenticity_notes:
        if "manipulat" in note or "edit" in note or "modified" in note:
            flags.add("possible_manipulation")
        if "screenshot" in note or "screen capture" in note:
            flags.add("non_original_image")
        if "stock" in note or "stock photo" in note:
            flags.add("non_original_image")
        if "original" in note and "not" in note:
            flags.add("non_original_image")

    # Object/part mismatch detection
    if parsed_claim and parsed_claim.claimed_object_part_guess:
        claimed_part = parsed_claim.claimed_object_part_guess
        if all_object_parts_seen and claimed_part not in all_object_parts_seen:
            flags.add("wrong_object_part")

    if parsed_claim and parsed_claim.looks_like_injection:
        flags.add("text_instruction_present")

    # Validate all flags against closed vocabulary
    valid_flags = set(RISK_FLAGS)
    flags = flags & valid_flags

    return flags


def assess_history_risk(user_history_row: dict | None) -> set[str]:
    """
    Turn elevated rejected_claim / manual_review_claim counts,
    high last_90_days_claim_count, or explicit history_flags into
    "user_history_risk" and/or "manual_review_required".

    Thresholds (documented in code/README.md):
    - rejected_claim >= 2 -> user_history_risk
    - manual_review_claim >= 2 -> manual_review_required
    - last_90_days_claim_count >= 3 -> user_history_risk
    - past_claim_count >= 5 AND rejected_claim ratio > 0.3 -> user_history_risk
    - history_flags field is not "none" -> user_history_risk
    """
    flags: set[str] = set()

    if user_history_row is None:
        return flags  # new user, not automatically risky

    rejected = user_history_row.get("rejected_claim", 0)
    manual = user_history_row.get("manual_review_claim", 0)
    last_90 = user_history_row.get("last_90_days_claim_count", 0)
    past = user_history_row.get("past_claim_count", 0)
    history_flags = user_history_row.get("history_flags", "none")

    # Explicit history flags
    if history_flags and history_flags.lower().strip() not in ("none", ""):
        flags.add("user_history_risk")

    # Rejected claims threshold
    if rejected >= 2:
        flags.add("user_history_risk")

    # Manual review threshold
    if manual >= 2:
        flags.add("manual_review_required")

    # High recent activity
    if last_90 >= 3:
        flags.add("user_history_risk")

    # High overall + high rejection ratio
    if past >= 5 and (rejected / max(past, 1)) > 0.3:
        flags.add("user_history_risk")

    return flags


def combine_risk_flags(*flag_sets: set[str]) -> str:
    """Union all sets, return 'none' if empty, else ';'-joined sorted flags."""
    combined: set[str] = set()
    for fs in flag_sets:
        combined |= fs

    # "none" should not be combined with other flags
    if "none" in combined and len(combined) > 1:
        combined.discard("none")

    if not combined:
        return "none"
    return ";".join(sorted(combined))
