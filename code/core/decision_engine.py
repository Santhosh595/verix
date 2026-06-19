"""
The only module allowed to produce the final claim_status. Fuses vision
findings, evidence sufficiency, and risk flags following the hierarchy in
prompt/TASK_BRIEF.md §4:

    vision evidence > conversation scope > evidence-requirements gate
    > history (context only, never overriding clear visual evidence)
"""

import logging
from dataclasses import dataclass

from config.schema import (
    CLAIM_STATUSES,
    ISSUE_TYPES,
    SEVERITIES,
    closest_allowed_value,
)

logger = logging.getLogger(__name__)

# Severity ordering for "max" aggregation
_SEVERITY_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3, "unknown": -1}


@dataclass
class Decision:
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: str   # ';'-joined, or "none"
    valid_image: str            # "true" / "false"
    severity: str
    risk_flags: str             # ';'-joined, or "none"
    evidence_standard_met: str  # "true" / "false"
    evidence_standard_met_reason: str


def _aggregate_severities(findings: list) -> str:
    """Pick the max severity across findings."""
    max_sev = "none"
    max_order = 0
    for f in findings:
        sev = f.severity if f.severity in SEVERITIES else "unknown"
        order = _SEVERITY_ORDER.get(sev, -1)
        if order > max_order:
            max_order = order
            max_sev = sev
    return max_sev


def _find_supporting_images(
    findings: list,
    claim_status: str,
    claimed_issue_type: str | None,
    claimed_object_part: str | None,
) -> list[str]:
    """
    Find image IDs that most directly support the decision.
    - For "supported": images that show the claimed issue/part
    - For "contradicted": images that clearly show something different
    - For "not_enough_information": empty (no image supports the decision)
    """
    supporting: list[str] = []

    if claim_status == "not_enough_information":
        return supporting

    for f in findings:
        if f.issue_type == "unknown" and f.object_part == "unknown":
            continue  # this image doesn't help

        if claim_status == "supported":
            # Image supports if it shows the claimed issue type or part
            if (
                (claimed_issue_type and f.issue_type == claimed_issue_type)
                or (claimed_object_part and f.object_part == claimed_object_part)
            ):
                if f.confidence >= 0.3:
                    supporting.append(f.image_id)

        elif claim_status == "contradicted":
            # Image contradicts if it shows a different issue or part
            if (
                (claimed_issue_type and f.issue_type != claimed_issue_type and f.issue_type != "unknown")
                or (claimed_object_part and f.object_part != claimed_object_part and f.object_part != "unknown")
            ):
                if f.confidence >= 0.3:
                    supporting.append(f.image_id)

    return supporting


def decide(
    parsed_claim,
    image_findings: list,
    evidence_result,
    risk_flags: str,
    user_history_row: dict | None,
) -> Decision:
    """
    Produce the final claim_status. Follows the hierarchy:
    1. Evidence gate (if not enough info -> not_enough_information)
    2. Vision vs claim comparison (supported or contradicted)
    3. History/risk only add flags, never override clear vision
    """
    risk_flag_set = set(risk_flags.split(";")) if risk_flags and risk_flags != "none" else set()

    # --- Step 1: Evidence gate ---
    if not evidence_result.evidence_standard_met:
        return Decision(
            issue_type=_majority_issue(image_findings),
            object_part=_majority_part(image_findings),
            claim_status="not_enough_information",
            claim_status_justification=(
                f"Insufficient image evidence to evaluate the claim. "
                f"Reason: {evidence_result.evidence_standard_met_reason}"
            ),
            supporting_image_ids="none",
            valid_image="false" if all(
                "unreadable" in f.quality_notes for f in image_findings
            ) else "true",
            severity="unknown",
            risk_flags=risk_flags,
            evidence_standard_met="false",
            evidence_standard_met_reason=evidence_result.evidence_standard_met_reason,
        )

    # --- Step 2: All images invalid ---
    valid_image = "true" if any(
        "unreadable" not in f.quality_notes and "blurry_image" not in f.quality_notes
        for f in image_findings
    ) else "false"

    if valid_image == "false":
        return Decision(
            issue_type="unknown",
            object_part="unknown",
            claim_status="not_enough_information",
            claim_status_justification=(
                "Images are present but all are unreadable or too blurry to assess."
            ),
            supporting_image_ids="none",
            valid_image="false",
            severity="unknown",
            risk_flags=risk_flags,
            evidence_standard_met="true",
            evidence_standard_met_reason=evidence_result.evidence_standard_met_reason,
        )

    # --- Step 3: Compare vision vs claim ---
    claimed_issue = parsed_claim.claimed_issue_type_guess
    claimed_part = parsed_claim.claimed_object_part_guess

    # Aggregate vision findings
    vision_issue = _majority_issue(image_findings)
    vision_part = _majority_part(image_findings)
    severity = _aggregate_severities(image_findings)

    # Count findings that match vs contradict
    match_count = 0
    contradict_count = 0
    no_damage_count = 0

    for f in image_findings:
        if f.issue_type == "none":
            no_damage_count += 1
        elif claimed_issue and f.issue_type == claimed_issue:
            match_count += 1
        elif claimed_issue and f.issue_type != claimed_issue and f.issue_type != "unknown":
            contradict_count += 1

        if claimed_part and f.object_part == claimed_part:
            match_count += 1
        elif claimed_part and f.object_part != claimed_part and f.object_part != "unknown":
            contradict_count += 1

    # Determine claim_status
    if match_count > 0 and contradict_count == 0:
        claim_status = "supported"
    elif contradict_count > 0 and match_count == 0:
        claim_status = "contradicted"
    elif no_damage_count > 0 and match_count == 0:
        # Images show no damage but user claims there is one
        claim_status = "contradicted"
    elif match_count > 0 and contradict_count > 0:
        # Mixed signals -- ambiguous
        claim_status = "not_enough_information"
    else:
        # No clear signal either way
        claim_status = "not_enough_information"

    # --- Step 4: Build justification ---
    supporting_ids = _find_supporting_images(
        image_findings, claim_status, claimed_issue, claimed_part
    )

    if claim_status == "supported":
        justification = (
            f"Image(s) {';'.join(supporting_ids)} show {vision_issue} on the {vision_part}, "
            f"consistent with the user's claim."
        )
    elif claim_status == "contradicted":
        justification = (
            f"Image(s) {';'.join(supporting_ids) if supporting_ids else 'submitted'} show "
            f"{vision_issue} on the {vision_part}, which does not match the claimed "
            f"{claimed_issue or 'issue'} on the {claimed_part or 'part'}."
        )
    else:
        justification = (
            f"Image evidence is insufficient for a definitive determination. "
            f"Vision model found {vision_issue} on {vision_part}; "
            f"claimed was {claimed_issue or 'unspecified'} on {claimed_part or 'unspecified'}."
        )

    # --- Step 5: History may add manual_review_required but not flip status ---
    if "manual_review_required" in risk_flag_set and claim_status != "not_enough_information":
        # Keep the status but note the need for review in justification
        justification += " Manual review recommended due to user history risk."

    return Decision(
        issue_type=vision_issue,
        object_part=vision_part,
        claim_status=claim_status,
        claim_status_justification=justification,
        supporting_image_ids=";".join(supporting_ids) if supporting_ids else "none",
        valid_image=valid_image,
        severity=severity,
        risk_flags=risk_flags,
        evidence_standard_met="true" if evidence_result.evidence_standard_met else "false",
        evidence_standard_met_reason=evidence_result.evidence_standard_met_reason,
    )


def _majority_issue(findings: list) -> str:
    """Return the most common non-unknown issue type, or 'unknown'."""
    from collections import Counter
    issues = [f.issue_type for f in findings if f.issue_type != "unknown"]
    if not issues:
        return "unknown"
    counter = Counter(issues)
    return counter.most_common(1)[0][0]


def _majority_part(findings: list) -> str:
    """Return the most common non-unknown object part, or 'unknown'."""
    from collections import Counter
    parts = [f.object_part for f in findings if f.object_part != "unknown"]
    if not parts:
        return "unknown"
    counter = Counter(parts)
    return counter.most_common(1)[0][0]
