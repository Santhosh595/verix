"""
The only module allowed to produce the final claim_status. Fuses vision
findings, evidence sufficiency, and risk flags following the hierarchy in
prompt/TASK_BRIEF.md §4:

    vision evidence > conversation scope > evidence-requirements gate
    > history (context only, never overriding clear visual evidence)

Key design: claim_status_justification explicitly shows conflict resolution
between vision, claim text, and history — demonstrating "images are ground
truth" rather than just asserting it.
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
    valid_image_reason: str     # explanation for valid_image decision
    severity: str
    risk_flags: str             # ';'-joined, or "none"
    evidence_standard_met: str  # "true" / "false"
    evidence_standard_met_reason: str
    confidence: str             # "high" / "medium" / "low" — self-reported


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


def _assess_confidence(
    findings: list,
    claim_status: str,
    evidence_standard_met: bool,
    risk_flags: str,
) -> str:
    """
    Self-reported confidence based on signal quality.
    - high: clean images, vision agrees with evidence, no risk flags
     - medium: some quality issues or partial mismatch
     - low: poor quality, ambiguous vision, or history risk flags present
    """
    if not evidence_standard_met:
        return "low"

    # Check for quality issues
    has_quality_issues = any(
        {"blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle"} & set(f.quality_notes)
        for f in findings
    )
    if has_quality_issues:
        return "medium"

    # Check for risk flags
    active_risk_flags = risk_flags.split(";") if risk_flags and risk_flags != "none" else []
    if "user_history_risk" in active_risk_flags or "manual_review_required" in active_risk_flags:
        return "medium"

    # Check for ambiguity in findings
    issue_types_seen = set(f.issue_type for f in findings if f.issue_type != "unknown")
    if len(issue_types_seen) > 1:
        return "medium"

    # Check confidence spread
    avg_confidence = sum(f.confidence for f in findings) / max(len(findings), 1)
    if avg_confidence < 0.5:
        return "low"

    return "high"


def _build_justification(
    claim_status: str,
    decision: dict,
    parsed_claim,
    image_findings: list,
    evidence_result,
    risk_flags: str,
    user_history_row: dict | None,
) -> str:
    """
    Build a justification that explicitly shows conflict resolution.
    Format: "[Vision evidence]. [Claim text context]. [Evidence gate]. [History context]."
    """
    parts: list[str] = []

    # 1. Vision evidence (ground truth)
    vision_issue = decision["vision_issue"]
    vision_part = decision["vision_part"]
    n_images = decision["n_images"]
    n_clear = decision["n_clear_images"]

    if claim_status == "supported":
        supporting = decision["supporting_ids"]
        img_ref = ";".join(supporting) if supporting else "images"
        parts.append(
            f"Vision evidence ({n_clear}/{n_images} images clear): shows {vision_issue} on {vision_part} in {img_ref}."
        )
    elif claim_status == "contradicted":
        parts.append(
            f"Vision evidence ({n_clear}/{n_images} images clear): shows {vision_issue} on {vision_part}, which contradicts the claim."
        )
    else:
        parts.append(
            f"Vision evidence ({n_clear}/{n_images} images clear): {vision_issue} on {vision_part} — insufficient for definitive determination."
        )

    # 2. Claim text context
    if parsed_claim and parsed_claim.claimed_issue_type_guess:
        parts.append(
            f"User claim: '{parsed_claim.claimed_issue_type_guess} on {parsed_claim.claimed_object_part_guess or 'unspecified'}'. "
            f"Vision {'supports' if claim_status == 'supported' else 'contradicts' if claim_status == 'contradicted' else 'is ambiguous relative to'} the claim."
        )
    elif parsed_claim:
        parts.append(f"User claim: '{parsed_claim.claimed_issue_summary[:80]}...' (text-based extraction).")

    # 3. Evidence gate (truncate on word boundary, never mid-word)
    def _truncate_reason(text: str, width: int = 80) -> str:
        """Truncate text on word boundary with ellipsis."""
        if len(text) <= width:
            return text
        truncated = text[:width]
        last_space = truncated.rfind(" ")
        if last_space > width // 2:
            truncated = truncated[:last_space]
        return truncated.rstrip() + "..."
    if evidence_result.evidence_standard_met:
        parts.append(f"Evidence standard met: {_truncate_reason(evidence_result.evidence_standard_met_reason)}")
    else:
        parts.append(f"Evidence standard NOT met: {_truncate_reason(evidence_result.evidence_standard_met_reason)}")

    # 4. History context (explicitly state it did NOT override)
    active_flags = risk_flags.split(";") if risk_flags and risk_flags != "none" else []
    if active_flags:
        parts.append(
            f"Risk flags: {', '.join(active_flags)}. "
            f"History context noted but did not override visual evidence per decision hierarchy."
        )

    return " | ".join(parts)


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
        vision_issue = _majority_issue(image_findings)
        vision_part = _majority_part(image_findings)
        n_clear = sum(1 for f in image_findings if "unreadable" not in f.quality_notes and "blurry_image" not in f.quality_notes)
        n_total = len(image_findings)

        valid_image = "true" if n_clear > 0 else "false"
        valid_image_reason = f"{n_clear}/{n_total} images usable" if n_clear > 0 else "All images unreadable or too blurry"

        return Decision(
            issue_type=vision_issue,
            object_part=vision_part,
            claim_status="not_enough_information",
            claim_status_justification=(
                f"Evidence standard not met: {evidence_result.evidence_standard_met_reason}. "
                f"Cannot evaluate claim due to insufficient image evidence."
            ),
            supporting_image_ids="none",
            valid_image=valid_image,
            valid_image_reason=valid_image_reason,
            severity="unknown",
            risk_flags=risk_flags,
            evidence_standard_met="false",
            evidence_standard_met_reason=evidence_result.evidence_standard_met_reason,
            confidence="low",
        )

    # --- Step 2: All images invalid ---
    n_clear = sum(1 for f in image_findings if "unreadable" not in f.quality_notes and "blurry_image" not in f.quality_notes)
    n_total = len(image_findings)
    valid_image = "true" if n_clear > 0 else "false"
    valid_image_reason = f"{n_clear}/{n_total} images usable" if n_clear > 0 else "All images unreadable or too blurry"

    if valid_image == "false":
        return Decision(
            issue_type="unknown",
            object_part="unknown",
            claim_status="not_enough_information",
            claim_status_justification=(
                f"Images present but all are unreadable or too blurry to assess. "
                f"Evidence standard met but image quality insufficient."
            ),
            supporting_image_ids="none",
            valid_image="false",
            valid_image_reason=valid_image_reason,
            severity="unknown",
            risk_flags=risk_flags,
            evidence_standard_met="true",
            evidence_standard_met_reason=evidence_result.evidence_standard_met_reason,
            confidence="low",
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

    # --- Step 4: Build decision dict for justification ---
    supporting_ids = _find_supporting_images(
        image_findings, claim_status, claimed_issue, claimed_part
    )

    decision_dict = {
        "vision_issue": vision_issue,
        "vision_part": vision_part,
        "n_images": n_total,
        "n_clear_images": n_clear,
        "supporting_ids": supporting_ids,
    }

    # --- Step 5: Build justification with conflict resolution transparency ---
    justification = _build_justification(
        claim_status, decision_dict, parsed_claim,
        image_findings, evidence_result, risk_flags, user_history_row
    )

    # --- Step 6: History may add manual_review_required but not flip status ---
    if "manual_review_required" in risk_flag_set and claim_status != "not_enough_information":
        justification += " Manual review recommended due to user history risk, but visual evidence takes precedence per hierarchy."

    # --- Step 7: Confidence assessment ---
    confidence = _assess_confidence(image_findings, claim_status, evidence_result.evidence_standard_met, risk_flags)

    return Decision(
        issue_type=vision_issue,
        object_part=vision_part,
        claim_status=claim_status,
        claim_status_justification=justification,
        supporting_image_ids=";".join(supporting_ids) if supporting_ids else "none",
        valid_image=valid_image,
        valid_image_reason=valid_image_reason,
        severity=severity,
        risk_flags=risk_flags,
        evidence_standard_met="true" if evidence_result.evidence_standard_met else "false",
        evidence_standard_met_reason=evidence_result.evidence_standard_met_reason,
        confidence=confidence,
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
