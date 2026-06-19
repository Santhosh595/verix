"""
Applies dataset/evidence_requirements.csv to decide whether the submitted
image set meets the minimum bar to evaluate this claim at all -- this is
independent of whether the claim turns out to be true.

The evidence requirements describe qualitative bars (e.g. "the claimed panel
should be visible from an angle where surface marks can be assessed"). This
module translates those bars into concrete checks against the actual
ImageFinding data we got from the VLM.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EvidenceCheckResult:
    evidence_standard_met: bool
    evidence_standard_met_reason: str


def _has_image_quality_issue(image_findings, required_object_part: str) -> bool:
    """
    True if ALL images showing the required object part have quality issues
    that would prevent proper assessment (blur, obstruction, wrong angle).
    """
    relevant_findings = [
        f for f in image_findings
        if f.object_part == required_object_part or required_object_part in f.quality_notes
    ]
    if not relevant_findings:
        # No finding mentions the required part -- check all findings
        relevant_findings = image_findings

    if not relevant_findings:
        return True  # no images at all

    for f in relevant_findings:
        blocking_issues = {"blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle"}
        if blocking_issues & set(f.quality_notes):
            return True
    return False


def _count_usable_images(image_findings) -> int:
    """Count images that aren't flagged with blocking quality issues."""
    blocking = {"blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle", "unreadable"}
    count = 0
    for f in image_findings:
        if not (blocking & set(f.quality_notes)):
            count += 1
    return count


def _has_authenticity_concern(image_findings) -> bool:
    """True if any image has authenticity/manipulation notes."""
    for f in image_findings:
        if f.authenticity_notes:
            return True
    return False


def _extract_required_object_part(issue_family: str, claim_object: str) -> str | None:
    """
    Extract the object part that evidence requirements are checking for,
    based on the issue family and claim_object. Returns the most specific
    part mentioned, or None for general requirements.
    """
    from config.schema import OBJECT_PARTS
    parts = OBJECT_PARTS.get(claim_object, [])
    issue_lower = issue_family.lower()
    for part in parts:
        if part in issue_lower:
            return part
    return None


def check_evidence(
    claim_object: str,
    issue_family_guess: str,
    image_findings: list,          # list[vision_analyzer.ImageFinding]
    missing_image_paths: list[str],
    evidence_requirements: list[dict],
) -> EvidenceCheckResult:
    """
    Evaluate whether the submitted image set meets the evidence standard.

    Checks:
    1. Missing images -> automatic fail
    2. Image quality (blur, obstruction, etc.) -> may fail
    3. Authenticity concerns -> may fail
    4. Minimum image count based on evidence rules
    5. Object part visibility (does any finding show the claimed part?)

    Returns EvidenceCheckResult with bool + human-readable reason.
    """
    reasons: list[str] = []

    # --- Check 1: Missing images ---
    if missing_image_paths:
        reasons.append(
            f"Image file(s) not found: {', '.join(missing_image_paths)}"
        )
        return EvidenceCheckResult(
            evidence_standard_met=False,
            evidence_standard_met_reason="; ".join(reasons),
        )

    # --- Check 2: No images at all ---
    if not image_findings:
        return EvidenceCheckResult(
            evidence_standard_met=False,
            evidence_standard_met_reason="No images provided for evaluation.",
        )

    # --- Check 3: Count usable images ---
    usable_count = _count_usable_images(image_findings)
    total_count = len(image_findings)

    if usable_count == 0:
        reasons.append("All images have blocking quality issues (blur, obstruction, or bad lighting)")
        return EvidenceCheckResult(
            evidence_standard_met=False,
            evidence_standard_met_reason="; ".join(reasons),
        )

    # --- Check 4: Look up evidence requirements and apply specific rules ---
    from core.data_loader import lookup_evidence_requirement

    matched_req = lookup_evidence_requirement(claim_object, issue_family_guess)

    if matched_req is None:
        # Fall back to general rules
        general_reqs = [
            r for r in evidence_requirements
            if r.get("claim_object", "").lower() == "all"
        ]
        if general_reqs:
            matched_req = general_reqs[0]

    # Build reason context
    req_id = ""
    if matched_req:
        req_id = matched_req.get("requirement_id", "")
        min_evidence = matched_req.get("minimum_image_evidence", "")

    # --- Check 5: Multi-image requirement ---
    # REQ_GENERAL_MULTI_IMAGE: if multiple images submitted, at least one
    # must show the claimed object/part clearly
    multi_image_req = None
    for r in evidence_requirements:
        if r.get("requirement_id") == "REQ_GENERAL_MULTI_IMAGE":
            multi_image_req = r
            break

    if total_count > 1 and multi_image_req:
        # At least one image must clearly show the claimed part
        has_clear_image = any(
            f.confidence >= 0.5 and "wrong_object" not in f.quality_notes
            for f in image_findings
        )
        if not has_clear_image:
            reasons.append(
                "Multiple images submitted but none clearly shows the claimed object/part"
            )

    # --- Check 6: Object part visibility ---
    required_part = _extract_required_object_part(issue_family_guess or "", claim_object)
    if required_part:
        part_visible = any(
            f.object_part == required_part and f.confidence >= 0.3
            for f in image_findings
        )
        if not part_visible:
            # Check if any image at least shows something related
            all_unknown = all(f.object_part == "unknown" for f in image_findings)
            if all_unknown:
                reasons.append(
                    f"Cannot determine if the claimed {required_part} is visible in any image"
                )
            else:
                found_parts = [f.object_part for f in image_findings if f.object_part != "unknown"]
                reasons.append(
                    f"Claimed part '{required_part}' not clearly visible; found: {', '.join(found_parts) or 'none'}"
                )

    # --- Check 7: Quality issues ---
    if usable_count < total_count:
        bad_count = total_count - usable_count
        reasons.append(
            f"{bad_count} of {total_count} images have quality issues"
        )

    # --- Check 8: Authenticity ---
    if _has_authenticity_concern(image_findings):
        reasons.append("Authenticity concerns flagged on submitted images")

    # --- Check 9: Reviewability rule ---
    reviewability_req = None
    for r in evidence_requirements:
        if r.get("requirement_id") == "REQ_REVIEW_TRUST":
            reviewability_req = r
            break

    if reviewability_req:
        # All images are "unknown" issue type and "unknown" object part
        all_undetermined = all(
            f.issue_type == "unknown" and f.object_part == "unknown"
            for f in image_findings
        )
        if all_undetermined:
            return EvidenceCheckResult(
                evidence_standard_met=False,
                evidence_standard_met_reason=(
                    "Images do not provide usable visual evidence relevant to the claim."
                ),
            )

    # --- Final decision ---
    if reasons:
        # If we have usable images and only minor quality issues, still pass
        # but note the issues. Fail only if the core evidence is insufficient.
        critical_fail = (
            usable_count == 0
            or (required_part and not part_visible)
        )
        if critical_fail:
            reason_text = "; ".join(reasons)
            if matched_req:
                reason_text = f"{reason_text} (rule: {req_id})"
            return EvidenceCheckResult(
                evidence_standard_met=False,
                evidence_standard_met_reason=reason_text,
            )

        # Pass with caveats
        reason_text = f"Evidence standard met with noted issues: {'; '.join(reasons)}"
        if matched_req:
            reason_text += f" (rule: {req_id})"
        return EvidenceCheckResult(
            evidence_standard_met=True,
            evidence_standard_met_reason=reason_text,
        )

    # Clean pass
    if matched_req:
        return EvidenceCheckResult(
            evidence_standard_met=True,
            evidence_standard_met_reason=(
                f"{usable_count} usable image(s) provided. "
                f"Requirement met: {matched_req.get('minimum_image_evidence', '')}"
            ),
        )
    return EvidenceCheckResult(
        evidence_standard_met=True,
        evidence_standard_met_reason=(
            f"{usable_count} usable image(s) provided; no specific evidence rule matched but "
            f"general standard satisfied."
        ),
    )
