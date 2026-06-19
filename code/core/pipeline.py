"""
Orchestrates one claim end-to-end. Used by both main.py (full run) and
evaluation/main.py (scoring against sample_claims.csv) so there is only
one code path that defines "how a claim gets decided".
"""

import logging
import time

from config.schema import OUTPUT_COLUMNS
from core import claim_parser, data_loader, decision_engine, evidence_checker, risk_assessor, vision_analyzer

logger = logging.getLogger(__name__)


def run_claim(record, evidence_requirements: list[dict], user_history: dict[str, dict]) -> dict:
    """
    Process one claim end-to-end:
    1. Parse the claim text
    2. Analyze images with VLM
    3. Check evidence sufficiency
    4. Assess risk flags
    5. Make final decision
    6. Assemble output dict with exact OUTPUT_COLUMNS
    """
    start = time.time()

    # 1. Parse claim
    parsed = claim_parser.parse_claim(record.user_claim, record.claim_object)

    # 2. Analyze images
    findings = vision_analyzer.analyze_claim_images(
        record.image_paths, record.claim_object, parsed.claimed_issue_summary
    )

    # 3. Check evidence
    evidence_result = evidence_checker.check_evidence(
        claim_object=record.claim_object,
        issue_family_guess=parsed.claimed_issue_type_guess or "general claim review",
        image_findings=findings,
        missing_image_paths=record.missing_image_paths,
        evidence_requirements=evidence_requirements,
    )

    # 4. Assess risk
    image_risk = risk_assessor.assess_image_risk(findings, record.claim_object, parsed)
    history_risk = risk_assessor.assess_history_risk(user_history.get(record.user_id))
    risk_flags = risk_assessor.combine_risk_flags(image_risk, history_risk)

    # 5. Make decision
    decision = decision_engine.decide(
        parsed, findings, evidence_result, risk_flags, user_history.get(record.user_id)
    )

    elapsed = time.time() - start
    logger.info(
        "Claim %s: status=%s, issue=%s, part=%s, severity=%s (%.1fs)",
        record.user_id, decision.claim_status, decision.issue_type,
        decision.object_part, decision.severity, elapsed,
    )

    # 6. Assemble output row
    image_paths_str = ";".join(str(p) for p in record.image_paths)
    # For output, use original relative paths if available
    if hasattr(record, 'raw_row') and 'image_paths' in record.raw_row:
        image_paths_str = record.raw_row['image_paths']

    output = {
        "user_id": record.user_id,
        "image_paths": image_paths_str,
        "user_claim": record.user_claim,
        "claim_object": record.claim_object,
        "evidence_standard_met": decision.evidence_standard_met,
        "evidence_standard_met_reason": decision.evidence_standard_met_reason,
        "risk_flags": decision.risk_flags,
        "issue_type": decision.issue_type,
        "object_part": decision.object_part,
        "claim_status": decision.claim_status,
        "claim_status_justification": decision.claim_status_justification,
        "supporting_image_ids": decision.supporting_image_ids,
        "valid_image": decision.valid_image,
        "valid_image_reason": decision.valid_image_reason,
        "confidence": decision.confidence,
        "severity": decision.severity,
    }

    # Sanity check: all keys match OUTPUT_COLUMNS
    assert set(output.keys()) == set(OUTPUT_COLUMNS), (
        f"Output columns mismatch: got {set(output.keys())}, expected {set(OUTPUT_COLUMNS)}"
    )

    return output


def run_batch(records: list, evidence_requirements: list[dict], user_history: dict[str, dict]) -> list[dict]:
    """
    Run run_claim for every record. Adds progress logging.
    Resumable via caching in vision_analyzer (each image is cached by hash).
    """
    total = len(records)
    results: list[dict] = []
    errors: list[tuple] = []

    for i, record in enumerate(records, 1):
        try:
            output = run_claim(record, evidence_requirements, user_history)
            results.append(output)
        except Exception as exc:
            logger.error("Error processing %s: %s", record.user_id, exc)
            errors.append((record.user_id, str(exc)))
            # Still produce a row so output count matches input count
            fallback = _fallback_row(record, str(exc))
            results.append(fallback)

    logger.info(
        "Batch complete: %d/%d processed, %d errors",
        len(results) - len(errors), total, len(errors),
    )

    if errors:
        logger.warning("Failed claims: %s", [uid for uid, _ in errors])

    return results


def _fallback_row(record, error_msg: str) -> dict:
    """Produce a safe fallback row when processing fails."""
    image_paths_str = ";".join(str(p) for p in record.image_paths)
    if hasattr(record, 'raw_row') and 'image_paths' in record.raw_row:
        image_paths_str = record.raw_row['image_paths']

    return {
        "user_id": record.user_id,
        "image_paths": image_paths_str,
        "user_claim": record.user_claim,
        "claim_object": record.claim_object,
        "evidence_standard_met": "false",
        "evidence_standard_met_reason": f"Processing error: {error_msg}",
        "risk_flags": "none",
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": f"Processing error: {error_msg}",
        "supporting_image_ids": "none",
        "valid_image": "false",
        "valid_image_reason": f"Processing error: {error_msg}",
        "confidence": "low",
        "severity": "unknown",
    }
