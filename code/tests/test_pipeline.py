"""
Basic smoke tests. TODO: expand as modules get implemented.

Suggested minimum coverage before submission:
- csv_io.write_output_csv enforces exact column order/names.
- decision_engine.decide never returns claim_status="supported"/
  "contradicted" when evidence_standard_met is False.
- risk_assessor.combine_risk_flags returns "none" for an empty set and a
  sorted ';'-joined string otherwise.
- A known-injection user_claim string sets looks_like_injection=True in
  claim_parser without altering downstream behavior.
"""


def test_output_columns_exact():
    from config.schema import OUTPUT_COLUMNS
    assert OUTPUT_COLUMNS[0] == "user_id"
    assert OUTPUT_COLUMNS[-1] == "severity"
    assert len(OUTPUT_COLUMNS) == 14
