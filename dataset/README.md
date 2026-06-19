# dataset/

Place the actual dataset files here (not included in this scaffold —
too large to bundle, and excluded from code.zip at submission time
anyway):

- sample_claims.csv
- claims.csv
- user_history.csv
- evidence_requirements.csv
- images/sample/   (referenced by sample_claims.csv)
- images/test/     (referenced by claims.csv)

code/config/settings.py already points at these exact paths — drop the
files in and the pipeline should find them without any path changes.
