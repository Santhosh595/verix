"""Make the ``code/`` directory importable for the whole test suite.

The modules under test (``config``, ``core``, ``utils``) are top-level
packages that live in ``code/``, so ``code/`` -- not the repository root --
is the directory that has to be on ``sys.path``.

Each test file used to insert that path itself, and did it incorrectly:
``code/tests/test_adversarial.py`` inserted ``code/tests`` (the tests
directory), which does not contain the packages it then imports. The
result was that a clean checkout could not collect the suite at all:

    ModuleNotFoundError: No module named 'core'

Setting the path once, here, fixes it for every test file and lets the
per-file ``sys.path`` hacks go.
"""

import sys
from pathlib import Path

# The repository's ``code`` directory is the import root: it holds the
# ``config``, ``core`` and ``utils`` packages that the tests import.
CODE_ROOT = Path(__file__).resolve().parent / "code"

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
