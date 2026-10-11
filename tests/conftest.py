"""Suite-wide test settings. The API tests fire many requests a minute from one client, so the
per-client rate limits (src/dashboard/ratelimit.py) are off by default here; test_ratelimit.py turns
them back on explicitly."""
import os
import sys
from pathlib import Path

# CI runs the `pytest` entry point, which (unlike `python -m pytest`) does not put the repo root on
# sys.path: a test module that imports `src` before its own path insert then fails to COLLECT and the
# whole run stops (exit 2 - CI run 40). conftest.py loads before every test module, so do it once here.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("DEPTH_RATE_LIMITS", "0")

# never write test approvals / labels into the real stores under runs/ (the agent now files approval
# requests on every survey, and survey decisions become training labels)
import tempfile as _tempfile

_TMP = _tempfile.mkdtemp(prefix="depth-tests-")
os.environ.setdefault("DEPTH_APPROVALS_DIR", os.path.join(_TMP, "approvals"))
os.environ.setdefault("DEPTH_FEEDBACK_DIR", os.path.join(_TMP, "feedback"))
