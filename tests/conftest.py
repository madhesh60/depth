"""Suite-wide test settings. The API tests fire many requests a minute from one client, so the
per-client rate limits (src/dashboard/ratelimit.py) are off by default here; test_ratelimit.py turns
them back on explicitly."""
import os

os.environ.setdefault("DEPTH_RATE_LIMITS", "0")

# never write test approvals / labels into the real stores under runs/ (the agent now files approval
# requests on every survey, and survey decisions become training labels)
import tempfile as _tempfile

_TMP = _tempfile.mkdtemp(prefix="depth-tests-")
os.environ.setdefault("DEPTH_APPROVALS_DIR", os.path.join(_TMP, "approvals"))
os.environ.setdefault("DEPTH_FEEDBACK_DIR", os.path.join(_TMP, "feedback"))
