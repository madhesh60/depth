"""Suite-wide test settings. The API tests fire many requests a minute from one client, so the
per-client rate limits (src/dashboard/ratelimit.py) are off by default here; test_ratelimit.py turns
them back on explicitly."""
import os

os.environ.setdefault("DEPTH_RATE_LIMITS", "0")
