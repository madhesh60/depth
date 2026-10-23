"""
sns_notify.py — the person who decides is told by email (Amazon SNS) when DEPTH asks for an approval.

The agent (after a survey), an MCP client or a person in the studio files an approval request. Webhooks carry it
to operations consoles (``integrations.py``); this module emails it to the approver subscribed to an SNS topic
(``DEPTH_SNS_APPROVALS_TOPIC``, which the stack sets through ``/etc/depth/aws.env``). The email says what is asked,
by whom, for how many targets and why, and where to decide: the studio's Approvals tab. It never carries the PIN,
a token or a position (protected wrecks stay generalised) - deciding stays in the studio, with the PIN.

Guards, so an email channel cannot become noise or a cost:

* **one email per survey** - requests for the same survey filed within ``BATCH_S`` seconds go out together (the
  agent files one request per planned action, in a burst);
* **at most ``max_per_hour`` emails an hour** (``DEPTH_SNS_MAX_PER_HOUR``, default 6); the rest are counted as
  suppressed - the studio inbox still shows every request;
* **best-effort**: it publishes on a timer thread; no boto3, no credentials or an API error is logged, never raised.

Every outcome is one JSON log line (``{"evt": "notify", "channel": "sns", "status": "sent" | "suppressed" |
"failed", ...}``) behind the CloudWatch metric ``DEPTH/approval_emails`` and the saved query ``approval-emails``.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import deque
from typing import Any, Callable, Optional

log = logging.getLogger("depth.sns")
BATCH_S = 5.0
SUBJECT_MAX = 99                       # SNS: ASCII, no line breaks, fewer than 100 characters
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _clean(text: Any, limit: int) -> str:
    """One line of untrusted text (an MCP client writes the rationale) - control characters out, length capped."""
    s = _CTRL.sub(" ", str(text if text is not None else "")).replace("\n", " ").strip()
    return s if len(s) <= limit else s[: limit - 3] + "..."


def compose(reqs: list[dict], public_url: str = "") -> tuple[str, str]:
    """(subject, message) for one survey's pending requests. Plain text; no PIN, token or position."""
    sid = _clean(reqs[0].get("survey_id") or "-", 60)
    n = len(reqs)
    subject = f"DEPTH: {n} action{'s' if n != 1 else ''} wait for your approval (survey {sid})"
    subject = subject.encode("ascii", "replace").decode("ascii")[:SUBJECT_MAX]
    lines = ["DEPTH asks a person to decide. Nothing is dispatched until you approve it in the DEPTH studio.", "",
             f"Survey: {sid}", ""]
    for i, r in enumerate(reqs, 1):
        targets = r.get("targets") or []
        lines += [f"{i}. {_clean(r.get('action'), 40).replace('_', ' ')} - {len(targets)} target(s) - asked by "
                  f"{_clean(r.get('requested_by') or 'agent', 60)} via {_clean(r.get('channel') or 'api', 20)}",
                  f"   Why: {_clean(r.get('rationale'), 400)}",
                  f"   Request: {_clean(r.get('id'), 24)}", ""]
    where = f"{public_url}/ -> Approvals" if public_url else "the DEPTH studio -> Approvals"
    lines += [f"Decide: {where}. You need the approver PIN (it is never sent by email).",
              "Agents and API clients can only ask; a named person decides."]
    return subject, "\n".join(lines) + "\n"


class SnsNotifier:
    """Batches approval requests per survey and publishes one email per batch to an SNS topic (see the module)."""

    def __init__(self, topic_arn: Optional[str] = None, public_url: str = "", client: Any = None,
                 batch_s: float = BATCH_S, max_per_hour: Optional[int] = None,
                 clock: Callable[[], float] = time.time):
        self.topic = (os.environ.get("DEPTH_SNS_APPROVALS_TOPIC", "") if topic_arn is None else topic_arn).strip()
        self.public_url = public_url.rstrip("/")
        self.batch_s = batch_s
        self.max_per_hour = int(max_per_hour if max_per_hour is not None
                                else os.environ.get("DEPTH_SNS_MAX_PER_HOUR", "6") or 6)
        self.clock = clock
        self._client = client
        self._lock = threading.Lock()
        self._pending: dict[str, list[dict]] = {}
        self._timers: dict[str, threading.Timer] = {}
        self._sent_at: deque = deque()                 # publish times in the last hour
        self.stats = {"sent": 0, "suppressed": 0, "failed": 0}
        self.recent: deque = deque(maxlen=50)

    @property
    def enabled(self) -> bool:
        return bool(self.topic)

    def status(self) -> dict:
        """What the health endpoint shows: on/off, counts and the cap - never the topic ARN (it names the account)."""
        return {"enabled": self.enabled, "max_per_hour": self.max_per_hour, **self.stats}

    def notify(self, event: str, data: dict) -> bool:
        """Queue an ``approval.requested`` event for its survey's email; anything else is ignored. True if queued."""
        if not self.enabled or event != "approval.requested":
            return False
        key = str(data.get("survey_id") or "-")
        with self._lock:
            self._pending.setdefault(key, []).append(dict(data))
            if key not in self._timers:
                t = threading.Timer(self.batch_s, self.flush, args=(key,))
                t.daemon = True
                self._timers[key] = t
                t.start()
        return True

    def flush(self, key: str) -> Optional[dict]:
        """Publish one survey's queued requests now (the timer calls this). Returns the outcome record."""
        with self._lock:
            reqs = self._pending.pop(key, [])
            timer = self._timers.pop(key, None)
            if timer is not None:
                timer.cancel()                         # harmless when the timer itself is the caller
            if not reqs:
                return None
            now = self.clock()
            while self._sent_at and now - self._sent_at[0] >= 3600:
                self._sent_at.popleft()
            capped = len(self._sent_at) >= self.max_per_hour
            if capped:
                self.stats["suppressed"] += 1
            else:
                self._sent_at.append(now)
        if capped:
            return self._record("suppressed", key, len(reqs))
        subject, message = compose(reqs, self.public_url)
        try:
            resp = self._sns().publish(TopicArn=self.topic, Subject=subject, Message=message)
        except Exception as e:                          # never breaks the product: the studio inbox has the request
            with self._lock:
                self.stats["failed"] += 1
            log.warning("approval email failed: %s: %s", type(e).__name__, e)
            return self._record("failed", key, len(reqs), error=f"{type(e).__name__}: {e}"[:200])
        with self._lock:
            self.stats["sent"] += 1
        return self._record("sent", key, len(reqs), message_id=(resp or {}).get("MessageId"))

    def flush_all(self) -> list[dict]:
        """Publish every queued survey now (shutdown, tests)."""
        with self._lock:
            keys = list(self._pending)
        return [r for r in (self.flush(k) for k in keys) if r]

    def _sns(self):
        if self._client is None:
            import boto3                                # optional dependency (on the server: /opt/depth/pydeps)
            parts = self.topic.split(":")               # arn:aws:sns:<region>:<account>:<name>
            self._client = boto3.client("sns", region_name=parts[3] if len(parts) > 5 else None)
        return self._client

    def _record(self, status: str, survey: str, n: int, **extra) -> dict:
        rec = {"evt": "notify", "channel": "sns", "event": "approval.requested", "status": status,
               "survey_id": survey, "requests": n, "t": round(self.clock(), 3), **extra}
        self.recent.appendleft(rec)
        if os.environ.get("DEPTH_LOG_JSON") == "1":     # the CloudWatch metric filter and saved query read it
            print(json.dumps(rec), flush=True)
        log.info("approval email %s (survey %s, %d request(s))", status, survey, n)
        return rec
