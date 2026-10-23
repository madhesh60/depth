"""
The approver's email (src/dashboard/sns_notify.py): one email per survey, a cap per hour, never the PIN, a token or
a position, and never an exception into the product. A fake SNS client stands in for AWS.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dashboard.sns_notify import SUBJECT_MAX, SnsNotifier, compose

TOPIC = "arn:aws:sns:us-east-1:111122223333:depth-ApprovalTopic-ABC"


class FakeSns:
    def __init__(self, fail: bool = False):
        self.calls: list[dict] = []
        self.fail = fail

    def publish(self, **kw):
        if self.fail:
            raise ConnectionError("network down")
        self.calls.append(kw)
        return {"MessageId": f"m-{len(self.calls)}"}


def _req(sid: str = "S-1", action: str = "dispatch_route", rid: str = "AR-1", **kw) -> dict:
    return {"id": rid, "survey_id": sid, "action": action, "targets": ["H001", "H002"],
            "rationale": "three cards above the review threshold on one line", "requested_by": "DEPTH agent",
            "channel": "agent", **kw}


def _notifier(**kw) -> tuple[SnsNotifier, FakeSns]:
    fake = kw.pop("client", None) or FakeSns()
    n = SnsNotifier(topic_arn=TOPIC, public_url="https://demo.example", client=fake, batch_s=60, **kw)
    return n, fake


def test_off_without_a_topic_and_only_for_approval_requests():
    off = SnsNotifier(topic_arn="", client=FakeSns())
    assert not off.enabled and off.notify("approval.requested", _req()) is False
    n, fake = _notifier()
    assert n.notify("survey.completed", {"survey_id": "S-1"}) is False
    assert n.notify("approval.decided", _req()) is False
    assert n.flush_all() == [] and fake.calls == []


def test_one_email_per_survey_lists_every_request():
    n, fake = _notifier()
    for i, act in enumerate(("dispatch_route", "resurvey", "full_review"), 1):
        assert n.notify("approval.requested", _req(action=act, rid=f"AR-{i}"))
    n.notify("approval.requested", _req(sid="S-2", rid="AR-9"))
    out = n.flush_all()
    assert [r["status"] for r in out] == ["sent", "sent"] and len(fake.calls) == 2
    first = next(c for c in fake.calls if "S-1" in c["Subject"])
    assert first["TopicArn"] == TOPIC and first["Subject"].startswith("DEPTH: 3 actions wait for your approval")
    assert all(f"AR-{i}" in first["Message"] for i in (1, 2, 3)) and "resurvey" in first["Message"]
    assert "https://demo.example/ -> Approvals" in first["Message"]
    assert n.status() == {"enabled": True, "max_per_hour": 6, "sent": 2, "suppressed": 0, "failed": 0}


def test_the_hourly_cap_suppresses_and_the_window_slides():
    now = [1000.0]
    n, fake = _notifier(max_per_hour=1, clock=lambda: now[0])
    n.notify("approval.requested", _req(sid="S-1"))
    n.notify("approval.requested", _req(sid="S-2"))
    assert [r["status"] for r in n.flush_all()] == ["sent", "suppressed"] and len(fake.calls) == 1
    now[0] += 3600                                     # an hour later the cap allows one more
    n.notify("approval.requested", _req(sid="S-3"))
    assert [r["status"] for r in n.flush_all()] == ["sent"] and n.status()["suppressed"] == 1


def test_a_failed_publish_is_counted_never_raised():
    n, _ = _notifier(client=FakeSns(fail=True))
    n.notify("approval.requested", _req())
    rec = n.flush_all()[0]
    assert rec["status"] == "failed" and "ConnectionError" in rec["error"] and n.status()["failed"] == 1


def test_the_email_carries_no_pin_token_or_position_and_a_safe_subject(monkeypatch):
    monkeypatch.setenv("DEPTH_APPROVER_PIN", "424242")
    evil = _req(sid="S-é\nX" + "x" * 200, rationale="ok\x07\x1b[31m " + "y" * 900,
                targets_detail=[{"id": "H001", "lat": 37.123456, "lon": -76.654321}], pin="424242")
    subject, message = compose([evil], "")
    assert len(subject) <= SUBJECT_MAX and subject.isascii() and "\n" not in subject
    assert "424242" not in subject + message and "37.123456" not in message
    assert "\x07" not in message and "\x1b" not in message and "the DEPTH studio -> Approvals" in message
    assert max(len(line) for line in message.splitlines()) < 460       # the 900-char rationale is capped


def test_the_json_log_line_feeds_the_cloudwatch_metric(monkeypatch, capsys):
    monkeypatch.setenv("DEPTH_LOG_JSON", "1")
    n, _ = _notifier()
    n.notify("approval.requested", _req())
    n.flush_all()
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert '"evt": "notify"' in line and '"status": "sent"' in line and '"channel": "sns"' in line


def test_the_service_emails_the_request_not_the_webhook_copy_with_positions(monkeypatch):
    from src.dashboard import service as S
    seen: list[tuple[str, dict]] = []
    monkeypatch.setattr(S._SNS, "notify", lambda event, data: seen.append((event, data)) or True)
    monkeypatch.setattr(S._WEBHOOKS, "emit", lambda event, data: 0)
    S._notify("approval.requested", _req())
    assert seen and seen[0][0] == "approval.requested" and "targets_detail" not in seen[0][1]
