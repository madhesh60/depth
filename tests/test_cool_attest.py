"""
Tests for the COOL execution receipt (src/bench/cool_attest.py), its verifier (src/bench/verify_cool.py)
and the origin guard — no AWS needed: AWS's signature is simulated with a throw-away key standing in for
AWS's certificate, and the process map is a synthetic /proc/self/maps dump.
"""
from __future__ import annotations

import base64
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.serialization import pkcs7
from cryptography.x509.oid import NameOID

from src.bench import cool_attest, verify_cool

COOL_MAPS = """\
aaaa-bbbb r-xp 00000000 103:01 11 /opt/cool/venvs/python_3.12/lib/python3.12/site-packages/cv2/cv2.cpython-312-aarch64-linux-gnu.so
aaaa-bbbb r-xp 00000000 103:01 12 /opt/cool/lib/libopencv_dnn.so.500
aaaa-bbbb r-xp 00000000 103:01 13 /opt/cool/lib/libkleidicv.so
aaaa-bbbb r-xp 00000000 103:01 14 /usr/lib/aarch64-linux-gnu/libc.so.6
aaaa-bbbb rw-p 00000000 00:00 0 [heap]
"""
STRAY = "aaaa-bbbb r-xp 00000000 103:01 15 /home/u/.venv/lib/python3.12/site-packages/cv2/cv2.abi3.so\n"


def _self_signed():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.ORGANIZATION_NAME, "test-aws")])
    now = dt.datetime.now(dt.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(days=1))
            .not_valid_after(now + dt.timedelta(days=1)).sign(key, hashes.SHA256()))
    return key, cert


G4 = "aajkmdd4qo3r7yhqg61a7aah9"          # the COOL Graviton4 listing (describe-images, 2026-10-19)


def _document(arch="arm64", itype="c8g.large", codes=(G4,)):
    return json.dumps({"accountId": "000000000000", "architecture": arch, "availabilityZone": "us-east-1a",
                       "imageId": "ami-0abc", "instanceId": "i-0abc", "instanceType": itype,
                       "marketplaceProductCodes": list(codes), "pendingTime": "2026-10-20T00:00:00Z",
                       "privateIp": "172.31.0.10", "region": "us-east-1", "version": "2017-09-30"}, indent=2)


def _receipt(key, cert, doc, maps=COOL_MAPS, cv2_file=None, nocerts=True):
    sig = base64.b64encode(key.sign(doc.encode(), padding.PKCS1v15(), hashes.SHA256())).decode()
    opts = [pkcs7.PKCS7Options.Binary] + ([pkcs7.PKCS7Options.NoCerts] if nocerts else [])
    p7 = pkcs7.PKCS7SignatureBuilder().set_data(doc.encode()).add_signer(cert, key, hashes.SHA256()) \
        .sign(serialization.Encoding.PEM, opts).decode()
    flat = "".join(line for line in p7.splitlines() if "PKCS7" not in line)
    body = "\n".join(flat[i:i + 76] for i in range(0, len(flat), 76))   # as IMDS serves it: 76-column lines
    cv2_file = cv2_file or "/opt/cool/venvs/python_3.12/lib/python3.12/site-packages/cv2/__init__.py"
    return {"aws_signed_identity": {"document": doc, "signature": sig, "rsa2048": body,
                                    "document_sha256": cool_attest.document_sha256(doc)},
            "process": {"cv2_file": cv2_file, "cv2_from_cool": cv2_file.startswith("/opt/cool/"),
                        **cool_attest.mapped_libraries(maps, "/opt/cool")}}


@pytest.fixture
def fake_aws(monkeypatch):
    key, cert = _self_signed()
    pem = cert.public_bytes(serialization.Encoding.PEM).decode()
    monkeypatch.setattr(verify_cool, "AWS_IID_CERTS", {"us-east-1": {"rsa": pem, "rsa2048": pem}})
    return key, cert


def test_bundled_aws_certificates_are_intact():
    for region, certs in verify_cool.AWS_IID_CERTS.items():
        for kind, pem in certs.items():
            c = x509.load_pem_x509_certificate(pem.encode())
            assert "Amazon Web Services LLC" in c.subject.rfc4514_string() and c.issuer == c.subject
            # self-signed: a single wrong character in the transcription breaks this
            c.public_key().verify(c.signature, c.tbs_certificate_bytes, padding.PKCS1v15(), c.signature_hash_algorithm)
            assert c.not_valid_after_utc > dt.datetime(2026, 11, 10, tzinfo=dt.timezone.utc), (region, kind)


def test_process_map_classifies_opencv_sources():
    m = cool_attest.mapped_libraries(COOL_MAPS, "/opt/cool")
    assert m["all_opencv_from_cool"] and not m["opencv_outside_cool"]
    assert len(m["opencv_libs"]) == 2 and any("libkleidicv" in f for f in m["cool_tree_libs"])
    m2 = cool_attest.mapped_libraries(COOL_MAPS + STRAY, "/opt/cool")
    assert not m2["all_opencv_from_cool"] and m2["opencv_outside_cool"] == [STRAY.split()[-1]]
    assert not cool_attest.mapped_libraries("", "/opt/cool")["all_opencv_from_cool"]   # nothing mapped ≠ COOL


def test_verifier_accepts_a_genuine_cool_receipt(fake_aws):
    key, cert = fake_aws
    res = verify_cool.verify(_receipt(key, cert, _document()))
    assert res["passed"], res["checks"]
    assert res["claims"]["instanceType"] == "c8g.large"


@pytest.mark.parametrize("why", ["tampered", "x86", "no_code", "stray_opencv", "not_cool_python"])
def test_verifier_rejects(fake_aws, why):
    key, cert = fake_aws
    doc = _document(arch="x86_64", itype="c8i.xlarge") if why == "x86" else \
        _document(codes=()) if why == "no_code" else _document()
    r = _receipt(key, cert, doc, maps=COOL_MAPS + (STRAY if why == "stray_opencv" else ""),
                 cv2_file="/usr/lib/python3/dist-packages/cv2/__init__.py" if why == "not_cool_python" else None)
    if why == "tampered":                                  # claim a different machine after signing
        r["aws_signed_identity"]["document"] = doc.replace("c8g.large", "c8g.48xlarge")
    res = verify_cool.verify(r)
    assert not res["passed"]
    failed = [c["check"] for c in res["checks"] if c["required"] and not c["ok"]]
    assert failed, res["checks"]


def test_pinned_cool_product_code_is_enforced(fake_aws):
    key, cert = fake_aws
    assert G4 in verify_cool.COOL_PRODUCT_CODES                        # pinned from describe-images
    assert verify_cool.verify(_receipt(key, cert, _document()))["passed"]
    # a different Marketplace AMI (any other product) on Graviton with a stand-in /opt/cool: NOT COOL
    res = verify_cool.verify(_receipt(key, cert, _document(codes=("someotheramicode",))))
    assert not res["passed"]
    assert [c for c in res["checks"] if not c["ok"] and c["required"]][0]["check"].startswith("launched from the COOL")


@pytest.mark.skipif(not shutil.which("openssl"), reason="openssl not on PATH")
def test_pkcs7_check_refuses_a_signer_embedded_in_the_signature(fake_aws):
    key, cert = fake_aws
    doc = _document()
    ok, why = verify_cool.pkcs7_signature_ok(doc, _receipt(key, cert, doc)["aws_signed_identity"]["rsa2048"], "us-east-1")
    assert ok is True, why
    # a forger signs with their own key and embeds their own certificate: must NOT verify
    fkey, fcert = _self_signed()
    forged = _receipt(fkey, fcert, doc, nocerts=False)["aws_signed_identity"]["rsa2048"]
    ok2, why2 = verify_cool.pkcs7_signature_ok(doc, forged, "us-east-1")
    assert ok2 is False, why2


def test_off_ec2_degrades_without_network(monkeypatch):
    monkeypatch.setattr(cool_attest, "on_ec2", lambda: False)
    monkeypatch.setattr(cool_attest, "_IDENTITY", {})
    s = cool_attest.summary()
    assert s["on_ec2"] is False and s["document_sha256"] is None
    r = cool_attest.receipt()
    assert r["aws_signed_identity"] is None and r["process"]["cv2_file"]
    assert "cv2_build_lines" in r["cool_build"]


def test_api_serves_receipt_and_health_summary(monkeypatch):
    from fastapi.testclient import TestClient
    import src.dashboard.app as app_mod
    monkeypatch.setattr(cool_attest, "on_ec2", lambda: False)
    monkeypatch.setattr(cool_attest, "_IDENTITY", {})
    c = TestClient(app_mod.app)
    assert c.get("/api/cool").json()["on_ec2"] is False
    assert c.get("/api/health").json()["cool"]["receipt"] == "/api/cool"


def test_origin_guard_blocks_requests_without_the_cloudfront_secret(monkeypatch):
    from fastapi.testclient import TestClient
    import src.dashboard.app as app_mod
    c = TestClient(app_mod.app)
    monkeypatch.delenv("DEPTH_ORIGIN_SECRET", raising=False)
    assert c.get("/api/samples").status_code == 200                 # guard off when no secret is configured
    monkeypatch.setenv("DEPTH_ORIGIN_SECRET", "cf-only")
    assert c.get("/api/samples").status_code == 403
    assert c.get("/api/samples", headers={"X-Origin-Verify": "nope"}).status_code == 403
    assert c.get("/api/samples", headers={"X-Origin-Verify": "cf-only"}).status_code == 200


def test_caddy_sends_the_origin_secret_because_uvicorn_trusts_loopback(monkeypatch):
    """PUBLIC_HTTPS=1: Caddy proxies from loopback with X-Forwarded-For, which uvicorn trusts by default, so the
    guard sees the visitor, not loopback. Without the secret every public request was refused (403)."""
    from fastapi.testclient import TestClient
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
    import src.dashboard.app as app_mod
    monkeypatch.setenv("DEPTH_ORIGIN_SECRET", "box-secret")
    c = TestClient(ProxyHeadersMiddleware(app_mod.app, trusted_hosts="127.0.0.1"), client=("127.0.0.1", 50000))
    assert c.get("/api/samples").status_code == 200                       # an on-box check
    xff = {"X-Forwarded-For": "203.0.113.9"}
    assert c.get("/api/samples", headers=xff).status_code == 403          # a proxy without the secret
    assert c.get("/api/samples", headers={**xff, "X-Origin-Verify": "box-secret"}).status_code == 200
    deploy = (Path(__file__).resolve().parents[1] / "infra" / "deploy.sh").read_text(encoding="utf-8")
    caddy = deploy.split('"DEPTH HTTPS (Caddy)" ssm_run', 1)[1].split("DEPTH restarted", 1)[0]
    assert ". /etc/depth/origin.env" in caddy and "header_up X-Origin-Verify %s" in caddy
