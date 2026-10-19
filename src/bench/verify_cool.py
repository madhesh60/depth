"""
verify_cool.py — check a DEPTH server's COOL execution receipt from anywhere ("Verified COOL on Graviton").

    python -m src.bench.verify_cool https://<cloudfront-domain>            # live server
    python -m src.bench.verify_cool receipt.json                           # a saved /api/cool response
        [--analyze]            # also run one sample through /api/analyze and check its provenance names
                               # the same signed document (the output is tied to the attested machine)
        [--md docs/cool_verification.md] [--json out.json]

What is checked (exit code 0 only if every REQUIRED check passes):

* AWS signed the instance identity document — RSA-SHA256 ``signature`` against AWS's published RSA
  certificate for the region (pure Python, ``cryptography``); the PKCS7 ``rsa2048`` signature as well when
  ``openssl`` is on the PATH. A server cannot forge either.
* From the signed document: ``architecture`` is ``arm64`` (Graviton), the instance type is a Graviton
  type, and the instance was launched from an AMI carrying an AWS Marketplace product code — the COOL
  listing's code once it is pinned in ``COOL_PRODUCT_CODES`` (otherwise compare it yourself with
  ``aws ec2 describe-images --image-ids <ami> --query Images[0].ProductCodes``).
* From the server process (self-reported by the open-source server code): ``cv2`` is imported from
  ``/opt/cool`` and every OpenCV shared object mapped into the process lives under ``/opt/cool``.

The certificates below were copied from the EC2 User Guide page "AWS public certificates for instance
identity document signatures" (docs.aws.amazon.com/AWSEC2/latest/UserGuide/regions-certs.html) on
2026-10-19; both are self-signed, and ``tests/test_cool_attest.py`` checks their self-signatures so a
transcription error cannot slip in.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

AWS_IID_CERTS = {
    "us-east-1": {
        # RSA certificate for the base64-encoded 'signature' (valid 2024-04-29 .. 2029-04-28)
        "rsa": """-----BEGIN CERTIFICATE-----
MIIDITCCAoqgAwIBAgIUE1y2NIKCU+Rg4uu4u32koG9QEYIwDQYJKoZIhvcNAQEL
BQAwXDELMAkGA1UEBhMCVVMxGTAXBgNVBAgTEFdhc2hpbmd0b24gU3RhdGUxEDAO
BgNVBAcTB1NlYXR0bGUxIDAeBgNVBAoTF0FtYXpvbiBXZWIgU2VydmljZXMgTExD
MB4XDTI0MDQyOTE3MzQwMVoXDTI5MDQyODE3MzQwMVowXDELMAkGA1UEBhMCVVMx
GTAXBgNVBAgTEFdhc2hpbmd0b24gU3RhdGUxEDAOBgNVBAcTB1NlYXR0bGUxIDAe
BgNVBAoTF0FtYXpvbiBXZWIgU2VydmljZXMgTExDMIGfMA0GCSqGSIb3DQEBAQUA
A4GNADCBiQKBgQCHvRjf/0kStpJ248khtIaN8qkDN3tkw4VjvA9nvPl2anJO+eIB
UqPfQG09kZlwpWpmyO8bGB2RWqWxCwuB/dcnIob6w420k9WY5C0IIGtDRNauN3ku
vGXkw3HEnF0EjYr0pcyWUvByWY4KswZV42X7Y7XSS13hOIcL6NLA+H94/QIDAQAB
o4HfMIHcMAsGA1UdDwQEAwIHgDAdBgNVHQ4EFgQUJdbMCBXKtvCcWdwUUizvtUF2
UTgwgZkGA1UdIwSBkTCBjoAUJdbMCBXKtvCcWdwUUizvtUF2UTihYKReMFwxCzAJ
BgNVBAYTAlVTMRkwFwYDVQQIExBXYXNoaW5ndG9uIFN0YXRlMRAwDgYDVQQHEwdT
ZWF0dGxlMSAwHgYDVQQKExdBbWF6b24gV2ViIFNlcnZpY2VzIExMQ4IUE1y2NIKC
U+Rg4uu4u32koG9QEYIwEgYDVR0TAQH/BAgwBgEB/wIBADANBgkqhkiG9w0BAQsF
AAOBgQAlxSmwcWnhT4uAeSinJuz+1BTcKhVSWb5jT8pYjQb8ZoZkXXRGb09mvYeU
NeqOBr27rvRAnaQ/9LUQf72+SahDFuS4CMI8nwowytqbmwquqFr4dxA/SDADyRiF
ea1UoMuNHTY49J/1vPomqsVn7mugTp+TbjqCfOJTpu0temHcFA==
-----END CERTIFICATE-----
""",
        # RSA-2048 certificate for the PKCS7 'rsa2048' signature (valid 2015-08-14 .. 2195-01-17)
        "rsa2048": """-----BEGIN CERTIFICATE-----
MIIEEjCCAvqgAwIBAgIJALFpzEAVWaQZMA0GCSqGSIb3DQEBCwUAMFwxCzAJBgNV
BAYTAlVTMRkwFwYDVQQIExBXYXNoaW5ndG9uIFN0YXRlMRAwDgYDVQQHEwdTZWF0
dGxlMSAwHgYDVQQKExdBbWF6b24gV2ViIFNlcnZpY2VzIExMQzAgFw0xNTA4MTQw
ODU5MTJaGA8yMTk1MDExNzA4NTkxMlowXDELMAkGA1UEBhMCVVMxGTAXBgNVBAgT
EFdhc2hpbmd0b24gU3RhdGUxEDAOBgNVBAcTB1NlYXR0bGUxIDAeBgNVBAoTF0Ft
YXpvbiBXZWIgU2VydmljZXMgTExDMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIB
CgKCAQEAjS2vqZu9mEOhOq+0bRpAbCUiapbZMFNQqRg7kTlr7Cf+gDqXKpHPjsng
SfNz+JHQd8WPI+pmNs+q0Z2aTe23klmf2U52KH9/j1k8RlIbap/yFibFTSedmegX
E5r447GbJRsHUmuIIfZTZ/oRlpuIIO5/Vz7SOj22tdkdY2ADp7caZkNxhSP915fk
2jJMTBUOzyXUS2rBU/ulNHbTTeePjcEkvzVYPahD30TeQ+/A+uWUu89bHSQOJR8h
Um4cFApzZgN3aD5j2LrSMu2pctkQwf9CaWyVznqrsGYjYOY66LuFzSCXwqSnFBfv
fFBAFsjCgY24G2DoMyYkF3MyZlu+rwIDAQABo4HUMIHRMAsGA1UdDwQEAwIHgDAd
BgNVHQ4EFgQUrynSPp4uqSECwy+PiO4qyJ8TWSkwgY4GA1UdIwSBhjCBg4AUrynS
Pp4uqSECwy+PiO4qyJ8TWSmhYKReMFwxCzAJBgNVBAYTAlVTMRkwFwYDVQQIExBX
YXNoaW5ndG9uIFN0YXRlMRAwDgYDVQQHEwdTZWF0dGxlMSAwHgYDVQQKExdBbWF6
b24gV2ViIFNlcnZpY2VzIExMQ4IJALFpzEAVWaQZMBIGA1UdEwEB/wQIMAYBAf8C
AQAwDQYJKoZIhvcNAQELBQADggEBADW/s8lXijwdP6NkEoH1m9XLrvK4YTqkNfR6
er/uRRgTx2QjFcMNrx+g87gAml11z+D0crAZ5LbEhDMs+JtZYR3ty0HkDk6SJM85
haoJNAFF7EQ/zCp1EJRIkLLsC7bcDL/Eriv1swt78/BB4RnC9W9kSp/sxd5svJMg
N9a6FAplpNRsWAnbP8JBlAP93oJzblX2LQXgykTghMkQO7NaY5hg/H5o4dMPclTK
lYGqlFUCH6A2vdrxmpKDLmTn5//5pujdD2MN0df6sZWtxwZ0osljV4rDjm9Q3VpA
NWIsDEcp3GUB4proOR+C7PNkY+VGODitBOw09qBGosCBstwyEqY=
-----END CERTIFICATE-----
""",
    },
}

# AWS Marketplace product code(s) of the COOL listing(s), read from `aws ec2 describe-images` for the AMI
# the demo was launched from (filled in on deploy day; until then the check reports the code it sees).
COOL_PRODUCT_CODES: dict[str, str] = {}

GRAVITON_TYPE = re.compile(r"^(a1|[a-z]+\d+g[a-z]*)\.", re.I)        # c8g, m7gd, t4g, x2gd, im4gn, hpc7g …


def rsa_signature_ok(document: str, signature_b64: str, region: str) -> tuple[bool, str]:
    """RSA-SHA256 over the exact document bytes with AWS's RSA certificate for ``region``."""
    pem = AWS_IID_CERTS.get(region, {}).get("rsa")
    if not pem:
        return False, f"no AWS RSA certificate bundled for region {region!r}"
    from cryptography import x509
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    key = x509.load_pem_x509_certificate(pem.encode()).public_key()
    try:
        key.verify(base64.b64decode(signature_b64), document.encode(), padding.PKCS1v15(), hashes.SHA256())
        return True, "RSA-SHA256 signature verifies against AWS's published certificate"
    except InvalidSignature:
        return False, "RSA signature does NOT verify - the document was altered or not signed by AWS"
    except Exception as e:
        return False, f"could not check the RSA signature: {type(e).__name__}: {e}"


def pkcs7_signature_ok(document: str, rsa2048_b64: str, region: str) -> tuple[bool | None, str]:
    """The PKCS7 RSA-2048 signature via ``openssl smime -verify`` (None when openssl is unavailable)."""
    pem = AWS_IID_CERTS.get(region, {}).get("rsa2048")
    exe = shutil.which("openssl")
    if not pem or not exe:
        return None, "skipped (" + ("no openssl on PATH" if not exe else f"no RSA-2048 certificate for {region}") + ")"
    with tempfile.TemporaryDirectory() as d:
        sig, cert = Path(d) / "rsa2048.p7", Path(d) / "cert.pem"
        body = "\n".join(rsa2048_b64[i:i + 64] for i in range(0, len(rsa2048_b64), 64))
        sig.write_text(f"-----BEGIN PKCS7-----\n{body}\n-----END PKCS7-----\n")
        cert.write_text(pem)
        # -nointern: the signer must be AWS's certificate given here, never one embedded in the signature
        # (AWS's documented command omits it, which would accept a forged PKCS7 carrying its own cert)
        r = subprocess.run([exe, "smime", "-verify", "-in", str(sig), "-inform", "PEM", "-certfile", str(cert),
                            "-nointern", "-noverify"], capture_output=True, timeout=30)
    if r.returncode != 0:
        return False, "PKCS7 signature does NOT verify: " + r.stderr.decode(errors="replace").strip()[:200]
    norm = lambda b: b.replace(b"\r\n", b"\n").strip()          # S/MIME may canonicalise line ends
    if norm(r.stdout) != norm(document.encode()):
        return False, "PKCS7 signature verifies, but it signs a different document"
    return True, "PKCS7 RSA-2048 signature verifies (openssl) and signs exactly this document"


def _get(url: str, timeout: float = 30, data: bytes | None = None) -> dict:
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET",
                                 headers={"User-Agent": "depth-verify-cool/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def verify(receipt: dict, analyze: dict | None = None) -> dict:
    checks: list[dict] = []

    def add(name: str, ok, detail: str, required: bool = True):
        checks.append({"check": name, "ok": ok, "required": required, "detail": detail})

    sid = receipt.get("aws_signed_identity")
    if not sid:
        add("AWS-signed instance identity present", False, "the server is not on EC2 (or IMDS is blocked)")
        return {"passed": False, "checks": checks}
    doc = sid["document"]
    c = json.loads(doc)
    region = c.get("region", "")
    ok, why = rsa_signature_ok(doc, sid["signature"], region)
    add("AWS signed this identity document (RSA-SHA256)", ok, why)
    ok7, why7 = pkcs7_signature_ok(doc, sid.get("rsa2048", ""), region)
    add("AWS signed this identity document (PKCS7 RSA-2048)", ok7, why7, required=False)
    add("architecture is arm64 (AWS Graviton)", c.get("architecture") == "arm64", f"architecture = {c.get('architecture')}")
    it = c.get("instanceType") or ""
    gen = re.match(r"^[a-z]+(\d+)g", it)
    add("instance type is a Graviton type", bool(GRAVITON_TYPE.match(it)),
        f"instanceType = {it}" + (f" (Graviton{4 if gen and gen.group(1) == '8' else ''} family)" if gen else ""))
    codes = c.get("marketplaceProductCodes") or []
    if COOL_PRODUCT_CODES:
        hit = [x for x in codes if x in COOL_PRODUCT_CODES]
        add("launched from the COOL Marketplace AMI (product code)", bool(hit),
            f"marketplaceProductCodes = {codes}; COOL = {sorted(COOL_PRODUCT_CODES)}")
    else:
        add("launched from an AWS Marketplace AMI (product code present)", bool(codes),
            f"marketplaceProductCodes = {codes} - compare with: aws ec2 describe-images --image-ids "
            f"{c.get('imageId')} --query Images[0].ProductCodes")
    p = receipt.get("process") or {}
    add("cv2 is imported from /opt/cool", bool(p.get("cv2_from_cool")), f"cv2.__file__ = {p.get('cv2_file')}")
    if p.get("available"):
        add("every OpenCV library mapped in the server process is under /opt/cool", bool(p.get("all_opencv_from_cool")),
            f"{len(p.get('opencv_libs') or [])} OpenCV object(s) mapped; outside /opt/cool: "
            f"{p.get('opencv_outside_cool') or 'none'}")
    else:
        add("process memory map available", False, p.get("reason", "no map"))
    if analyze is not None:
        prov = analyze.get("provenance") or {}
        att = (prov.get("attestation") or {}).get("document_sha256")
        add("a live analysis names the same signed document (output tied to this machine)",
            att == sid.get("document_sha256") == hashlib.sha256(doc.encode()).hexdigest(),
            f"provenance attestation {str(att)[:16]}… vs receipt {str(sid.get('document_sha256'))[:16]}…",
            required=False)
    passed = all(x["ok"] for x in checks if x["required"])
    return {"passed": passed, "checks": checks, "claims": {k: c.get(k) for k in
            ("instanceType", "architecture", "imageId", "marketplaceProductCodes", "region", "instanceId",
             "pendingTime")}, "cool_build": receipt.get("cool_build"), "document_sha256": sid.get("document_sha256")}


def markdown(res: dict, source: str) -> str:
    mark = {True: "PASS", False: "FAIL", None: "skip"}
    L = ["# COOL verification — `python -m src.bench.verify_cool`", "",
         f"_Source: `{source}` · checked {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} · "
         f"result: **{'VERIFIED' if res['passed'] else 'NOT VERIFIED'}**_", "",
         "| check | result | evidence |", "|---|---|---|"]
    for x in res["checks"]:
        L.append(f"| {x['check']}{'' if x['required'] else ' _(extra)_'} | **{mark[x['ok']]}** | {x['detail']} |")
    if res.get("claims"):
        L += ["", "Signed claims (AWS instance identity document): " +
              " · ".join(f"`{k}` = {v}" for k, v in res["claims"].items()), "",
              f"Document SHA-256: `{res.get('document_sha256')}` — every DEPTH result's provenance carries it."]
    b = res.get("cool_build") or {}
    if b:
        L += ["", "COOL build, read from the machine (not typed in):", ""]
        for k in ("version_files", "dpkg", "python_dists", "cool_venvs"):
            if b.get(k):
                L.append(f"* `{k}`: {b[k]}")
        for ln in (b.get("cv2_build_lines") or [])[:12]:
            L.append(f"  * `{ln}`")
    L += ["", "Trust model: AWS's signature proves which machine this is (AMI + Marketplace product code, "
          "arm64, instance type). The library map is reported by the open-source server code; re-run the "
          "fleet (`infra/bench_fleet.sh`) in your own account to see the same on a fresh machine.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Verify a DEPTH server's COOL execution receipt.")
    ap.add_argument("source", help="https://<server> or a saved /api/cool JSON file")
    ap.add_argument("--analyze", action="store_true", help="also run one sample and check its provenance")
    ap.add_argument("--md", help="write a Markdown report here")
    ap.add_argument("--json", help="write the JSON result here")
    a = ap.parse_args()
    analyze = None
    if a.source.startswith(("http://", "https://")):
        base = a.source.rstrip("/")
        receipt = _get(base + "/api/cool")
        if a.analyze:
            sid = (_get(base + "/api/samples") or [{}])
            sid = (sid[0] if isinstance(sid, list) else (sid.get("samples") or [{}])[0]).get("id")
            analyze = _get(f"{base}/api/analyze?sample={sid}", timeout=120, data=b"")
    else:
        receipt = json.loads(Path(a.source).read_text(encoding="utf-8"))
    res = verify(receipt, analyze)
    for x in res["checks"]:
        print(f"  [{ {True: 'PASS', False: 'FAIL', None: 'skip'}[x['ok']] }] {x['check']} - {x['detail']}")
    print("VERIFIED: COOL on Graviton" if res["passed"] else "NOT VERIFIED")
    if a.md:
        Path(a.md).write_text(markdown(res, a.source), encoding="utf-8")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2), encoding="utf-8")
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
