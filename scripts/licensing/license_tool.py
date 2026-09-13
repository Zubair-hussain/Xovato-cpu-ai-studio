"""Offline, key-based licensing for Xovato models.

Trust chain (all Ed25519, all offline):

    ROOT  (you)         keygen --role root      -> root_private.pem / root_public.pem
      |  signs
    ADMIN (delegate)    keygen --role admin     -> admin_private.pem / admin_public.pem
      |                 admin-add (by root)     -> admin_cert.json   (root signs admin pubkey)
      |  signs
    LICENSE             issue   (by admin/root) -> license.json      (bound to a machine)
                        verify                  -> checks chain + expiry + this machine

The machine binding is a fingerprint of the host (hostname + hardware id), so a
license issued for one laptop does not validate on another. Root is the single
trust anchor: ship ``root_public.pem`` with the product and verify against it.

CLI:

    python -m scripts.licensing.license_tool keygen  --role root  --out keys/
    python -m scripts.licensing.license_tool keygen  --role admin --out keys/admin/
    python -m scripts.licensing.license_tool admin-add \
        --root-key keys/root_private.pem --admin-pub keys/admin/admin_public.pem \
        --name "Alice" --out keys/admin/admin_cert.json
    python -m scripts.licensing.license_tool fingerprint
    python -m scripts.licensing.license_tool issue \
        --signer-key keys/admin/admin_private.pem --admin-cert keys/admin/admin_cert.json \
        --subject "ACME Studio" --machine <fingerprint> --features tts,image,shorts \
        --days 365 --out license.json
    python -m scripts.licensing.license_tool verify \
        --license license.json --root-pub keys/root_public.pem
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


class LicenseError(RuntimeError):
    """Raised when a license is missing, malformed, expired, or fails verification."""


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #


def generate_keypair() -> tuple[bytes, bytes]:
    """Return (private_pem, public_pem) for a fresh Ed25519 key."""
    private = Ed25519PrivateKey.generate()
    private_pem = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


def _load_private(path: Path) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(Path(path).read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise LicenseError(f"{path} is not an Ed25519 private key.")
    return key


def _load_public(path: Path) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(Path(path).read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise LicenseError(f"{path} is not an Ed25519 public key.")
    return key


def _public_pem_str(public_pem: bytes) -> str:
    return public_pem.decode("ascii")


def _sign(private: Ed25519PrivateKey, payload: dict) -> str:
    message = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(private.sign(message)).decode("ascii")


def _verify_sig(public: Ed25519PublicKey, payload: dict, signature_b64: str) -> None:
    message = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    try:
        public.verify(base64.b64decode(signature_b64), message)
    except (InvalidSignature, ValueError, TypeError) as error:
        raise LicenseError("Signature verification failed.") from error


# --------------------------------------------------------------------------- #
# Machine binding
# --------------------------------------------------------------------------- #


def machine_fingerprint() -> str:
    """A stable, non-reversible id for this machine (hostname + hardware node)."""
    import platform

    raw = f"{platform.node()}|{uuid.getnode():x}|{platform.system()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


# --------------------------------------------------------------------------- #
# Admin certificates (root delegates issuing power to an admin)
# --------------------------------------------------------------------------- #


def make_admin_cert(root_private: Ed25519PrivateKey, admin_public_pem: bytes, name: str) -> dict:
    """Root signs an admin's public key, granting it the right to issue licenses."""
    body = {
        "type": "admin-cert",
        "name": name,
        "admin_public_key": _public_pem_str(admin_public_pem),
        "issued_at": datetime.now(UTC).isoformat(),
    }
    return {"body": body, "signature": _sign(root_private, body)}


def _verify_admin_cert(root_public: Ed25519PublicKey, cert: dict) -> Ed25519PublicKey:
    if cert.get("body", {}).get("type") != "admin-cert":
        raise LicenseError("Not an admin certificate.")
    _verify_sig(root_public, cert["body"], cert["signature"])
    return serialization.load_pem_public_key(cert["body"]["admin_public_key"].encode("ascii"))


# --------------------------------------------------------------------------- #
# Licenses
# --------------------------------------------------------------------------- #


def issue_license(
    signer_private: Ed25519PrivateKey,
    *,
    subject: str,
    machine: str,
    features: list[str],
    days: int,
    admin_cert: dict | None = None,
) -> dict:
    """Sign a license. ``admin_cert`` is embedded when an admin (not root) signs,
    so the verifier can walk admin -> root."""
    now = datetime.now(UTC)
    body = {
        "type": "license",
        "subject": subject,
        "machine": machine,
        "features": sorted(features),
        "issued_at": now.isoformat(),
        "expires_at": (now + timedelta(days=days)).isoformat(),
        "copyright": "(c) Zubair Hussain - Xovato AI Studio",
    }
    license_doc = {"body": body, "signature": _sign(signer_private, body)}
    if admin_cert is not None:
        license_doc["admin_cert"] = admin_cert
    return license_doc


def verify_license(
    license_doc: dict,
    root_public: Ed25519PublicKey,
    *,
    machine: str | None = None,
    now: datetime | None = None,
) -> dict:
    """Verify signature chain, expiry and machine binding. Returns the license body
    on success; raises :class:`LicenseError` otherwise."""
    body = license_doc.get("body")
    if not body or body.get("type") != "license":
        raise LicenseError("Malformed license.")

    # 1. Establish the key that signed the license.
    admin_cert = license_doc.get("admin_cert")
    if admin_cert is not None:
        signer_public = _verify_admin_cert(root_public, admin_cert)  # admin, vouched for by root
    else:
        signer_public = root_public  # root signed directly

    # 2. License signature.
    _verify_sig(signer_public, body, license_doc["signature"])

    # 3. Expiry.
    now = now or datetime.now(UTC)
    if now > datetime.fromisoformat(body["expires_at"]):
        raise LicenseError(f"License expired on {body['expires_at']}.")

    # 4. Machine binding.
    current = machine or machine_fingerprint()
    if body["machine"] not in ("*", current):
        raise LicenseError("License is not valid for this machine.")

    return body


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _cmd_keygen(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    private_pem, public_pem = generate_keypair()
    (out / f"{args.role}_private.pem").write_bytes(private_pem)
    (out / f"{args.role}_public.pem").write_bytes(public_pem)
    print(f"Wrote {args.role}_private.pem and {args.role}_public.pem to {out}")
    print("Keep the PRIVATE key secret. Ship only the public key with the product.")
    return 0


def _cmd_admin_add(args) -> int:
    root = _load_private(Path(args.root_key))
    admin_pub_pem = Path(args.admin_pub).read_bytes()
    cert = make_admin_cert(root, admin_pub_pem, args.name)
    Path(args.out).write_text(json.dumps(cert, indent=2), encoding="utf-8")
    print(f"Signed admin certificate for '{args.name}' -> {args.out}")
    return 0


def _cmd_fingerprint(args) -> int:
    print(machine_fingerprint())
    return 0


def _cmd_issue(args) -> int:
    signer = _load_private(Path(args.signer_key))
    admin_cert = json.loads(Path(args.admin_cert).read_text(encoding="utf-8")) if args.admin_cert else None
    license_doc = issue_license(
        signer,
        subject=args.subject,
        machine=args.machine or machine_fingerprint(),
        features=[f.strip() for f in args.features.split(",") if f.strip()],
        days=args.days,
        admin_cert=admin_cert,
    )
    Path(args.out).write_text(json.dumps(license_doc, indent=2), encoding="utf-8")
    print(f"Issued license for '{args.subject}' -> {args.out}")
    return 0


def _cmd_verify(args) -> int:
    root_public = _load_public(Path(args.root_pub))
    license_doc = json.loads(Path(args.license).read_text(encoding="utf-8"))
    try:
        body = verify_license(license_doc, root_public, machine=args.machine)
    except LicenseError as error:
        print(f"INVALID: {error}")
        return 1
    print("VALID")
    print(f"  subject : {body['subject']}")
    print(f"  features: {', '.join(body['features'])}")
    print(f"  expires : {body['expires_at']}")
    print(f"  {body['copyright']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="license_tool", description="Offline Xovato model licensing.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("keygen", help="Create an Ed25519 keypair (role: root or admin).")
    p.add_argument("--role", choices=["root", "admin"], required=True)
    p.add_argument("--out", default="keys")
    p.set_defaults(func=_cmd_keygen)

    p = sub.add_parser("admin-add", help="Root signs an admin's public key (delegates issuing).")
    p.add_argument("--root-key", required=True)
    p.add_argument("--admin-pub", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out", default="admin_cert.json")
    p.set_defaults(func=_cmd_admin_add)

    p = sub.add_parser("fingerprint", help="Print this machine's fingerprint.")
    p.set_defaults(func=_cmd_fingerprint)

    p = sub.add_parser("issue", help="Issue (sign) a license for a machine.")
    p.add_argument("--signer-key", required=True, help="Admin (or root) private key.")
    p.add_argument("--admin-cert", help="Admin certificate; omit when root signs directly.")
    p.add_argument("--subject", required=True)
    p.add_argument("--machine", help="Target machine fingerprint (default: this machine; '*' = any).")
    p.add_argument("--features", default="tts,image,audio,shorts")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--out", default="license.json")
    p.set_defaults(func=_cmd_issue)

    p = sub.add_parser("verify", help="Verify a license against the root public key.")
    p.add_argument("--license", required=True)
    p.add_argument("--root-pub", required=True)
    p.add_argument("--machine", help="Override the machine fingerprint to check against.")
    p.set_defaults(func=_cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
