"""Tests for the offline Ed25519 licensing chain."""

from datetime import UTC, datetime, timedelta

import pytest
from cryptography.hazmat.primitives import serialization

from scripts.licensing.license_tool import (
    LicenseError,
    generate_keypair,
    issue_license,
    machine_fingerprint,
    make_admin_cert,
    verify_license,
)


def _priv(pem):
    return serialization.load_pem_private_key(pem, password=None)


def _pub(pem):
    return serialization.load_pem_public_key(pem)


def test_fingerprint_stable_and_hex():
    fp = machine_fingerprint()
    assert fp == machine_fingerprint()
    assert len(fp) == 32 and all(c in "0123456789abcdef" for c in fp)


def test_root_signed_license_verifies():
    root_priv, root_pub = generate_keypair()
    lic = issue_license(_priv(root_priv), subject="Acme", machine="*", features=["tts"], days=30)
    body = verify_license(lic, _pub(root_pub))
    assert body["subject"] == "Acme"
    assert "Zubair" in body["copyright"]


def test_admin_chain_verifies():
    root_priv, root_pub = generate_keypair()
    admin_priv, admin_pub = generate_keypair()
    cert = make_admin_cert(_priv(root_priv), admin_pub, "Alice")
    lic = issue_license(_priv(admin_priv), subject="Beta", machine="*", features=["image"], days=10, admin_cert=cert)
    body = verify_license(lic, _pub(root_pub))
    assert body["features"] == ["image"]


def test_admin_cert_forged_root_fails():
    real_root_priv, real_root_pub = generate_keypair()
    fake_root_priv, _ = generate_keypair()
    admin_priv, admin_pub = generate_keypair()
    # Admin cert signed by the WRONG root.
    cert = make_admin_cert(_priv(fake_root_priv), admin_pub, "Mallory")
    lic = issue_license(_priv(admin_priv), subject="X", machine="*", features=["tts"], days=10, admin_cert=cert)
    with pytest.raises(LicenseError):
        verify_license(lic, _pub(real_root_pub))


def test_tampered_body_fails():
    root_priv, root_pub = generate_keypair()
    lic = issue_license(_priv(root_priv), subject="Acme", machine="*", features=["tts"], days=30)
    lic["body"]["features"] = ["tts", "image", "shorts"]  # tamper after signing
    with pytest.raises(LicenseError):
        verify_license(lic, _pub(root_pub))


def test_expired_license_fails():
    root_priv, root_pub = generate_keypair()
    lic = issue_license(_priv(root_priv), subject="Acme", machine="*", features=["tts"], days=1)
    future = datetime.now(UTC) + timedelta(days=3)
    with pytest.raises(LicenseError):
        verify_license(lic, _pub(root_pub), now=future)


def test_machine_binding_enforced():
    root_priv, root_pub = generate_keypair()
    lic = issue_license(_priv(root_priv), subject="Acme", machine="abc123", features=["tts"], days=30)
    with pytest.raises(LicenseError):
        verify_license(lic, _pub(root_pub), machine="different-machine")
    # Correct machine passes.
    assert verify_license(lic, _pub(root_pub), machine="abc123")["subject"] == "Acme"


def test_cli_end_to_end(tmp_path):
    from scripts.licensing.license_tool import main

    keys = tmp_path / "keys"
    admin = keys / "admin"
    assert main(["keygen", "--role", "root", "--out", str(keys)]) == 0
    assert main(["keygen", "--role", "admin", "--out", str(admin)]) == 0
    cert = admin / "admin_cert.json"
    assert (
        main(
            [
                "admin-add",
                "--root-key",
                str(keys / "root_private.pem"),
                "--admin-pub",
                str(admin / "admin_public.pem"),
                "--name",
                "Alice",
                "--out",
                str(cert),
            ]
        )
        == 0
    )
    lic = tmp_path / "license.json"
    assert (
        main(
            [
                "issue",
                "--signer-key",
                str(admin / "admin_private.pem"),
                "--admin-cert",
                str(cert),
                "--subject",
                "Acme",
                "--machine",
                "*",
                "--features",
                "tts,image",
                "--days",
                "30",
                "--out",
                str(lic),
            ]
        )
        == 0
    )
    assert main(["verify", "--license", str(lic), "--root-pub", str(keys / "root_public.pem")]) == 0
