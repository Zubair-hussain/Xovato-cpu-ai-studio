"""Offline license & key management for securing Xovato models.

A small, self-contained Ed25519 trust chain:

    root keypair  ->  signs admin certificates  ->  admins sign licenses
                                                     licenses bind to a machine

Nothing here phones home; verification is fully offline. See
``docs/licensing.md`` for the full workflow.
"""

from scripts.licensing.license_tool import (
    LicenseError,
    issue_license,
    machine_fingerprint,
    make_admin_cert,
    verify_license,
)

__all__ = [
    "LicenseError",
    "issue_license",
    "machine_fingerprint",
    "make_admin_cert",
    "verify_license",
]
