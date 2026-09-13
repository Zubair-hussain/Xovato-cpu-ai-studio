# Model Licensing (optional)

If you want to **secure your models** so only authorised machines can run them,
Xovato ships a small offline licensing tool built on Ed25519 signatures. It never
phones home - verification is fully local.

> Optional. The backend runs without it. Enable it only if you sell or restrict
> access to a build. Install with `pip install -r requirements-security.txt`.

## The trust chain

```
ROOT keypair (you, the owner)
   │  signs an admin's public key  ->  admin certificate
   ▼
ADMIN keypair (a delegate who may issue licenses)
   │  signs a license bound to a machine
   ▼
LICENSE  ->  verified offline against the ROOT public key
```

- **Root** is the single trust anchor. Ship `root_public.pem` with your product;
  keep `root_private.pem` offline and secret.
- **Admins** are optional delegates. Root signs an admin's public key once; that
  admin can then issue licenses without ever touching the root private key.
- **Licenses** are bound to a machine fingerprint (hostname + hardware id), carry
  a feature list and an expiry, and embed the admin certificate so a verifier can
  walk `license → admin → root`.

## Quick start

```bash
pip install -r requirements-security.txt

# 1. Owner: create the root keypair (do this once, keep the private key safe)
python -m scripts.licensing.license_tool keygen --role root --out keys

# 2. (optional) Create a delegated admin and have root vouch for it
python -m scripts.licensing.license_tool keygen --role admin --out keys/admin
python -m scripts.licensing.license_tool admin-add \
  --root-key keys/root_private.pem \
  --admin-pub keys/admin/admin_public.pem \
  --name "Alice" --out keys/admin/admin_cert.json

# 3. On the target machine: get its fingerprint
python -m scripts.licensing.license_tool fingerprint

# 4. Issue a license (admin signs; use root_private + no --admin-cert to sign as root)
python -m scripts.licensing.license_tool issue \
  --signer-key keys/admin/admin_private.pem \
  --admin-cert keys/admin/admin_cert.json \
  --subject "ACME Studio" --machine <fingerprint from step 3> \
  --features tts,image,audio,shorts --days 365 --out license.json

# 5. Anywhere: verify offline against the root public key
python -m scripts.licensing.license_tool verify \
  --license license.json --root-pub keys/root_public.pem
```

`--machine "*"` issues a license valid on any machine (useful for internal
builds). Omit `--admin-cert` and pass `root_private.pem` as `--signer-key` to have
root issue directly.

## Enforce it in the app

Verify at startup and refuse to serve gated features when the license is invalid:

```python
from pathlib import Path
import json
from cryptography.hazmat.primitives import serialization
from scripts.licensing.license_tool import verify_license, LicenseError

def load_features() -> list[str]:
    root_pub = serialization.load_pem_public_key(Path("root_public.pem").read_bytes())
    try:
        body = verify_license(json.loads(Path("license.json").read_text()), root_pub)
    except (FileNotFoundError, LicenseError):
        return []            # unlicensed: expose only free features
    return body["features"]  # e.g. ["tts", "image"]
```

## Keys, admin, and CI

- **Never commit private keys.** `.gitignore` already excludes `keys/` and
  `*.pem`. Commit only `root_public.pem` if you want CI to verify licenses.
- The **admin role** is for a small team: root stays offline, admins issue
  day-to-day licenses. Revoke an admin by rotating the root key and re-issuing
  the certs you still trust.
- A GitHub workflow, [`.github/workflows/license-check.yml`](../.github/workflows/license-check.yml),
  runs the licensing tests and (when a `license.json` + `root_public.pem` are
  present) verifies the committed license on every push.

## Copyright

The licensing tool stamps every issued license with
`(c) Zubair Hussain - Xovato AI Studio`. Removing that line invalidates the
signature, so it cannot be silently stripped.
