"""Activate the captured 2026-08-20 image-matched ATOM/AITER main stack."""

import official_clean


IMAGE = "rocm/atom-dev@sha256:a5cfa1ab503af6e0f55e0ed83cd7e999edbea6940a38dba44aaeee9a22758976"
IMAGE_ID = "sha256:a5cfa1ab503af6e0f55e0ed83cd7e999edbea6940a38dba44aaeee9a22758976"
ATOM_COMMIT = "1e7659fde32eeaa0d9aa868c3e90847e5e46a51c"
AITER_COMMIT = "eb84cb02200b1707f1076edf3f4930d3626adfb2"


def activate() -> None:
    official_clean.IMAGE = IMAGE
    official_clean.IMAGE_ID = IMAGE_ID
    official_clean.ATOM_COMMIT = ATOM_COMMIT
    official_clean.AITER_COMMIT = AITER_COMMIT
