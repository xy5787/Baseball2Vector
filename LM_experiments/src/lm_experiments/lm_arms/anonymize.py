"""Deterministic anonymization for Arm B (and Arm D's own-subject masking)."""

from __future__ import annotations

import hashlib


def anon_id(player_id: str, salt: str) -> str:
    """Map a real player id to a stable, non-reversible-by-inspection anon id.

    Deterministic (same input + salt -> same output) so repeated queries about
    the same player stay linkable within a run without revealing identity.
    Not a security mechanism — the salt is a fixed reproducibility constant in
    config.yaml, not a secret.

    Args:
        player_id: Real player identifier (Name).
        salt: Fixed salt from config (lm_arms.anon_salt).

    Returns:
        "player_XXXXXXXX" — an 8-hex-char id.
    """
    digest = hashlib.sha256(f"{salt}:{player_id}".encode()).hexdigest()[:8]
    return f"player_{digest}"
