"""Provider-neutral LM arm request and response contracts (stub only).

This module performs no network calls and imports no provider SDK. A future
adapter may send only ``request["prompt"]`` and ``request["tools"]`` to a
chosen API, then pass the returned payload through ``validate_prediction``.
The envelope's record_id is local scoring metadata and must not be exposed to
blind Arms B or D.
"""

from __future__ import annotations

from math import isfinite
from typing import Protocol

from lm_experiments.lm_arms.tools import SUBMIT_PREDICTION_SCHEMA, TOOL_SCHEMAS_D

ARMS = {"b", "c", "d"}
REQUEST_SCHEMA_VERSION = "v1"
PREDICTION_SCHEMA_VERSION = "v1"


def tools_for_arm(arm: str) -> list[dict]:
    """Return provider-neutral tool definitions for an arm."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")
    if arm == "d":
        return [*TOOL_SCHEMAS_D, SUBMIT_PREDICTION_SCHEMA]
    return [SUBMIT_PREDICTION_SCHEMA]


def build_request(record_id: str, arm: str, prompt_text: str) -> dict:
    """Build a local request envelope without performing an API call."""
    if not record_id or not isinstance(record_id, str):
        raise ValueError("record_id must be a non-empty string")
    if not prompt_text.strip():
        raise ValueError("prompt_text must not be empty")
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "record_id": record_id,
        "arm": arm,
        "prompt": prompt_text,
        "tools": tools_for_arm(arm),
    }


def validate_prediction(payload: dict, record_id: str | None = None) -> dict:
    """Validate and normalize one provider response to Prediction v1."""
    if not isinstance(payload, dict):
        raise TypeError("prediction payload must be a dict")
    required = ("delta_war", "delta_wrc", "delta_ops")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"prediction missing required fields: {missing}")

    prediction: dict = {"schema_version": PREDICTION_SCHEMA_VERSION}
    if record_id is not None:
        prediction["record_id"] = record_id
    for key in required:
        value = float(payload[key])
        if not isfinite(value):
            raise ValueError(f"{key} must be finite")
        prediction[key] = value

    confidence = payload.get("confidence")
    if confidence is not None:
        confidence = float(confidence)
        if not isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be null or in [0, 1]")
    prediction["confidence"] = confidence

    comparables = payload.get("comparables")
    if comparables is not None:
        if not isinstance(comparables, list) or not all(
            isinstance(item, str) for item in comparables
        ):
            raise ValueError("comparables must be null or list[str]")
        if len(comparables) > 10:
            raise ValueError("comparables must contain at most 10 IDs")
    prediction["comparables"] = comparables

    rationale = payload.get("rationale")
    if rationale is not None and not isinstance(rationale, str):
        raise ValueError("rationale must be null or a string")
    prediction["rationale"] = rationale
    return prediction


class LMProviderAdapter(Protocol):
    """Interface a future provider-specific API adapter must implement."""

    def predict(self, request: dict) -> dict:
        """Return a raw submit_prediction payload for one request envelope."""
        ...


def call_lm_arm(*args: object, **kwargs: object) -> dict:
    """Deliberate Phase 5 stub: external API execution is out of scope."""
    raise NotImplementedError(
        "Phase 5 is stub-only. Implement an LMProviderAdapter in a later phase; "
        "see API_INTEGRATION.md."
    )
