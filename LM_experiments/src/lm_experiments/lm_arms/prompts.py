"""Load versioned prompt templates stored as standalone appendix-ready files."""

from __future__ import annotations

from pathlib import Path
from string import Template

TEMPLATES_DIR = Path(__file__).resolve().parents[3] / "prompts"
VALID_ARMS = {"b", "c", "d"}


def template_path(arm: str, version: str = "v1") -> Path:
    """Resolve a versioned arm template and reject unknown arms/versions."""
    arm = arm.lower()
    if arm not in VALID_ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(VALID_ARMS)}")
    if not version.replace("_", "").replace("-", "").isalnum():
        raise ValueError("prompt version must be alphanumeric")
    path = TEMPLATES_DIR / f"arm_{arm}_{version}.md"
    if not path.is_file():
        raise FileNotFoundError(f"prompt template not found: {path}")
    return path


def render(arm: str, version: str = "v1", **kwargs: object) -> str:
    """Strictly substitute all placeholders in one prompt template."""
    template = Template(template_path(arm, version).read_text(encoding="utf-8"))
    return template.substitute(**kwargs)
