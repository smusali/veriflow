"""Load model settings from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
_TEMPERATURE = 0.0


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: str
    model: str
    temperature: float


def _setting(name: str, provided: str | None) -> str:
    if provided is not None:
        return provided.strip()
    return os.environ.get(name, "").strip()


def load_settings(api_key: str | None = None, model: str | None = None) -> Settings:
    """Read the API key and model, loading .env when a value is omitted."""

    if api_key is None or model is None:
        load_dotenv(_ENV_FILE)
    return Settings(
        api_key=_setting("OPENAI_API_KEY", api_key),
        model=_setting("OPENAI_MODEL", model),
        temperature=_TEMPERATURE,
    )


__all__ = ["Settings", "load_settings"]
