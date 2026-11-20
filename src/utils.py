"""Decode JSON text and read its structure."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from typing import NoReturn

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type Reject = Callable[[str], NoReturn]


def reject_json_constant(constant: str) -> NoReturn:
    """Reject NaN and Infinity, which are not JSON numbers."""

    raise json.JSONDecodeError(f"invalid number {constant}", constant, 0)


def _as_json(value: object) -> JsonValue:
    if value is None or isinstance(value, str | int | float):
        return value
    if isinstance(value, list):
        return [_as_json(item) for item in value]
    if isinstance(value, dict):
        collected: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise json.JSONDecodeError("JSON object keys must be strings", str(key), 0)
            collected[key] = _as_json(item)
        return collected
    raise json.JSONDecodeError("JSON value is not a supported type", "", 0)


def decode_json(text: str) -> JsonValue:
    """Parse JSON text into a value whose shape has been checked."""

    loaded: object = json.loads(text, parse_constant=reject_json_constant)
    return _as_json(loaded)


class JsonReader:
    """Structural JSON checks that report failure through one callback."""

    def __init__(self, reject: Reject, *, reject_whitespace: bool = False) -> None:
        self._reject = reject
        self._reject_whitespace = reject_whitespace

    def expect_object(self, value: JsonValue, location: str) -> dict[str, JsonValue]:
        if not isinstance(value, dict):
            self._reject(f"{location} must be a JSON object.")
        collected: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                self._reject(f"{location} must use string keys.")
            collected[key] = item
        return collected

    def expect_exact_object(
        self,
        value: JsonValue,
        location: str,
        allowed: frozenset[str],
    ) -> dict[str, JsonValue]:
        fields = self.expect_object(value, location)
        extra = [key for key in fields if key not in allowed]
        if extra:
            self._reject(f"{location} has unsupported fields: {', '.join(extra)}.")
        return fields

    def expect_array(self, value: JsonValue, location: str) -> tuple[JsonValue, ...]:
        if not isinstance(value, list):
            self._reject(f"{location} must be a JSON array.")
        return tuple(value)

    def require_field(
        self,
        fields: Mapping[str, JsonValue],
        name: str,
        location: str,
    ) -> JsonValue:
        if name not in fields:
            self._reject(f"{location}.{name} is required.")
        return fields[name]

    def require_text(
        self,
        value: JsonValue,
        location: str,
        *,
        allow_empty: bool = False,
    ) -> str:
        if not isinstance(value, str):
            self._reject(f"{location} must be a string.")
        if self._reject_whitespace:
            if value.strip() == "":
                self._reject(f"{location} must be a non-empty string.")
            return value
        if not allow_empty and value == "":
            self._reject(f"{location} must be a non-empty string.")
        return value

    def require_text_field(
        self,
        fields: Mapping[str, JsonValue],
        name: str,
        location: str,
        *,
        allow_empty: bool = False,
    ) -> str:
        return self.require_text(
            self.require_field(fields, name, location),
            f"{location}.{name}",
            allow_empty=allow_empty,
        )

    def require_optional_text(self, value: JsonValue, location: str) -> str | None:
        if value is None:
            return None
        return self.require_text(value, location)

    def require_bool(self, value: JsonValue, location: str) -> bool:
        if isinstance(value, bool):
            return value
        self._reject(f"{location} must be a boolean.")

    def read_finite_number(
        self,
        value: JsonValue,
        location: str,
        reject: Reject | None = None,
    ) -> int | float:
        report = self._reject if reject is None else reject
        # bool is an int subclass, so it is excluded before the int check.
        if isinstance(value, bool):
            report(f"{location} must be a finite number.")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        report(f"{location} must be a finite number.")


__all__ = [
    "JsonReader",
    "JsonValue",
    "Reject",
    "decode_json",
    "reject_json_constant",
]
