"""Public JSON helper contracts."""

from __future__ import annotations

import json
import math
import re
from enum import StrEnum
from typing import NoReturn

import pytest

from src.utils import (
    JsonReader,
    JsonValue,
    Reject,
    decode_document,
    decode_json,
    enum_member,
    json_object,
)


class Rejected(Exception):
    """Failure reported through a test reject callback."""


class Color(StrEnum):
    RED = "red"


def reject(message: str) -> NoReturn:
    raise Rejected(message)


def reader(*, reject_whitespace: bool = False) -> JsonReader:
    return JsonReader(reject, reject_whitespace=reject_whitespace)


def assert_same_json(rebuilt: object, expected: JsonValue) -> None:
    if isinstance(expected, list):
        assert isinstance(rebuilt, list)
        assert type(rebuilt) is list
        assert len(rebuilt) == len(expected)
        for item, wanted in zip(rebuilt, expected, strict=True):
            assert_same_json(item, wanted)
        return
    if isinstance(expected, dict):
        assert isinstance(rebuilt, dict)
        assert type(rebuilt) is dict
        assert list(rebuilt) == list(expected)
        for key, wanted in expected.items():
            assert_same_json(rebuilt[key], wanted)
        return
    assert rebuilt == expected
    if isinstance(expected, bool):
        assert type(rebuilt) is bool
    elif isinstance(expected, int):
        assert type(rebuilt) is int
    elif isinstance(expected, float):
        assert type(rebuilt) is float


def test_decode_json_accepts_scalars_and_nested_values() -> None:
    assert decode_json("null") is None
    assert decode_json('"ok"') == "ok"
    assert decode_json("1") == 1
    assert type(decode_json("1")) is int
    assert decode_json("1.5") == 1.5
    loaded = decode_json('[true, false, {"a": [null]}]')
    assert isinstance(loaded, list)
    assert type(loaded[0]) is bool
    assert type(loaded[1]) is bool
    assert loaded == [True, False, {"a": [None]}]


@pytest.mark.parametrize("text", ["NaN", "Infinity", "-Infinity"])
def test_decode_json_rejects_non_json_numbers(text: str) -> None:
    with pytest.raises(json.JSONDecodeError, match="invalid number"):
        decode_json(text)


def test_decode_json_rejects_syntax() -> None:
    with pytest.raises(json.JSONDecodeError):
        decode_json("{")


def test_decode_document_returns_value() -> None:
    loaded = decode_document('{"ok": true}', label="Evidence", reject=reject)
    assert loaded == {"ok": True}
    assert isinstance(loaded, dict)
    assert type(loaded["ok"]) is bool


def test_decode_document_rejects_invalid_text() -> None:
    with pytest.raises(Rejected, match=r"Evidence is not valid JSON: .+\."):
        decode_document("{", label="Evidence", reject=reject)


def test_json_object_rebuilds_nested_values() -> None:
    source: JsonValue = {"flag": True, "items": [None, 1, "a", {"ok": False}]}
    assert_same_json(json_object(source), source)


def test_enum_member_returns_named_value() -> None:
    assert (
        enum_member(Color, "red", "field", reject, supported="Supported values are red.")
        is Color.RED
    )


def test_enum_member_rejects_unknown_name() -> None:
    supported = "Supported values are red."
    with pytest.raises(Rejected, match=re.escape(f"field 'blue' is unsupported. {supported}")):
        enum_member(Color, "blue", "field", reject, supported=supported)


def test_reader_accepts_objects_and_arrays() -> None:
    fields = reader().expect_object({"a": 1}, "place")
    assert fields == {"a": 1}
    assert reader().expect_array([1, "a"], "place") == (1, "a")


def test_reader_rejects_wrong_containers() -> None:
    with pytest.raises(Rejected, match=re.escape("place must be a JSON object.")):
        reader().expect_object([], "place")
    with pytest.raises(Rejected, match=re.escape("place must be a JSON array.")):
        reader().expect_array({}, "place")


def test_reader_names_unsupported_fields() -> None:
    with pytest.raises(Rejected, match=re.escape("place has unsupported fields: extra, other.")):
        reader().expect_exact_object(
            {"a": 1, "extra": True, "other": False}, "place", frozenset({"a"})
        )


def test_reader_requires_fields() -> None:
    fields: dict[str, JsonValue] = {"name": "Ada"}
    assert reader().require_field(fields, "name", "place") == "Ada"
    with pytest.raises(Rejected, match=re.escape("place.missing is required.")):
        reader().require_field(fields, "missing", "place")


def test_reader_requires_text() -> None:
    checked = reader()
    assert checked.require_text("  hello  ", "place") == "  hello  "
    assert checked.require_text("", "place", allow_empty=True) == ""
    with pytest.raises(Rejected, match=re.escape("place must be a string.")):
        checked.require_text(1, "place")
    with pytest.raises(Rejected, match=re.escape("place must be a non-empty string.")):
        checked.require_text("", "place")


def test_reader_rejects_blank_text_when_requested() -> None:
    checked = reader(reject_whitespace=True)
    assert checked.require_text("  hello  ", "place") == "  hello  "
    for blank in ("", "  "):
        with pytest.raises(Rejected, match=re.escape("place must be a non-empty string.")):
            checked.require_text(blank, "place")


def test_reader_requires_text_fields() -> None:
    fields: dict[str, JsonValue] = {"name": "  Ada  "}
    assert reader().require_text_field(fields, "name", "place") == "  Ada  "
    with pytest.raises(Rejected, match=re.escape("place.name is required.")):
        reader().require_text_field({}, "name", "place")
    with pytest.raises(Rejected, match=re.escape("place.name must be a string.")):
        reader().require_text_field({"name": 1}, "name", "place")


def test_reader_optional_text_accepts_null_only() -> None:
    assert reader().require_optional_text(None, "place") is None
    with pytest.raises(Rejected, match=re.escape("place must be a non-empty string.")):
        reader().require_optional_text("", "place")


def test_reader_requires_bools() -> None:
    checked = reader()
    assert checked.require_bool(True, "place") is True
    assert checked.require_bool(False, "place") is False
    for value in (0, 1):
        with pytest.raises(Rejected, match=re.escape("place must be a boolean.")):
            checked.require_bool(value, "place")


def test_reader_requires_ints_and_excludes_bools() -> None:
    assert reader().require_int(2, "place", label="offset") == 2
    with pytest.raises(Rejected, match=re.escape("place must be an integer offset.")):
        reader().require_int(True, "place", label="offset")
    with pytest.raises(Rejected, match=re.escape("place must be an integer offset.")):
        reader().require_int(1.5, "place", label="offset")


def test_reader_accepts_finite_numbers() -> None:
    checked = reader()
    assert checked.read_finite_number(0, "n") == 0
    assert type(checked.read_finite_number(1, "n")) is int
    assert checked.read_finite_number(1.5, "n") == 1.5


@pytest.mark.parametrize("value", [True, False, math.nan, math.inf, -math.inf, "1"])
def test_reader_rejects_non_finite_numbers(value: JsonValue) -> None:
    with pytest.raises(Rejected, match=re.escape("n must be a finite number.")):
        reader().read_finite_number(value, "n")


def test_reader_uses_the_supplied_number_reject() -> None:
    def custom(message: str) -> NoReturn:
        raise Rejected(f"custom {message}")

    supplied: Reject = custom
    with pytest.raises(Rejected, match=re.escape("custom n must be a finite number.")):
        reader().read_finite_number(True, "n", supplied)
