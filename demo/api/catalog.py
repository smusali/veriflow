"""Read the case catalog and the fixture files it names."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from src.utils import JsonReader, JsonValue, decode_json


def _reject(message: str) -> NoReturn:
    raise ValueError(message)


_reader = JsonReader(_reject)


@dataclass(frozen=True, slots=True)
class CatalogCase:
    id: str
    title: str
    category: str
    notes: str | None
    graph_path: Path | None
    transcript_path: Path | None


def resolve_fixture(root: Path, relative: str | None) -> Path | None:
    """Return a fixture file inside the fixtures directory, or none."""

    if relative is None:
        return None
    base = root.resolve()
    candidate = (base / relative).resolve()
    if not candidate.is_relative_to(base) or not candidate.is_file():
        return None
    return candidate


def _notes(fields: dict[str, JsonValue], location: str) -> str | None:
    if "notes" not in fields:
        return None
    return _reader.require_optional_text(fields["notes"], f"{location}.notes")


def _case(value: JsonValue, index: int, root: Path) -> CatalogCase:
    location = f"cases[{index}]"
    fields = _reader.expect_object(value, location)
    workflow = _reader.require_optional_text(
        _reader.require_field(fields, "workflow_fixture", location),
        f"{location}.workflow_fixture",
    )
    transcript = _reader.require_optional_text(
        _reader.require_field(fields, "transcript_fixture", location),
        f"{location}.transcript_fixture",
    )
    return CatalogCase(
        id=_reader.require_text_field(fields, "id", location),
        title=_reader.require_text_field(fields, "title", location),
        category=_reader.require_text_field(fields, "category", location),
        notes=_notes(fields, location),
        graph_path=resolve_fixture(root, workflow),
        transcript_path=resolve_fixture(root, transcript),
    )


def load_catalog(path: Path, root: Path) -> tuple[CatalogCase, ...]:
    """Load every catalog case in file order."""

    document = decode_json(path.read_text(encoding="utf-8"))
    fields = _reader.expect_object(document, "catalog")
    items = _reader.expect_array(_reader.require_field(fields, "cases", "catalog"), "catalog.cases")
    cases = tuple(_case(item, index, root) for index, item in enumerate(items))
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            _reject(f"Duplicate case id {case.id}.")
        seen.add(case.id)
    return cases


def find_case(cases: tuple[CatalogCase, ...], case_id: str) -> CatalogCase | None:
    """Return the catalog case with this id."""

    for case in cases:
        if case.id == case_id:
            return case
    return None


def read_fixture_text(path: Path | None) -> str | None:
    """Read a fixture file, or return none when the case has no file."""

    if path is None:
        return None
    return path.read_text(encoding="utf-8")


__all__ = [
    "CatalogCase",
    "find_case",
    "load_catalog",
    "read_fixture_text",
    "resolve_fixture",
]
