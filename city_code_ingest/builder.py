"""Assemble enriched sections into the canonical JSON structure."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from jsonschema import Draft7Validator
except ImportError:  # pragma: no cover - optional dependency
    Draft7Validator = None  # type: ignore


SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "code_schema.json"


def build_json(
    sections: Iterable[Dict[str, Any]],
    *,
    city: str,
    state: str,
    version: str,
    source_url: str = "",
) -> Dict[str, Any]:
    """Build schema-compliant JSON and validate it."""

    print("[builder] Building hierarchical JSON document")
    titles: dict[int, Dict[str, Any]] = {}

    for section in sections:
        section_id = section.get("section_id")
        if not section_id:
            raise ValueError("Section missing required 'section_id'")

        title_number = _resolve_number(section.get("title_number"), section_id, index=0)
        chapter_number = _resolve_number(section.get("chapter_number"), section_id, index=1)

        title_entry = titles.setdefault(
            title_number,
            {
                "title_number": title_number,
                "title_name": section.get("title_name") or f"Title {title_number}",
                "chapters": {},
            },
        )

        chapters: dict[int, Dict[str, Any]] = title_entry["chapters"]
        chapter_entry = chapters.setdefault(
            chapter_number,
            {
                "chapter_number": chapter_number,
                "chapter_name": section.get("chapter_name") or f"Chapter {chapter_number}",
                "sections": [],
            },
        )

        chapter_entry["sections"].append(
            {
                "section_id": section_id,
                "section_title": section.get("heading", ""),
                "text": section.get("body", ""),
                "topics": section.get("topics", []),
                "references": section.get("references", []),
                "requires_documents": section.get("requires_documents", []),
                "effective_date": section.get("effective_date", ""),
                "breadcrumb": section.get("breadcrumb", ""),
                "decision_points": section.get("decision_points", []),
            }
        )

    titles_payload: list[dict[str, Any]] = []
    for title_number, title_entry in sorted(titles.items()):
        chapters = title_entry.pop("chapters")
        titles_payload.append(
            {
                "title_number": title_entry["title_number"],
                "title_name": title_entry.get("title_name", f"Title {title_entry['title_number']}") or "",
                "chapters": [
                    {
                        "chapter_number": chapter_data["chapter_number"],
                        "chapter_name": chapter_data.get(
                            "chapter_name", f"Chapter {chapter_data['chapter_number']}"
                        )
                        or "",
                        "sections": chapter_data["sections"],
                    }
                    for chapter_number, chapter_data in sorted(chapters.items())
                ],
            }
        )

    payload: Dict[str, Any] = {
        "jurisdiction": {
            "city": city,
            "state": state,
            "version": version,
            "last_updated": datetime.utcnow().date().isoformat(),
            "source_url": source_url,
        },
        "titles": titles_payload,
    }

    _validate_payload(payload)
    print("[builder] JSON document validated successfully")
    return payload


def _resolve_number(value: Optional[Any], section_id: str, *, index: int) -> int:
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass

    parts = section_id.split(".")
    if len(parts) > index:
        try:
            return int(parts[index])
        except ValueError:
            pass

    fallback = index + 1
    print(f"[builder] Falling back to inferred number {fallback} for section {section_id}")
    return fallback


def _validate_payload(payload: Dict[str, Any]) -> None:
    if Draft7Validator is None:
        print("[builder] jsonschema not available; skipping schema validation")
        return

    with SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)
    if errors:
        messages = ", ".join(f"{'/'.join(map(str, err.path))}: {err.message}" for err in errors)
        raise ValueError(f"Payload failed schema validation: {messages}")


__all__ = ["build_json"]
