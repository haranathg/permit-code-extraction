"""Assemble wizard and guidance payloads from decision point data."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:  # pragma: no cover - optional dependency
    from jsonschema import Draft7Validator
except ImportError:  # pragma: no cover - keep running without validation
    Draft7Validator = None  # type: ignore


WIZARD_SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "wizard_schema.json"


def build_outputs(
    decision_points: List[Dict[str, Any]],
    *,
    sections: Iterable[Dict[str, Any]],
    catalog: Dict[str, List[Dict[str, Any]]],
    city: str,
    state: str,
    version: str,
    source_url: str = "",
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Return wizard and guidance payloads."""

    sections_list = list(sections)
    if not sections_list:
        raise ValueError("No sections available to build wizard output")

    section_assignments = _assign_decision_points_to_sections(decision_points, sections_list)

    wizard_payload = _build_wizard(section_assignments, city, state, version, source_url)
    _validate_wizard(wizard_payload)

    guidance_payload = _build_guidance(section_assignments, catalog, wizard_payload["jurisdiction"])

    return wizard_payload, guidance_payload


def _assign_decision_points_to_sections(
    decision_points: List[Dict[str, Any]],
    sections_list: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    section_wrappers: list[dict[str, Any]] = []
    for idx, section in enumerate(sections_list):
        section_wrappers.append(
            {
                "meta": section,
                "decision_points": [],
            }
        )

    # Simple heuristic: assign by round-robin order of appearance
    # Future enhancement could use page numbers.
    current_index = 0
    for dp in decision_points:
        section_wrappers[current_index]["decision_points"].append(dp)
        if len(section_wrappers) > 1:
            current_index = (current_index + 1) % len(section_wrappers)

    return section_wrappers


def _build_wizard(
    section_assignments: List[Dict[str, Any]],
    city: str,
    state: str,
    version: str,
    source_url: str,
) -> Dict[str, Any]:
    titles: dict[int, Dict[str, Any]] = {}

    for wrapper in section_assignments:
        section_meta = wrapper["meta"]
        decision_points = wrapper["decision_points"]

        section_id = section_meta.get("section_id") or "section-1"
        title_number = section_meta.get("title_number") or 1
        chapter_number = section_meta.get("chapter_number") or 1

        title_entry = titles.setdefault(
            title_number,
            {
                "title_number": int(title_number),
                "title_name": section_meta.get("title_name") or f"Title {title_number}",
                "chapters": {},
            },
        )

        chapters = title_entry["chapters"]
        chapter_entry = chapters.setdefault(
            chapter_number,
            {
                "chapter_number": int(chapter_number),
                "chapter_name": section_meta.get("chapter_name") or f"Chapter {chapter_number}",
                "sections": [],
            },
        )

        breadcrumbs = _build_breadcrumbs(section_meta)
        topics = section_meta.get("topics", []) or []
        section_entry = {
            "section_id": section_id,
            "section_title": section_meta.get("heading") or section_id,
            "breadcrumbs": breadcrumbs,
            "topics": topics,
            "decision_points": decision_points,
        }

        chapter_entry["sections"].append(section_entry)

    titles_payload: list[dict[str, Any]] = []
    for title_number, title_entry in sorted(titles.items(), key=lambda item: item[0]):
        chapters_payload = []
        for chapter_number, chapter_entry in sorted(title_entry["chapters"].items(), key=lambda item: item[0]):
            sections_payload = chapter_entry["sections"]
            chapters_payload.append(
                {
                    "chapter_number": chapter_entry["chapter_number"],
                    "chapter_name": chapter_entry["chapter_name"],
                    "sections": sections_payload,
                }
            )

        titles_payload.append(
            {
                "title_number": title_entry["title_number"],
                "title_name": title_entry["title_name"],
                "chapters": chapters_payload,
            }
        )

    return {
        "jurisdiction": {
            "city": city,
            "state": state,
            "version": version,
            "last_updated": datetime.utcnow().date().isoformat(),
            "source_url": source_url,
        },
        "titles": titles_payload,
    }


def _build_guidance(
    section_assignments: List[Dict[str, Any]],
    catalog: Dict[str, List[Dict[str, Any]]],
    jurisdiction: Dict[str, Any],
) -> Dict[str, Any]:
    guidance_entries: list[dict[str, Any]] = []

    for wrapper in section_assignments:
        section_meta = wrapper["meta"]
        section_id = section_meta.get("section_id") or "section-1"
        entry_base = {
            "section_id": section_id,
            "section_title": section_meta.get("heading") or section_id,
            "title_number": section_meta.get("title_number") or 1,
            "title_name": section_meta.get("title_name") or "",
            "chapter_number": section_meta.get("chapter_number") or 1,
            "chapter_name": section_meta.get("chapter_name") or "",
            "breadcrumb": section_meta.get("breadcrumb") or "",
        }

        section_body = section_meta.get("body", "").strip()
        if section_body:
            guidance_entries.append({
                **entry_base,
                "entry_type": "section",
                "guidance": section_body,
            })

        for dp in wrapper["decision_points"]:
            guidance_entries.append(
                {
                    **entry_base,
                    "entry_type": "decision_point",
                    "question_id": dp.get("rad_id"),
                    "guidance": dp.get("rad_text"),
                    "po_details": dp.get("po_details", []),
                    "ead_details": dp.get("ead_details", []),
                    "source_refs": dp.get("source_refs", []),
                }
            )

    return {
        "jurisdiction": jurisdiction,
        "guidance": guidance_entries,
        "catalog_summary": {key: len(value) for key, value in catalog.items()},
    }


def _build_breadcrumbs(section_meta: Dict[str, Any]) -> List[str]:
    breadcrumbs: list[str] = []
    title_number = section_meta.get("title_number")
    title_name = section_meta.get("title_name")
    chapter_number = section_meta.get("chapter_number")
    chapter_name = section_meta.get("chapter_name")

    if title_number is not None:
        label = f"Title {title_number}"
        if title_name:
            label += f": {title_name}"
        breadcrumbs.append(label)

    if chapter_number is not None:
        label = f"Chapter {chapter_number}"
        if chapter_name:
            label += f": {chapter_name}"
        breadcrumbs.append(label)

    section_heading = section_meta.get("heading")
    if section_heading:
        breadcrumbs.append(section_heading)

    return breadcrumbs or [section_meta.get("breadcrumb", "")] if section_meta.get("breadcrumb") else []


def _validate_wizard(payload: Dict[str, Any]) -> None:
    if Draft7Validator is None:
        print("[builder] jsonschema not available; skipping wizard validation")
        return

    with WIZARD_SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)
    if errors:
        messages = ", ".join(f"{'/'.join(map(str, err.path))}: {err.message}" for err in errors)
        raise ValueError(f"Wizard payload failed schema validation: {messages}")


__all__ = ["build_outputs"]
