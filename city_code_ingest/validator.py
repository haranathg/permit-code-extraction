"""Validation utilities for the catalog and wizard outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set


def run_checks(
    *,
    wizard: Dict[str, object],
    catalog: Dict[str, List[Dict[str, object]]],
) -> Dict[str, object]:
    rad_ids = {item["id"] for item in catalog.get("RAD", [])}
    po_ids = {item["id"] for item in catalog.get("PO", [])}
    ead_ids = {item["id"] for item in catalog.get("EAD", [])}

    duplicate_ids = _find_duplicates(catalog)
    span_conflicts = _find_missing_spans(catalog)

    missing_links: list[str] = []
    dangling_refs: list[str] = []

    for decision_point in _iter_decision_points(wizard):
        rad_id = decision_point.get("rad_id")
        po_links = decision_point.get("po_links", [])
        no_po = decision_point.get("no_po_applicable", False)

        if not po_links and not no_po:
            missing_links.append(str(rad_id))

        for po_id in po_links:
            if po_id not in po_ids:
                dangling_refs.append(f"{rad_id}->{po_id}")

        for ead in decision_point.get("ead_links", []):
            if ead not in ead_ids:
                dangling_refs.append(f"{rad_id}->{ead}")

        if rad_id not in rad_ids:
            dangling_refs.append(f"wizard_rad_missing:{rad_id}")

    report = {
        "issues": {
            "missing_links": sorted(set(missing_links)),
            "dangling_refs": sorted(set(dangling_refs)),
            "span_conflicts": span_conflicts,
            "duplicate_ids": duplicate_ids,
        },
        "counts": {
            "rad": len(rad_ids),
            "po": len(po_ids),
            "ead": len(ead_ids),
            "decision_points": sum(1 for _ in _iter_decision_points(wizard)),
        },
    }

    report["status"] = "ok" if not any(report["issues"].values()) else "issues_detected"
    return report


def save_report(report: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def _find_duplicates(catalog: Dict[str, List[Dict[str, object]]]) -> List[str]:
    seen: Set[str] = set()
    duplicates: Set[str] = set()
    for items in catalog.values():
        for item in items:
            identifier = item.get("id")
            if identifier in seen:
                duplicates.add(identifier)
            else:
                seen.add(identifier)
    return sorted(duplicates)


def _find_missing_spans(catalog: Dict[str, List[Dict[str, object]]]) -> List[str]:
    conflicts: list[str] = []
    for item_type, items in catalog.items():
        for item in items:
            span = item.get("span")
            page = item.get("page")
            if span is None or not isinstance(span, list) or len(span) != 2:
                conflicts.append(f"{item_type}:{item.get('id')} missing span")
            if page is None:
                conflicts.append(f"{item_type}:{item.get('id')} missing page")
    return conflicts


def _iter_decision_points(wizard: Dict[str, object]):
    for title in wizard.get("titles", []):
        for chapter in title.get("chapters", []):
            for section in chapter.get("sections", []):
                for decision_point in section.get("decision_points", []):
                    yield decision_point


__all__ = ["run_checks", "save_report"]
