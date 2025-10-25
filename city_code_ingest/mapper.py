"""Link RAD items to related PO and EAD entries to form decision points."""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Dict, Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm missing
    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable


def link_items(
    catalog: Dict[str, List[Dict[str, object]]],
    layout: Dict[str, object],
    *,
    rad_po_table: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, object]]:
    """Create decision points linking RADs to POs and EADs."""

    rad_items = {item["id"]: item for item in catalog.get("RAD", [])}
    po_items = {item["id"]: item for item in catalog.get("PO", [])}
    ead_items = {item["id"]: item for item in catalog.get("EAD", [])}

    if rad_po_table is None:
        rad_po_table = _parse_correspondence_table(layout)

    decision_points: list[dict[str, object]] = []

    for rad_id, rad_item in tqdm(rad_items.items(), desc="Mapping RAD -> PO/EAD"):
        rad_text = rad_item.get("text", "")
        page = rad_item.get("page")

        linked_po_ids = list(rad_po_table.get(rad_id, []))
        if not linked_po_ids:
            linked_po_ids = _find_nearby_po(rad_item, po_items)

        if not linked_po_ids:
            linked_po_ids = _similar_po(rad_text, po_items)

        po_details = [
            {
                "po_id": po_id,
                "text": po_items[po_id]["text"],
                "page": po_items[po_id]["page"],
                "span": po_items[po_id]["span"],
            }
            for po_id in linked_po_ids
            if po_id in po_items
        ]

        ead_links, ead_details = _find_related_ead(rad_item, ead_items)

        decision_points.append(
            {
                "rad_id": rad_id,
                "rad_text": rad_text,
                "question": _format_question(rad_id, rad_text),
                "po_links": linked_po_ids,
                "po_details": po_details,
                "ead_links": ead_links,
                "ead_details": ead_details,
                "source_refs": [
                    {
                        "page": page,
                        "span": rad_item.get("span"),
                    }
                ],
                "no_po_applicable": not bool(linked_po_ids),
            }
        )

    decision_points.sort(key=lambda dp: dp["rad_id"])
    return decision_points


def _parse_correspondence_table(layout: Dict[str, object]) -> Dict[str, List[str]]:
    pages = layout.get("pages", [])
    mapping: dict[str, list[str]] = defaultdict(list)
    last_po: Optional[List[str]] = None
    in_table = False
    blank_rows = 0

    for page in pages:
        blocks = page.get("blocks", [])
        for block in blocks:
            text = (block.get("text") or "").strip()
            if not text:
                blank_rows += 1
                if in_table and blank_rows > 5:
                    in_table = False
                continue

            lowered = text.lower()
            if "corresponding po" in lowered:
                in_table = True
                blank_rows = 0
                continue

            if not in_table:
                continue

            blank_rows = 0
            if lowered.startswith("where accepted"):
                in_table = False
                continue

            if lowered.startswith("po"):
                last_po = _expand_po_tokens(text)
                continue

            if lowered.startswith("rad") and last_po is not None:
                rad_id = text.split()[0].replace(" ", "").upper()
                mapping[rad_id].extend([po for po in last_po if po])
                continue

    return {rad: list(dict.fromkeys(pos)) for rad, pos in mapping.items()}


def _expand_po_tokens(token_str: str) -> List[str]:
    token_str = token_str.replace("–", "-")
    tokens: list[str] = []
    for part in re.split(r",\s*", token_str):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            left_num = _extract_digits(left)
            right_num = _extract_digits(right)
            if left_num is not None and right_num is not None:
                prefix = "PO"
                for num in range(left_num, right_num + 1):
                    tokens.append(f"{prefix}{num}")
                continue
        number = _extract_digits(part)
        if number is not None:
            tokens.append(f"PO{number}")
    return tokens or [token_str.replace(" ", "").upper()]


def _extract_digits(value: str) -> Optional[int]:
    match = re.search(r"\d+", value)
    if not match:
        return None
    return int(match.group())


def _find_nearby_po(rad_item: Dict[str, object], po_items: Dict[str, Dict[str, object]]) -> List[str]:
    page = rad_item.get("page", 0)
    candidates = [
        po_id
        for po_id, po_item in po_items.items()
        if abs(po_item.get("page", 0) - page) <= 1
    ]
    return sorted(candidates)


def _similar_po(rad_text: str, po_items: Dict[str, Dict[str, object]]) -> List[str]:
    if not rad_text:
        return []

    rad_tokens = set(_tokenize(rad_text))
    if not rad_tokens:
        return []

    scored: list[tuple[str, float]] = []
    for po_id, po_item in po_items.items():
        po_tokens = set(_tokenize(po_item.get("text", "")))
        if not po_tokens:
            continue
        overlap = len(rad_tokens & po_tokens) / max(len(rad_tokens | po_tokens), 1)
        if overlap > 0.1:
            scored.append((po_id, overlap))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [po_id for po_id, _ in scored[:3]]


def _find_related_ead(rad_item: Dict[str, object], ead_items: Dict[str, Dict[str, object]]) -> tuple[List[str], List[Dict[str, object]]]:
    if not ead_items:
        return [], []

    page = rad_item.get("page", 0)
    rad_tokens = set(_tokenize(rad_item.get("text", "")))

    ead_links: list[str] = []
    ead_detail: list[dict[str, object]] = []
    for ead_id, ead_item in ead_items.items():
        if abs(ead_item.get("page", 0) - page) > 1:
            continue

        score = 0.0
        if rad_tokens:
            ead_tokens = set(_tokenize(ead_item.get("text", "")))
            if ead_tokens:
                score = len(rad_tokens & ead_tokens) / max(len(rad_tokens | ead_tokens), 1)

        if score >= 0.05:
            ead_links.append(ead_id)
            ead_detail.append(
                {
                    "ead_id": ead_id,
                    "text": ead_item.get("text", ""),
                    "page": ead_item.get("page"),
                    "span": ead_item.get("span"),
                }
            )

    return ead_links, ead_detail


def _format_question(rad_id: str, rad_text: str) -> str:
    rad_summary = rad_text.splitlines()[0].strip() if rad_text else ""
    if rad_summary:
        rad_summary = rad_summary.rstrip(":")
        return f"Does the development comply with {rad_id} ({rad_summary})?"
    return f"Does the development comply with {rad_id}?"


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9]+", text)]


__all__ = ["link_items", "_parse_correspondence_table"]
