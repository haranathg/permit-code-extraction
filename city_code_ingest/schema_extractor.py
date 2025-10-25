"""Catalog RAD, PO, and EAD items from layout-aware document data."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from dotenv import load_dotenv

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm missing
    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable


load_dotenv()


CATALOG_ITEM_RE = re.compile(r"\b((?:RAD|PO|EAD)\s*\d+[\.\d]*)\b", re.IGNORECASE)


@dataclass
class CatalogItem:
    id: str
    text: str
    page: int
    span: List[int]
    type: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "text": self.text.strip(),
            "page": self.page,
            "span": self.span,
            "type": self.type,
        }


def catalog_items(layout: Dict[str, object], *, use_llm: bool = False) -> Dict[str, List[Dict[str, object]]]:
    """Produce a catalog of RAD/PO/EAD items from layout data."""

    pages = layout.get("pages", [])
    if use_llm:
        items = extract_via_llm(pages)
    else:
        items = _extract_via_regex(pages)

    catalog_by_type: Dict[str, List[Dict[str, object]]] = {"RAD": [], "PO": [], "EAD": []}
    for item in items:
        catalog_by_type.setdefault(item["type"], []).append(item)

    # Sort by page and then id for deterministic output
    for values in catalog_by_type.values():
        values.sort(key=lambda entry: (entry["page"], entry["id"]))

    return catalog_by_type


def save_catalog(catalog: Dict[str, List[Dict[str, object]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(catalog, handle, indent=2)


def extract_via_llm(pages: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Placeholder for Bedrock/Nova structured extraction API."""
    try:  # pragma: no cover - network dependency
        import openai

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        client = openai.OpenAI(api_key=api_key)

        # Simplified prompt; real implementation will leverage structured output.
        page_text = "\n\n".join(
            "\n".join(block.get("text", "") for block in page.get("blocks", [])) for page in pages
        )
        prompt = (
            "Extract all RAD, PO, and EAD identifiers with their surrounding sentences. "
            "Respond as JSON array of objects {id, text, type, page, span_start, span_end}.\n\n"
            + page_text[:6000]
        )

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=0,
        )

        content = response.output_text
        parsed = json.loads(content)
        items: list[Dict[str, object]] = []
        for entry in parsed:
            identifier = entry.get("id", "").upper().replace(" ", "")
            if not identifier:
                continue
            item_type = entry.get("type", "RAD").upper()
            if item_type not in {"RAD", "PO", "EAD"}:
                continue
            items.append(
                CatalogItem(
                    id=identifier,
                    text=entry.get("text", ""),
                    page=int(entry.get("page", 1)),
                    span=[int(entry.get("span_start", 0)), int(entry.get("span_end", 0))],
                    type=item_type,
                ).to_dict()
            )
        if items:
            return items
    except Exception as exc:  # pragma: no cover
        print(f"[schema_extractor] LLM extraction failed ({exc}); falling back to regex")

    # TODO: Replace with Bedrock Nova structured extraction API.
    return _extract_via_regex(pages)


def _extract_via_regex(pages: List[Dict[str, object]]) -> List[Dict[str, object]]:
    items: list[CatalogItem] = []
    for page in tqdm(pages, desc="Cataloging items"):
        page_num = int(page.get("page_number", 1))
        blocks = page.get("blocks", [])
        char_cursor = 0
        for block in blocks:
            text = (block.get("text") or "").strip()
            if not text:
                char_cursor += 1
                continue

            matches = list(CATALOG_ITEM_RE.finditer(text))
            if not matches:
                char_cursor += len(text) + 1
                continue

            for match in matches:
                identifier = match.group(1).replace(" ", "").upper()
                item_type = "RAD" if identifier.startswith("RAD") else "PO" if identifier.startswith("PO") else "EAD"
                span_start = char_cursor + match.start()
                span_end = char_cursor + match.end()

                # Attempt to gather context lines around the identifier
                context = text
                items.append(
                    CatalogItem(
                        id=identifier,
                        text=context,
                        page=page_num,
                        span=[span_start, span_end],
                        type=item_type,
                    )
                )

            char_cursor += len(text) + 1

    # Deduplicate by keeping the first occurrence per id/type
    seen: dict[tuple[str, str], CatalogItem] = {}
    for item in items:
        key = (item.id, item.type)
        if key not in seen:
            seen[key] = item

    return [item.to_dict() for item in seen.values()]


__all__ = ["catalog_items", "save_catalog", "extract_via_llm"]
