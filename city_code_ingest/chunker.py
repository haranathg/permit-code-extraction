"""Split extracted text into hierarchical sections."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional


SECTION_PATTERN = re.compile(
    r"^(?:Section\s+)?(?P<id>\d{1,2}\.\d{1,2}\.\d{1,3})\s+[-:]?\s*(?P<title>.+)$",
    re.IGNORECASE,
)
TITLE_PATTERN = re.compile(r"^Title\s+(?P<num>\d+)\s*[-:\.]?\s*(?P<name>.*)$", re.IGNORECASE)
CHAPTER_PATTERN = re.compile(r"^Chapter\s+(?P<num>\d+)\s*[-:\.]?\s*(?P<name>.*)$", re.IGNORECASE)


@dataclass
class _SectionAccumulator:
    section_id: str
    heading: str
    title_number: Optional[int]
    title_name: Optional[str]
    chapter_number: Optional[int]
    chapter_name: Optional[str]
    breadcrumb: str
    body_lines: List[str]

    def to_dict(self) -> Dict[str, str]:
        return {
            "section_id": self.section_id,
            "heading": self.heading,
            "body": "\n".join(self.body_lines).strip(),
            "title_number": self.title_number,
            "title_name": self.title_name,
            "chapter_number": self.chapter_number,
            "chapter_name": self.chapter_name,
            "breadcrumb": self.breadcrumb,
        }


def split_sections(text_lines: List[str]) -> List[Dict[str, str]]:
    """Group lines into titled sections using heading regex heuristics."""

    print("[chunker] Splitting text into sections")
    sections: list[dict[str, str]] = []
    current: Optional[_SectionAccumulator] = None
    current_title: tuple[Optional[int], Optional[str]] = (None, None)
    current_chapter: tuple[Optional[int], Optional[str]] = (None, None)

    for line in text_lines:
        title_match = TITLE_PATTERN.match(line)
        if title_match:
            number = int(title_match.group("num"))
            name = title_match.group("name").strip() or None
            current_title = (number, name)
            print(f"[chunker] Found title {number}: {name or 'Unnamed'}")
            continue

        chapter_match = CHAPTER_PATTERN.match(line)
        if chapter_match:
            number = int(chapter_match.group("num"))
            name = chapter_match.group("name").strip() or None
            current_chapter = (number, name)
            print(f"[chunker] Found chapter {number}: {name or 'Unnamed'}")
            continue

        section_match = SECTION_PATTERN.match(line)
        if section_match:
            if current is not None:
                sections.append(current.to_dict())
            section_id = section_match.group("id")
            title_fragment = section_match.group("title").strip()
            heading = f"Section {section_id} {title_fragment}".strip()
            breadcrumb = _build_breadcrumb(
                title_number=current_title[0],
                title_name=current_title[1],
                chapter_number=current_chapter[0],
                chapter_name=current_chapter[1],
                section_id=section_id,
                section_title=title_fragment,
            )

            current = _SectionAccumulator(
                section_id=section_id,
                heading=heading,
                title_number=current_title[0],
                title_name=current_title[1],
                chapter_number=current_chapter[0],
                chapter_name=current_chapter[1],
                breadcrumb=breadcrumb,
                body_lines=[],
            )
            continue

        if current is not None:
            current.body_lines.append(line)

    if current is not None:
        sections.append(current.to_dict())

    print(f"[chunker] Created {len(sections)} sections")
    return sections


def _build_breadcrumb(
    *,
    title_number: Optional[int],
    title_name: Optional[str],
    chapter_number: Optional[int],
    chapter_name: Optional[str],
    section_id: str,
    section_title: str,
) -> str:
    parts: List[str] = []
    if title_number is not None:
        title_label = f"Title {title_number}"
        if title_name:
            title_label += f": {title_name}"
        parts.append(title_label)

    if chapter_number is not None:
        chapter_label = f"Chapter {chapter_number}"
        if chapter_name:
            chapter_label += f": {chapter_name}"
        parts.append(chapter_label)

    section_label = f"Section {section_id}"
    if section_title:
        section_label += f": {section_title}"
    parts.append(section_label)

    return " > ".join(parts)


__all__ = ["split_sections"]
