"""Mock metadata enrichment stage."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


TOPIC_KEYWORDS = {
    "fire": "Fire Safety",
    "electrical": "Electrical",
    "plumbing": "Plumbing",
    "permit": "Permitting",
    "energy": "Energy Efficiency",
    "access": "Accessibility",
    "structural": "Structural",
}

DOCUMENT_KEYWORDS = {
    "application": "Application Form",
    "site plan": "Site Plan",
    "engineering": "Engineering Report",
    "inspection": "Inspection Report",
    "permit": "Permit Certificate",
}


def add_metadata(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach mock topics, references, and required documents."""

    print("[enricher] Adding metadata to sections")
    enriched: list[dict[str, object]] = []
    for section in sections:
        combined_text = " ".join([section.get("heading", ""), section.get("body", "")]).lower()

        topics = sorted({label for keyword, label in TOPIC_KEYWORDS.items() if keyword in combined_text})
        references = sorted({f"Section {match}" for match in re.findall(r"section\s+(\d{1,2}\.\d{1,2}\.\d{1,3})", combined_text)})
        required_docs = list({label for keyword, label in DOCUMENT_KEYWORDS.items() if keyword in combined_text})

        effective_date_match = re.search(r"effective\s+(?P<date>[a-z]+\s+\d{4})", combined_text)
        effective_date = effective_date_match.group("date").title() if effective_date_match else ""

        enriched_section = dict(section)
        decision_points = _extract_decision_points(
            section,
            references=references,
            breadcrumb=section.get("breadcrumb", ""),
        )

        enriched_section.update(
            {
                "topics": topics,
                "references": references,
                "requires_documents": required_docs,
                "effective_date": effective_date,
                "decision_points": decision_points,
            }
        )
        enriched.append(enriched_section)

    print(f"[enricher] Enriched {len(enriched)} sections")
    return enriched


__all__ = ["add_metadata"]


QUESTION_PATTERN = re.compile(r"^(?:ead\s+)?question[:\-]\s*(?P<body>.+)$", re.IGNORECASE)
RAD_PATTERN = re.compile(r"^RAD\d+\b", re.IGNORECASE)
PO_PATTERN = re.compile(r"^PO\d+", re.IGNORECASE)


def _extract_decision_points(
    section: Dict[str, Any],
    *,
    references: List[str],
    breadcrumb: str,
) -> List[Dict[str, Any]]:
    body_text = section.get("body", "")
    if not body_text:
        return []

    lines = [line.strip() for line in body_text.splitlines() if line.strip()]
    if not lines:
        return []

    rad_points = _extract_rad_po_decision_points(lines)
    if rad_points:
        return rad_points

    return _extract_simple_questions(lines, section, references, breadcrumb)


def _normalize_question(line: str) -> str:
    match = QUESTION_PATTERN.match(line)
    if match:
        question = match.group("body").strip()
        if question:
            return _ensure_question_suffix(question)

    if line.endswith("?"):
        return line

    # detect phrases like "Provide fire safety plan" -> treat as instruction, not question
    return ""


def _ensure_question_suffix(text: str) -> str:
    return text if text.endswith("?") else f"{text}?"


def _infer_response_options(line: str) -> List[str]:
    if re.search(r"\byes\b", line, re.IGNORECASE) or re.search(r"\bno\b", line, re.IGNORECASE):
        return ["Yes", "No"]
    return ["Yes", "No"]


def _extract_simple_questions(
    lines: List[str],
    section: Dict[str, Any],
    references: List[str],
    breadcrumb: str,
) -> List[Dict[str, Any]]:
    decision_points: list[dict[str, Any]] = []
    for line in lines:
        question_text = _normalize_question(line)
        if not question_text:
            continue

        reference_targets = references.copy()
        if not reference_targets and breadcrumb:
            reference_targets = [breadcrumb]

        decision_points.append(
            {
                "question_id": f"{section.get('section_id', 'unknown')}-q{len(decision_points) + 1}",
                "text": question_text,
                "response_options": _infer_response_options(line),
                "po_references": reference_targets,
                "guidance": section.get("heading", ""),
            }
        )

    return decision_points


def _extract_rad_po_decision_points(lines: List[str]) -> List[Dict[str, Any]]:
    rad_po_map = _build_rad_po_mapping(lines)
    if not rad_po_map:
        # Still attempt to extract RAD text even if table not found; there may still be RAD blocks
        rad_po_map = {}

    rad_texts, po_texts, rad_context = _collect_rad_po_texts(lines)
    if not rad_texts:
        return []

    decision_points: list[dict[str, Any]] = []
    for rad_id in sorted(rad_texts.keys(), key=_rad_sort_key):
        rad_body = rad_texts.get(rad_id, "")
        po_refs = rad_po_map.get(rad_id, [])
        question_text = _format_rad_question(rad_id, rad_body, rad_context.get(rad_id))
        decision_points.append(
            {
                "question_id": rad_id,
                "text": question_text,
                "response_options": ["Yes", "No"],
                "po_references": po_refs,
                "rad_text": rad_body,
                "po_details": [
                    {"po_id": po, "text": po_texts.get(po, "")} for po in po_refs
                ],
            }
        )

    return decision_points


def _build_rad_po_mapping(lines: List[str]) -> Dict[str, List[str]]:
    mapping: dict[str, list[str]] = {}
    last_po: Optional[List[str]] = None
    in_table = False
    blank_streak = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_table:
                blank_streak += 1
                if blank_streak > 4:
                    break
            continue

        lowered = stripped.lower()
        if "corresponding po" in lowered:
            in_table = True
            blank_streak = 0
            continue

        if not in_table:
            continue

        blank_streak = 0
        if lowered.startswith("effective") or lowered.startswith("moreton") or lowered.startswith("9 development"):
            continue

        if lowered.startswith("where accepted"):
            break

        if lowered.startswith("po"):
            last_po = _expand_po_tokens(stripped)
            continue

        if lowered.startswith("rad"):
            rad_id = stripped.split()[0].upper()
            if last_po is None:
                mapping.setdefault(rad_id, [])
            else:
                mapping[rad_id] = list(dict.fromkeys(last_po))
            continue

    return mapping


def _collect_rad_po_texts(lines: List[str]) -> tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    rad_texts: dict[str, str] = {}
    po_texts: dict[str, str] = {}
    rad_context: dict[str, str] = {}

    current_type: Optional[str] = None
    current_labels: List[str] = []
    buffer: List[str] = []
    last_nonempty: str = ""

    for line in lines:
        stripped = line.strip()
        if not stripped and not current_type:
            continue

        label_info = _match_label_line(stripped)
        if label_info:
            label_type, labels, remainder = label_info
            if current_type and current_labels:
                _flush_label_buffer(current_type, current_labels, buffer, rad_texts, po_texts)

            current_type = label_type
            current_labels = labels
            buffer = []
            if remainder:
                buffer.append(remainder.strip())

            if label_type == "RAD" and labels:
                context_line = last_nonempty.strip()
                if context_line:
                    rad_context[labels[0]] = context_line
            continue

        if stripped:
            last_nonempty = stripped
        if current_type:
            buffer.append(stripped)

    if current_type and current_labels:
        _flush_label_buffer(current_type, current_labels, buffer, rad_texts, po_texts)

    return rad_texts, po_texts, rad_context


def _flush_label_buffer(
    label_type: str,
    labels: List[str],
    buffer: List[str],
    rad_texts: Dict[str, str],
    po_texts: Dict[str, str],
) -> None:
    content = "\n".join(line for line in buffer if line).strip()
    if not content:
        return

    if label_type == "RAD":
        rad_texts[labels[0]] = content
    elif label_type == "PO":
        for label in labels:
            po_texts[label] = content


def _match_label_line(line: str) -> Optional[tuple[str, List[str], str]]:
    rad_match = re.match(r"^(RAD\d+)(?:[:.\-]\s*|\s+)?(.*)$", line, re.IGNORECASE)
    if rad_match:
        return "RAD", [rad_match.group(1).upper()], rad_match.group(2)

    po_match = re.match(r"^(PO[\d,\s\-–]+)(?:[:.\-]\s*|\s+)?(.*)$", line, re.IGNORECASE)
    if po_match:
        labels = _expand_po_tokens(po_match.group(1))
        remainder = po_match.group(2)
        return "PO", labels, remainder

    return None


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
    return tokens


def _extract_digits(value: str) -> Optional[int]:
    match = re.search(r"\d+", value)
    if not match:
        return None
    return int(match.group())


def _format_rad_question(rad_id: str, rad_body: str, context: Optional[str]) -> str:
    context_text = (context or rad_body.splitlines()[0] if rad_body else "").strip()
    if context_text:
        context_text = context_text.rstrip(":")
        question = f"Does the development comply with {rad_id} ({context_text})?"
    else:
        question = f"Does the development comply with {rad_id}?"
    return question


def _rad_sort_key(rad_id: str) -> int:
    match = re.search(r"\d+", rad_id)
    return int(match.group()) if match else 0
