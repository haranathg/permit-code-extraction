"""Utilities for reading city code source documents."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, List


@dataclass
class _HTMLTextExtractor(HTMLParser):
    """Lightweight HTML parser that collects visible text."""

    buffer: io.StringIO = field(default_factory=io.StringIO)

    def handle_data(self, data: str) -> None:  # pragma: no cover - simple passthrough
        self.buffer.write(data)

    def get_text(self) -> str:
        return self.buffer.getvalue()


def extract_text(file_path: str) -> List[str]:
    """Extract plain text lines from a PDF or HTML file.

    Falls back to a stub implementation if optional PDF libraries are missing.
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    print(f"[ingest] Extracting text from {path}")
    ext = path.suffix.lower()

    if ext == ".pdf":
        return _extract_pdf(path)
    if ext in {".html", ".htm"}:
        return _extract_html(path)

    print("[ingest] Unrecognized extension, treating as plain text")
    return _extract_plain_text(path)


def _extract_pdf(path: Path) -> List[str]:
    try:
        import fitz  # type: ignore

        lines: list[str] = []
        try:
            with fitz.open(path) as doc:
                for page in doc:
                    page_text = page.get_text("text") or ""
                    lines.extend(_normalize_lines(page_text.splitlines()))
        except Exception as exc:  # pragma: no cover - depends on pymupdf availability
            print(f"[ingest] PyMuPDF failed to read file ({exc}); using binary fallback")
            return _extract_plain_text(path)
        if not lines:
            print("[ingest] PyMuPDF returned no text; using binary fallback")
            return _extract_plain_text(path)
        return lines
    except ImportError:
        print("[ingest] PyMuPDF not available; attempting pdfplumber")

    try:
        import pdfplumber  # type: ignore

        lines: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                lines.extend(_normalize_lines(page_text.splitlines()))
        if not lines:
            print("[ingest] pdfplumber returned no text; using binary fallback")
            return _extract_plain_text(path)
        return lines
    except ImportError:
        print("[ingest] pdfplumber not available; evaluating fallback")

    with path.open("rb") as handle:
        header = handle.read(5)
        handle.seek(0)
        if header.startswith(b"%PDF"):
            raise RuntimeError(
                "PDF parsing requires PyMuPDF (pymupdf) or pdfplumber. Install one of them to continue."
            )
    print("[ingest] Treating PDF as plain text (non-binary content)")
    return _extract_plain_text(path)


def _extract_html(path: Path) -> List[str]:
    parser = _HTMLTextExtractor()
    with path.open("r", encoding="utf-8") as handle:
        html_content = handle.read()
    parser.feed(html_content)
    text = parser.get_text()
    return _normalize_lines(text.splitlines())


def _extract_plain_text(path: Path) -> List[str]:
    with path.open("rb") as handle:
        raw_bytes = handle.read()
    decoded = raw_bytes.decode("utf-8", errors="ignore")
    return _normalize_lines(decoded.splitlines())


def _normalize_lines(lines: Iterable[str]) -> List[str]:
    normalized: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            normalized.append(stripped)
    if not normalized:
        print("[ingest] Warning: no textual content found after normalization")
    return normalized


__all__ = ["extract_text"]
