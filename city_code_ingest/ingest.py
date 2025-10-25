"""Utilities for reading city code source documents."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Optional


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


def extract_document(file_path: str) -> Dict[str, object]:
    """Return both normalized lines and layout information for a document."""

    lines = extract_text(file_path)
    layout = extract_layout(file_path)
    return {
        "lines": lines,
        "layout": layout,
    }


def extract_layout(file_path: str) -> Dict[str, object]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    ext = path.suffix.lower()
    if ext == ".pdf":
        layout = _extract_pdf_layout(path)
        if layout:
            return layout

    # Treat everything else as plain text layout fallback
    lines = extract_text(file_path)
    blocks = []
    for idx, line in enumerate(lines):
        blocks.append(
            {
                "id": f"block-{idx}",
                "text": line,
                "page": 1,
                "span": [idx, idx + 1],
                "bbox": [0.0, float(idx), 0.0, float(idx + 1)],
            }
        )
    return {
        "pages": [
            {
                "page_number": 1,
                "blocks": blocks,
            }
        ]
    }


def _extract_pdf_layout(path: Path) -> Optional[Dict[str, object]]:
    try:
        import pdfplumber  # type: ignore

        pages: list[dict[str, object]] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                blocks: list[dict[str, object]] = []
                text_chunks = page.extract_words(use_text_flow=True, keep_blank_chars=False)
                if text_chunks:
                    for idx, chunk in enumerate(text_chunks):
                        chunk_text = chunk.get("text", "").strip()
                        if not chunk_text:
                            continue
                        bbox = [
                            float(chunk.get("x0", 0.0)),
                            float(chunk.get("top", 0.0)),
                            float(chunk.get("x1", 0.0)),
                            float(chunk.get("bottom", 0.0)),
                        ]
                        blocks.append(
                            {
                                "id": f"p{page.page_number}-w{idx}",
                                "text": chunk_text,
                                "page": page.page_number,
                                "span": [idx, idx + len(chunk_text)],
                                "bbox": bbox,
                            }
                        )
                else:
                    # Fallback to raw text if words not available
                    page_text = page.extract_text() or ""
                    for idx, line in enumerate(page_text.splitlines()):
                        cleaned = line.strip()
                        if not cleaned:
                            continue
                        blocks.append(
                            {
                                "id": f"p{page.page_number}-l{idx}",
                                "text": cleaned,
                                "page": page.page_number,
                                "span": [idx, idx + 1],
                                "bbox": [0.0, float(idx), 0.0, float(idx + 1)],
                            }
                        )

                pages.append(
                    {
                        "page_number": page.page_number,
                        "blocks": blocks,
                    }
                )

        return {"pages": pages}
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover - depends on pdfplumber internals
        print(f"[ingest] pdfplumber failed to produce layout ({exc}); falling back to text layout")

    try:
        import fitz  # type: ignore

        pages: list[dict[str, object]] = []
        with fitz.open(path) as doc:
            for page in doc:
                blocks: list[dict[str, object]] = []
                page_blocks = page.get_text("blocks") or []
                for idx, block in enumerate(page_blocks):
                    x0, y0, x1, y1, _, text, *_ = block + ("",)
                    text = (text or "").strip()
                    if not text:
                        continue
                    blocks.append(
                        {
                            "id": f"p{page.number + 1}-b{idx}",
                            "text": text,
                            "page": page.number + 1,
                            "span": [idx, idx + len(text)],
                            "bbox": [float(x0), float(y0), float(x1), float(y1)],
                        }
                    )

                pages.append({"page_number": page.number + 1, "blocks": blocks})

        return {"pages": pages}
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover
        print(f"[ingest] PyMuPDF layout extraction failed ({exc}); falling back to text layout")

    return None


__all__ = ["extract_text", "extract_document", "extract_layout"]
