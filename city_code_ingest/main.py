"""Local entry-point for the building code ingestion pipeline."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Dict

from city_code_ingest import builder, chunker, embedder, enricher, ingest


MODULE_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = MODULE_ROOT / "output"
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"


def run_pipeline(
    input_path: str | Path,
    *,
    city: str = "San Jose",
    state: str = "CA",
    version: str = "2025-01",
    source_url: str = "",
    output_dir: str | Path | None = None,
    pinecone_api_key: str | None = None,
    pinecone_index_name: str | None = None,
    pinecone_environment: str | None = None,
    pinecone_namespace: str | None = None,
    pinecone_host: str | None = None,
) -> Dict[str, Any]:
    source_path = Path(input_path)
    output_root = Path(output_dir) if output_dir else OUTPUT_DIR
    intermediate_dir = output_root / "intermediate"

    output_root.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    lines = ingest.extract_text(str(source_path))
    _write_lines(intermediate_dir / "raw_text.txt", lines)

    sections = chunker.split_sections(lines)
    _write_json(intermediate_dir / "sections.json", sections)

    enriched_sections = enricher.add_metadata(sections)
    _write_json(intermediate_dir / "enriched_sections.json", enriched_sections)

    payload = builder.build_json(
        enriched_sections,
        city=city,
        state=state,
        version=version,
        source_url=source_url,
    )
    wizard_payload, guidance_payload = _split_payload(payload)

    wizard_filename = f"{source_path.stem}_wizard.json"
    wizard_path = output_root / wizard_filename
    _write_json(wizard_path, wizard_payload)

    guidance_filename = f"{source_path.stem}_guidance.json"
    guidance_path = output_root / guidance_filename
    _write_json(guidance_path, guidance_payload)

    pinecone_config = _resolve_pinecone_config(
        api_key=pinecone_api_key,
        index_name=pinecone_index_name,
        environment=pinecone_environment,
        namespace=pinecone_namespace,
        host=pinecone_host,
    )

    embeddings_filename = f"{source_path.stem}_embeddings.json"
    embeddings_path = embedder.generate_embeddings(
        wizard_path,
        output_root / embeddings_filename,
        pinecone_config=pinecone_config,
        extra_metadata=payload.get("jurisdiction"),
    )

    return {
        "wizard_path": wizard_path,
        "guidance_path": guidance_path,
        "embeddings_path": embeddings_path,
        "payload": payload,
        "pinecone_enabled": bool(pinecone_config),
    }


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"[main] Wrote {len(lines)} lines to {path}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[main] Saved JSON to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the city code ingestion pipeline")
    parser.add_argument("--input", required=True, help="Path to PDF or HTML source document")
    parser.add_argument("--city", default="San Jose")
    parser.add_argument("--state", default="CA")
    parser.add_argument("--version", default="2025-01")
    parser.add_argument("--source-url", default="", dest="source_url")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--pinecone-api-key", default=None)
    parser.add_argument("--pinecone-index-name", default=None)
    parser.add_argument("--pinecone-environment", default=None)
    parser.add_argument("--pinecone-namespace", default=None)
    parser.add_argument("--pinecone-host", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        args.input,
        city=args.city,
        state=args.state,
        version=args.version,
        source_url=args.source_url,
        output_dir=args.output_dir,
        pinecone_api_key=args.pinecone_api_key,
        pinecone_index_name=args.pinecone_index_name,
        pinecone_environment=args.pinecone_environment,
        pinecone_namespace=args.pinecone_namespace,
        pinecone_host=args.pinecone_host,
    )


def _generate_section_summary(title: str | None, text: str) -> str:
    if not text:
        return title or ""

    normalized = " ".join(text.split())
    if not normalized:
        return title or ""

    match = re.match(r"(.+?[.!?])(?:\s|$)", normalized)
    if match:
        summary = match.group(1)
    else:
        summary = normalized[:200]

    if len(summary) > 200:
        summary = summary[:197].rstrip() + "..."

    return summary


def _split_payload(payload: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    wizard_payload = copy.deepcopy(payload)
    guidance_entries: list[dict[str, Any]] = []

    for title in wizard_payload.get("titles", []):
        title_number = title.get("title_number")
        title_name = title.get("title_name")
        for chapter in title.get("chapters", []):
            chapter_number = chapter.get("chapter_number")
            chapter_name = chapter.get("chapter_name")
            for section in chapter.get("sections", []):
                section_id = section.get("section_id")
                section_title = section.get("section_title")
                breadcrumb = section.get("breadcrumb")
                section_text = section.get("text", "")
                section["text"] = _generate_section_summary(section_title, section_text)
                guidance_entries.append(
                    {
                        "entry_type": "section",
                        "section_id": section_id,
                        "section_title": section_title,
                        "title_number": title_number,
                        "title_name": title_name,
                        "chapter_number": chapter_number,
                        "chapter_name": chapter_name,
                        "breadcrumb": breadcrumb,
                        "guidance": section_text.strip(),
                    }
                )
                decision_points = section.get("decision_points", [])
                for decision_point in decision_points:
                    rad_text = (decision_point.get("rad_text") or "").strip()
                    po_details_raw = decision_point.get("po_details") or []
                    guidance_text = (decision_point.pop("guidance", "") or "").strip()

                    cleaned_po_details = [
                        {k: v for k, v in detail.items() if v}
                        for detail in po_details_raw
                        if detail.get("po_id") or (detail.get("text") and detail.get("text").strip())
                    ]

                    if cleaned_po_details:
                        decision_point["po_details"] = cleaned_po_details
                    elif "po_details" in decision_point:
                        decision_point.pop("po_details", None)

                    if rad_text:
                        decision_point["rad_text"] = rad_text
                    elif "rad_text" in decision_point:
                        decision_point.pop("rad_text", None)

                    if not guidance_text:
                        parts: list[str] = []
                        if rad_text:
                            parts.append(rad_text)
                        for detail in cleaned_po_details:
                            po_id = detail.get("po_id")
                            po_text = (detail.get("text") or "").strip()
                            if not po_text:
                                continue
                            if po_id:
                                parts.append(f"{po_id}: {po_text}")
                            else:
                                parts.append(po_text)
                        guidance_text = "\n\n".join(parts).strip()

                    if not (guidance_text or rad_text or cleaned_po_details):
                        continue
                    guidance_entries.append(
                        {
                            "entry_type": "decision_point",
                            "question_id": decision_point.get("question_id"),
                            "section_id": section_id,
                            "section_title": section_title,
                            "title_number": title_number,
                            "title_name": title_name,
                            "chapter_number": chapter_number,
                            "chapter_name": chapter_name,
                            "breadcrumb": breadcrumb,
                            "po_references": decision_point.get("po_references", []),
                            "guidance": guidance_text,
                            "rad_text": rad_text,
                            "po_details": cleaned_po_details,
                        }
                    )

    guidance_payload: Dict[str, Any] = {
        "jurisdiction": payload.get("jurisdiction", {}),
        "guidance": guidance_entries,
    }

    return wizard_payload, guidance_payload


def _resolve_pinecone_config(
    *,
    api_key: str | None,
    index_name: str | None,
    environment: str | None,
    namespace: str | None,
    host: str | None,
) -> Dict[str, str] | None:
    api_key = api_key or os.getenv("PINECONE_API_KEY")
    index_name = index_name or os.getenv("PINECONE_INDEX")
    environment = environment or os.getenv("PINECONE_ENVIRONMENT") or os.getenv("PINECONE_ENV")
    namespace = namespace or os.getenv("PINECONE_NAMESPACE")
    host = host or os.getenv("PINECONE_HOST")

    if not (api_key and index_name):
        return None

    config: Dict[str, str] = {
        "api_key": api_key,
        "index_name": index_name,
    }
    if environment:
        config["environment"] = environment
    if namespace:
        config["namespace"] = namespace
    if host:
        config["host"] = host
    return config


if __name__ == "__main__":
    main()
