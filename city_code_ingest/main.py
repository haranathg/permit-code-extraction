"""Local entry-point for the city code ingestion pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from city_code_ingest import (
    builder,
    chunker,
    embedder,
    ingest,
    mapper,
    schema_extractor,
    validator,
)


MODULE_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = MODULE_ROOT / "output"

load_dotenv()


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
    use_llm: bool = False,
) -> Dict[str, Any]:
    source_path = Path(input_path)
    output_root = Path(output_dir) if output_dir else OUTPUT_DIR
    intermediate_dir = output_root / "intermediate"

    output_root.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    document = ingest.extract_document(str(source_path))
    lines: List[str] = document["lines"]  # type: ignore[assignment]
    layout: Dict[str, object] = document["layout"]  # type: ignore[assignment]

    _write_lines(intermediate_dir / "raw_text.txt", lines)
    _write_json(intermediate_dir / "layout.json", layout)

    sections = chunker.split_sections(lines)
    _write_json(intermediate_dir / "sections.json", sections)

    catalog = schema_extractor.catalog_items(layout, use_llm=use_llm)
    catalog_path = output_root / f"{source_path.stem}_catalog.json"
    schema_extractor.save_catalog(catalog, catalog_path)

    decision_points = mapper.link_items(catalog, layout)
    _write_json(intermediate_dir / "decision_points.json", decision_points)

    wizard_payload, guidance_payload = builder.build_outputs(
        decision_points,
        sections=sections,
        catalog=catalog,
        city=city,
        state=state,
        version=version,
        source_url=source_url,
    )

    wizard_path = output_root / f"{source_path.stem}_wizard.json"
    guidance_path = output_root / f"{source_path.stem}_guidance.json"
    _write_json(wizard_path, wizard_payload)
    _write_json(guidance_path, guidance_payload)

    validation_report = validator.run_checks(wizard=wizard_payload, catalog=catalog)
    validation_path = output_root / f"{source_path.stem}_validation.json"
    validator.save_report(validation_report, validation_path)

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
        extra_metadata=wizard_payload.get("jurisdiction"),
        use_llm=use_llm,
    )

    _log_summary(catalog, decision_points, validation_report)

    return {
        "wizard_path": wizard_path,
        "guidance_path": guidance_path,
        "catalog_path": catalog_path,
        "validation_path": validation_path,
        "embeddings_path": embeddings_path,
        "catalog": catalog,
        "decision_points": decision_points,
        "pinecone_enabled": bool(pinecone_config),
    }


def _write_lines(path: Path, lines: List[str]) -> None:
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
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM-backed extraction and embeddings")
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
        use_llm=args.use_llm,
    )


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


def _log_summary(
    catalog: Dict[str, List[Dict[str, object]]],
    decision_points: List[Dict[str, object]],
    validation_report: Dict[str, object],
) -> None:
    rad_count = len(catalog.get("RAD", []))
    po_count = len(catalog.get("PO", []))
    ead_count = len(catalog.get("EAD", []))
    print(
        f"[main] Summary: RAD={rad_count}, PO={po_count}, EAD={ead_count}, decision_points={len(decision_points)}"
    )
    issue_counts = {k: len(v) for k, v in validation_report.get("issues", {}).items()}
    print(f"[main] Validation issues: {issue_counts}")


if __name__ == "__main__":
    main()
