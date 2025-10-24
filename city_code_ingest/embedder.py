"""Simulated embedding generator for indexed sections."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


EMBEDDING_DIM = 768


def generate_embeddings(
    json_path: str | Path,
    output_path: str | Path | None = None,
    *,
    pinecone_config: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Produce mock Titan-style embeddings for each section."""

    input_path = Path(json_path)
    if not input_path.exists():
        raise FileNotFoundError(f"JSON payload not found: {json_path}")

    print(f"[embedder] Generating embeddings for {input_path}")
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    embeddings: list[dict[str, Any]] = []
    rng = random.Random(42)

    for title in payload.get("titles", []):
        for chapter in title.get("chapters", []):
            for section in chapter.get("sections", []):
                vector = [rng.uniform(-1.0, 1.0) for _ in range(EMBEDDING_DIM)]
                embeddings.append(
                    {
                        "section_id": section.get("section_id"),
                        "title_number": title.get("title_number"),
                        "chapter_number": chapter.get("chapter_number"),
                        "breadcrumb": section.get("breadcrumb", ""),
                        "section_title": section.get("section_title", ""),
                        "text": section.get("text", ""),
                        "embedding": vector,
                    }
                )

    target_path = Path(output_path) if output_path else input_path.parent / "embeddings.json"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with target_path.open("w", encoding="utf-8") as handle:
        json.dump({"embeddings": embeddings}, handle, indent=2)

    if pinecone_config:
        _persist_to_pinecone(embeddings, pinecone_config, extra_metadata=extra_metadata)

    print(f"[embedder] Wrote embeddings to {target_path}")
    return target_path


def _persist_to_pinecone(
    embeddings: Iterable[Dict[str, Any]],
    pinecone_config: Dict[str, Any],
    *,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    required_keys = {"api_key", "index_name"}
    if not required_keys.issubset(pinecone_config):
        print("[embedder] Pinecone config incomplete; skipping upsert")
        return

    try:
        from city_code_ingest.vector_store import PineconeVectorStore
    except ImportError:
        print("[embedder] Pinecone support module unavailable; skipping upsert")
        return

    store = PineconeVectorStore(
        api_key=pinecone_config["api_key"],
        index_name=pinecone_config["index_name"],
        environment=pinecone_config.get("environment"),
        host=pinecone_config.get("host"),
    )

    namespace = pinecone_config.get("namespace")
    records = []
    for item in embeddings:
        section_id = item.get("section_id")
        if not section_id:
            continue

        metadata: Dict[str, Any] = {
            "section_id": section_id,
            "title_number": item.get("title_number"),
            "chapter_number": item.get("chapter_number"),
            "breadcrumb": item.get("breadcrumb", ""),
            "section_title": item.get("section_title", ""),
            "text": item.get("text", ""),
        }

        if extra_metadata:
            for key, value in extra_metadata.items():
                if value is None:
                    continue
                metadata_key = f"jurisdiction_{key}"
                if isinstance(value, (str, int, float, bool)):
                    metadata_value = value
                else:
                    metadata_value = str(value)
                metadata[metadata_key] = metadata_value

        records.append({
            "id": str(section_id),
            "values": item.get("embedding", []),
            "metadata": metadata,
        })

    if not records:
        print("[embedder] No valid vectors to upsert")
        return

    try:
        upserted = store.upsert_embeddings(records, namespace=namespace)
        print(f"[embedder] Upserted {upserted} vectors to Pinecone index '{store.index_name}'")
    except RuntimeError as exc:
        print(f"[embedder] Unable to persist embeddings to Pinecone: {exc}")


__all__ = ["generate_embeddings"]
