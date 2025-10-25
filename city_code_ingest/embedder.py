"""Embedding utilities for wizard decision points."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback
    def tqdm(iterable, **_):
        return iterable


EMBEDDING_DIM = 1536


def generate_embeddings(
    json_path: str | Path,
    output_path: str | Path | None = None,
    *,
    pinecone_config: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    use_llm: bool = False,
    embed_level: str = "decision_point",
) -> Path:
    input_path = Path(json_path)
    if not input_path.exists():
        raise FileNotFoundError(f"JSON payload not found: {json_path}")

    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    jurisdiction = payload.get("jurisdiction", {})
    city = jurisdiction.get("city", "city")
    version = jurisdiction.get("version", "version")
    namespace = f"{_slug(city)}_{version}"

    vectors: list[Dict[str, Any]] = []
    section_count = 0
    item_count = 0

    for title in payload.get("titles", []):
        title_name = title.get("title_name", "")
        for chapter in title.get("chapters", []):
            chapter_name = chapter.get("chapter_name", "")
            for section in chapter.get("sections", []):
                section_count += 1
                section_vectors, section_items = _embed_section(
                    section,
                    title_name=title_name,
                    chapter_name=chapter_name,
                    city=city,
                    version=version,
                    use_llm=use_llm,
                    embed_level=embed_level,
                )
                vectors.extend(section_vectors)
                item_count += section_items

    if not vectors:  # fallback to section-level embeddings when decision points absent
        for title in payload.get("titles", []):
            title_name = title.get("title_name", "")
            for chapter in title.get("chapters", []):
                chapter_name = chapter.get("chapter_name", "")
                for section in chapter.get("sections", []):
                    section_id = section.get("section_id", "section")
                    text = section.get("text") or section.get("section_title", "")
                    vector = _generate_vector(text, seed=section_id, use_llm=use_llm)
                    vectors.append(
                        {
                            "id": _vector_id(city, version, section_id),
                            "embedding": vector,
                            "metadata": {
                                "type": "section",
                                "section_id": section_id,
                                "section_title": section.get("section_title"),
                                "chapter": chapter_name,
                                "title": title_name,
                                "page_span": None,
                                "jurisdiction": city,
                                "po_ids": [],
                            },
                        }
                    )
        item_count = len(vectors)

    target_path = (
        Path(output_path)
        if output_path
        else input_path.parent / f"{_slug(city)}_{version}_embeddings.json"
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(vectors, handle, indent=2)

    if pinecone_config and vectors:
        _persist_to_pinecone(
            vectors,
            pinecone_config,
            namespace=namespace,
            extra_metadata=extra_metadata,
        )

    print(f"[embedder] Embedded {item_count} {embed_level}(s) across {section_count} sections.")
    print(f"[embedder] Wrote embeddings to {target_path}")
    return target_path


def _embed_section(
    section: Dict[str, Any],
    *,
    title_name: str,
    chapter_name: str,
    city: str,
    version: str,
    use_llm: bool,
    embed_level: str,
) -> tuple[List[Dict[str, Any]], int]:
    section_id = section.get("section_id", "section")
    decision_points = section.get("decision_points", []) or []

    if embed_level == "decision_point" and decision_points:
        records: list[Dict[str, Any]] = []
        for dp in tqdm(decision_points, desc="Embedding decision points", leave=False):
            rad_id = dp.get("rad_id", "RAD")
            text_blob = _compose_decision_blob(dp)
            vector = _generate_vector(text_blob, seed=rad_id, use_llm=use_llm)
            metadata = {
                "type": "decision_point",
                "rad_id": rad_id,
                "po_ids": dp.get("po_links", []),
                "section_id": section_id,
                "section_title": section.get("section_title"),
                "chapter": chapter_name,
                "title": title_name,
                "page_span": _resolve_page_span(dp.get("source_refs", [])),
                "jurisdiction": city,
            }
            records.append(
                {
                    "id": _vector_id(city, version, rad_id),
                    "embedding": vector,
                    "metadata": metadata,
                }
            )
        return records, len(records)

    if embed_level == "po" and decision_points:
        records: list[Dict[str, Any]] = []
        for dp in tqdm(decision_points, desc="Embedding PO entries", leave=False):
            rad_id = dp.get("rad_id", "RAD")
            for detail in dp.get("po_details", []):
                po_id = detail.get("po_id", "PO")
                text_blob = _compose_po_blob(detail, dp)
                vector = _generate_vector(text_blob, seed=f"{rad_id}_{po_id}", use_llm=use_llm)
                metadata = {
                    "type": "po",
                    "rad_id": rad_id,
                    "po_id": po_id,
                    "section_id": section_id,
                    "section_title": section.get("section_title"),
                    "chapter": chapter_name,
                    "title": title_name,
                    "page_span": detail.get("span") or _resolve_page_span(dp.get("source_refs", [])),
                    "jurisdiction": city,
                }
                records.append(
                    {
                        "id": _vector_id(city, version, f"{rad_id}_{po_id}"),
                        "embedding": vector,
                        "metadata": metadata,
                    }
                )
        return records, len(records)

    text_blob = section.get("text") or section.get("section_title", "")
    vector = _generate_vector(text_blob, seed=section_id, use_llm=use_llm)
    metadata = {
        "type": "section",
        "section_id": section_id,
        "section_title": section.get("section_title"),
        "chapter": chapter_name,
        "title": title_name,
        "page_span": None,
        "jurisdiction": city,
        "po_ids": [],
    }
    return (
        [
            {
                "id": _vector_id(city, version, section_id),
                "embedding": vector,
                "metadata": metadata,
            }
        ],
        1,
    )


def _generate_vector(text: str, *, seed: str, use_llm: bool) -> List[float]:
    text = text or ""
    if use_llm:
        try:  # pragma: no cover - network dependency
            import openai

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set")

            client = openai.OpenAI(api_key=api_key)
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            return response.data[0].embedding
        except Exception as exc:  # pragma: no cover
            print(f"[embedder] OpenAI embedding failed ({exc}); using fallback")

    rng = random.Random(hash(seed) % 1_000_000)
    return [rng.uniform(-1.0, 1.0) for _ in range(EMBEDDING_DIM)]


def _persist_to_pinecone(
    vectors: Iterable[Dict[str, Any]],
    pinecone_config: Dict[str, Any],
    *,
    namespace: str,
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

    processed = 0
    vectors_list = list(vectors)
    batch_size = 50

    try:
        for start in range(0, len(vectors_list), batch_size):
            batch = vectors_list[start:start + batch_size]
            pinecone_records = []
            for record in batch:
                metadata = dict(record.get("metadata", {}))
                if extra_metadata:
                    for key, value in extra_metadata.items():
                        if value is not None:
                            metadata[f"jurisdiction_{key}"] = value

                pinecone_records.append(
                    {
                        "id": str(record["id"]),
                        "values": record.get("embedding", []),
                        "metadata": metadata,
                    }
                )

            processed += store.upsert_embeddings(pinecone_records, namespace=namespace)

        print(f"[embedder] Upserted {processed} vectors to Pinecone index '{store.index_name}'")
    except Exception as exc:  # pragma: no cover - network failure
        print(f"[embedder] Unable to persist embeddings to Pinecone: {exc}")


def _compose_decision_blob(decision_point: Dict[str, Any]) -> str:
    rad_id = decision_point.get("rad_id", "RAD")
    rad_text = decision_point.get("rad_text", "")

    po_lines = [
        f"{detail.get('po_id', '')}: {detail.get('text', '').strip()}"
        for detail in decision_point.get("po_details", [])
        if detail.get("text")
    ]
    ead_lines = [
        f"{detail.get('ead_id', '')}: {detail.get('text', '').strip()}"
        for detail in decision_point.get("ead_details", [])
        if detail.get("text")
    ]

    parts = [f"{rad_id}: {rad_text}".strip()]
    if po_lines:
        parts.append("POs:\n" + "\n".join(po_lines))
    if ead_lines:
        parts.append("EADs:\n" + "\n".join(ead_lines))
    return "\n\n".join(filter(None, parts))


def _compose_po_blob(detail: Dict[str, Any], decision_point: Dict[str, Any]) -> str:
    po_id = detail.get("po_id", "PO")
    po_text = detail.get("text", "")
    rad_text = decision_point.get("rad_text", "")
    return f"{po_id}: {po_text}\nRelated RAD: {rad_text}"


def _resolve_page_span(source_refs: List[Dict[str, Any]]) -> Optional[List[Any]]:
    if not source_refs:
        return None
    ref = source_refs[0]
    span = ref.get("span")
    if span and isinstance(span, list):
        return span
    return None


def _vector_id(city: str, version: str, identifier: str) -> str:
    return f"{_slug(city)}_{version}_{identifier}".replace(" ", "")


def _slug(value: str) -> str:
    return value.replace(" ", "")


__all__ = ["generate_embeddings"]
