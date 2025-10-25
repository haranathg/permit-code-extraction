from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import city_code_ingest.embedder as embedder


def test_generate_embeddings_decision_point(monkeypatch, tmp_path: Path) -> None:
    wizard_payload = {
        "jurisdiction": {"city": "Test City", "version": "2025-01"},
        "titles": [
            {
                "title_number": 1,
                "title_name": "Title 1",
                "chapters": [
                    {
                        "chapter_number": 1,
                        "chapter_name": "Chapter 1",
                        "sections": [
                            {
                                "section_id": "1.1.1",
                                "section_title": "Section 1",
                                "text": "Section body",
                                "decision_points": [
                                    {
                                        "rad_id": "RAD1",
                                        "rad_text": "Requirement text",
                                        "po_links": ["PO1"],
                                        "po_details": [
                                            {
                                                "po_id": "PO1",
                                                "text": "PO body",
                                                "span": [0, 10],
                                                "page": 1,
                                            }
                                        ],
                                        "ead_links": [],
                                        "ead_details": [],
                                        "source_refs": [{"page": 1, "span": [0, 10]}],
                                    },
                                    {
                                        "rad_id": "RAD2",
                                        "rad_text": "Another requirement",
                                        "po_links": [],
                                        "po_details": [],
                                        "ead_links": [],
                                        "ead_details": [],
                                        "source_refs": [{"page": 2, "span": [5, 15]}],
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }

    wizard_path = tmp_path / "wizard.json"
    with wizard_path.open("w", encoding="utf-8") as handle:
        json.dump(wizard_payload, handle)

    batches: list[List[Dict[str, Any]]] = []

    def fake_persist(vectors, pinecone_config, *, namespace=None, extra_metadata=None):
        batches.append(list(vectors))

    monkeypatch.setattr(embedder, "_persist_to_pinecone", fake_persist)

    output_path = tmp_path / "embeddings.json"
    result_path = embedder.generate_embeddings(
        wizard_path,
        output_path=output_path,
        pinecone_config={"api_key": "key", "index_name": "idx"},
        embed_level="decision_point",
    )

    with result_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    assert len(data) == 2
    assert all("rad_id" in item["metadata"] for item in data)
    assert len(data[0]["embedding"]) == embedder.EMBEDDING_DIM

    total_upserts = sum(len(batch) for batch in batches)
    assert total_upserts == len(data)
