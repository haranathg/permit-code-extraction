from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from city_code_ingest.main import run_pipeline


SAMPLE_PATH = Path(__file__).resolve().parents[2] / "data" / "mbrc-planning-scheme-part-9.3.1.pdf"


def test_pipeline_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    result = run_pipeline(
        SAMPLE_PATH,
        city="Moreton Bay",
        state="QLD",
        version="2025-01",
        source_url="https://example.org/code",
        output_dir=output_dir,
    )

    wizard_path = Path(result["wizard_path"])
    guidance_path = Path(result["guidance_path"])
    catalog_path = Path(result["catalog_path"])
    validation_path = Path(result["validation_path"])
    embeddings_path = Path(result["embeddings_path"])
    assert isinstance(result["pinecone_enabled"], bool)

    assert wizard_path.exists(), "wizard json should be generated"
    assert guidance_path.exists(), "guidance json should be generated"
    assert catalog_path.exists(), "catalog json should be generated"
    assert validation_path.exists(), "validation json should be generated"
    assert embeddings_path.exists(), "embeddings.json should be generated"

    with wizard_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["jurisdiction"]["city"] == "Moreton Bay"
    assert payload["titles"], "titles should not be empty"
    first_title = payload["titles"][0]
    assert first_title["chapters"], "chapters should not be empty"
    first_section = first_title["chapters"][0]["sections"][0]
    assert isinstance(first_section["breadcrumbs"], list)
    assert isinstance(first_section["topics"], list)
    decision_points = first_section["decision_points"]
    assert decision_points, "decision points should be extracted"
    for dp in decision_points:
        assert "rad_id" in dp
        assert dp["source_refs"], "decision point should include source refs"
        assert dp["po_links"] or dp.get("no_po_applicable")

    with guidance_path.open("r", encoding="utf-8") as handle:
        guidance_payload = json.load(handle)

    assert guidance_payload["guidance"], "guidance entries should exist"
    first_guidance = guidance_payload["guidance"][0]
    assert first_guidance.get("entry_type") == "section"
    assert first_guidance["guidance"], "section guidance should carry text"

    decision_guidances = [g for g in guidance_payload["guidance"] if g.get("entry_type") == "decision_point"]
    assert decision_guidances, "decision guidances should be captured"
    for dg in decision_guidances:
        assert "po_details" in dg

    with catalog_path.open("r", encoding="utf-8") as handle:
        catalog_payload = json.load(handle)

    rad_count = len(catalog_payload.get("RAD", []))
    assert rad_count == len(result["decision_points"])

    with validation_path.open("r", encoding="utf-8") as handle:
        validation_payload = json.load(handle)

    assert validation_payload["status"] == "ok"
    assert not any(validation_payload["issues"].values())

    with embeddings_path.open("r", encoding="utf-8") as handle:
        embeddings_payload = json.load(handle)

    assert embeddings_payload, "embeddings should not be empty"
    assert len(embeddings_payload) == len(result["decision_points"])
    vector = embeddings_payload[0]["embedding"]
    assert len(vector) == 1536
    assert embeddings_payload[0]["metadata"].get("rad_id")
