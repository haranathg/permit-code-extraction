from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from city_code_ingest.main import run_pipeline


SAMPLE_PATH = Path(__file__).resolve().parents[2] / "samples" / "sanjose_building_code.pdf"


def test_pipeline_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    result = run_pipeline(
        SAMPLE_PATH,
        city="San Jose",
        state="CA",
        version="2025-01",
        source_url="https://example.org/code",
        output_dir=output_dir,
    )

    wizard_path = Path(result["wizard_path"])
    guidance_path = Path(result["guidance_path"])
    embeddings_path = Path(result["embeddings_path"])
    assert result["pinecone_enabled"] is False

    assert wizard_path.exists(), "wizard json should be generated"
    assert guidance_path.exists(), "guidance json should be generated"
    assert embeddings_path.exists(), "embeddings.json should be generated"

    with wizard_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["jurisdiction"]["city"] == "San Jose"
    assert payload["titles"], "titles should not be empty"
    first_title = payload["titles"][0]
    assert first_title["chapters"], "chapters should not be empty"
    first_section = first_title["chapters"][0]["sections"][0]
    assert first_section["section_id"].count(".") == 2
    assert isinstance(first_section["topics"], list)
    assert "Section" in first_section["breadcrumb"]
    assert ">" in first_section["breadcrumb"]
    assert 0 < len(first_section["text"]) <= 200
    decision_points = first_section["decision_points"]
    assert decision_points, "decision points should be extracted"
    assert decision_points[0]["question_id"].startswith(first_section["section_id"])
    assert "Yes" in decision_points[0]["response_options"]
    assert "guidance" not in decision_points[0]

    with guidance_path.open("r", encoding="utf-8") as handle:
        guidance_payload = json.load(handle)

    assert guidance_payload["guidance"], "guidance entries should exist"
    first_guidance = guidance_payload["guidance"][0]
    assert first_guidance.get("entry_type") == "section"
    assert first_guidance["guidance"], "section guidance should carry text"

    decision_guidances = [g for g in guidance_payload["guidance"] if g.get("entry_type") == "decision_point"]
    if decision_guidances:
        assert decision_guidances[0]["question_id"]
        assert "po_details" in decision_guidances[0]
        if decision_guidances[0]["po_details"]:
            assert decision_guidances[0]["po_details"][0]["po_id"].startswith("PO")

    with embeddings_path.open("r", encoding="utf-8") as handle:
        embeddings_payload = json.load(handle)

    assert embeddings_payload["embeddings"], "embeddings should not be empty"
    vector = embeddings_payload["embeddings"][0]["embedding"]
    assert len(vector) == 768
