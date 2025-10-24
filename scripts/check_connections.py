"""Utility script to verify Pinecone and OpenAI connectivity."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def _clean_env_value(key: str) -> Optional[str]:
    value = os.getenv(key)
    if value is None:
        return None
    return value.strip() or None


def _to_serializable(value):
    if isinstance(value, (dict, list, str, int, float, type(None))):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return {k: _to_serializable(v) for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def load_env(path: Path) -> None:
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.strip().startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.strip().split("=", 1)
            if key and key not in os.environ:
                os.environ[key] = value


def check_openai(model: str = "text-embedding-3-small") -> bool:
    api_key = _clean_env_value("OPENAI_API_KEY")
    if not api_key:
        print("[openai] OPENAI_API_KEY not set")
        return False

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        print("[openai] openai package not installed")
        return False

    client = OpenAI(api_key=api_key)

    try:
        response = client.embeddings.create(model=model, input="connection test")
    except Exception as exc:  # pragma: no cover - requires network access
        print(f"[openai] Request failed: {exc}")
        return False

    vector_len = len(response.data[0].embedding)
    print(f"[openai] Success - received embedding of length {vector_len}")
    return True


def check_pinecone(namespace: Optional[str] = None) -> bool:
    api_key = _clean_env_value("PINECONE_API_KEY")
    index_name = _clean_env_value("PINECONE_INDEX")
    environment = _clean_env_value("PINECONE_ENVIRONMENT") or _clean_env_value("PINECONE_ENV")
    host = _clean_env_value("PINECONE_HOST")

    if not api_key or not index_name:
        print("[pinecone] PINECONE_API_KEY and PINECONE_INDEX must be set")
        return False

    try:
        import pinecone  # type: ignore
    except ImportError:
        print("[pinecone] pinecone-client package not installed")
        return False

    try:
        if hasattr(pinecone, "Pinecone"):
            kwargs = {"api_key": api_key}
            if environment:
                kwargs["environment"] = environment
            client = pinecone.Pinecone(**kwargs)
            index_kwargs = {}
            if host:
                index_kwargs["host"] = host
            index = client.Index(index_name, **index_kwargs)
        else:
            init_kwargs = {"api_key": api_key}
            if environment:
                init_kwargs["environment"] = environment
            pinecone.init(**init_kwargs)
            index_kwargs = {}
            if host:
                index_kwargs["host"] = host
            index = pinecone.Index(index_name, **index_kwargs)

        stats = index.describe_index_stats()
        print("[pinecone] Index stats:")
        print(json.dumps(_to_serializable(stats), indent=2))

        if namespace:
            namespace_stats = index.describe_namespace_statistics(namespace=namespace)
            print(f"[pinecone] Namespace '{namespace}' stats:")
            print(json.dumps(_to_serializable(namespace_stats), indent=2))

        return True
    except Exception as exc:  # pragma: no cover - requires network access
        print(f"[pinecone] Request failed: {exc}")
        return False


def main() -> None:
    load_env(ENV_PATH)

    namespace = _clean_env_value("PINECONE_NAMESPACE") or None
    if namespace in {"", "default"}:
        namespace = None

    ok_openai = check_openai()
    ok_pinecone = check_pinecone(namespace=namespace)

    if ok_openai and ok_pinecone:
        print("[status] All integrations verified")
    else:
        print("[status] Integration check failed")


if __name__ == "__main__":
    main()
