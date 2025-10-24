"""Utility wrappers for Pinecone vector storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional


@dataclass
class PineconeVectorStore:
    api_key: str
    index_name: str
    environment: Optional[str] = None
    host: Optional[str] = None

    _client: Any = None
    _index: Any = None

    def _ensure_index(self) -> Any:
        if self._index is not None:
            return self._index

        try:
            import pinecone  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise RuntimeError("pinecone-client package is not installed") from exc

        if hasattr(pinecone, "Pinecone"):
            kwargs = {"api_key": self.api_key}
            if self.environment:
                kwargs["environment"] = self.environment
            client = pinecone.Pinecone(**kwargs)
            index_kwargs = {}
            if self.host:
                index_kwargs["host"] = self.host
            index = client.Index(self.index_name, **index_kwargs)
            self._client = client
            self._index = index
            return index

        # Legacy SDK fallback
        init_kwargs = {"api_key": self.api_key}
        if self.environment:
            init_kwargs["environment"] = self.environment
        pinecone.init(**init_kwargs)
        index_kwargs = {}
        if self.host:
            index_kwargs["host"] = self.host
        index = pinecone.Index(self.index_name, **index_kwargs)
        self._client = pinecone
        self._index = index
        return index

    @property
    def index(self) -> Any:
        return self._ensure_index()

    def upsert_embeddings(self, vectors: Iterable[dict], *, namespace: Optional[str] = None) -> int:
        vectors_list: List[dict] = list(vectors)
        if not vectors_list:
            return 0

        index = self._ensure_index()
        index.upsert(vectors=vectors_list, namespace=namespace)
        return len(vectors_list)


__all__ = ["PineconeVectorStore"]
