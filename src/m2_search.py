from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words.

    Dùng underthesea.word_tokenize để tách từ ghép tiếng Việt (VD: "nghỉ_phép"),
    rồi replace "_" bằng " " để BM25 có thể split bằng khoảng trắng.
    """
    if not text:
        return ""
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ")
    except Exception:
        # Fallback: lowercase + collapse whitespace
        return " ".join(text.lower().split())


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        if not chunks:
            return
        self.documents = chunks
        self.corpus_tokens = []
        for chunk in chunks:
            segmented = segment_vietnamese(chunk.get("text", ""))
            tokens = segmented.split()
            self.corpus_tokens.append(tokens)
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or not self.documents:
            return []
        tokenized_query = segment_vietnamese(query).split()
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        # Sort by score descending
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results: list[SearchResult] = []
        for idx, score in ranked[:top_k]:
            if score <= 0:
                continue
            doc = self.documents[idx]
            results.append(SearchResult(
                text=doc.get("text", ""),
                score=float(score),
                metadata=doc.get("metadata", {}),
                method="bm25",
            ))
        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        if not chunks:
            return
        from qdrant_client.models import Distance, VectorParams, PointStruct
        self.client.recreate_collection(
            collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        texts = [c.get("text", "") for c in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=True, normalize_embeddings=True)
        points = []
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            payload = {**chunk.get("metadata", {}), "text": chunk.get("text", "")}
            points.append(PointStruct(id=i, vector=vec.tolist(), payload=payload))
        self.client.upsert(collection, points=points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        try:
            query_vector = self._get_encoder().encode(query, normalize_embeddings=True).tolist()
        except Exception as e:
            print(f"  ⚠️  Dense encode failed: {e}")
            return []
        try:
            response = self.client.query_points(collection, query=query_vector, limit=top_k)
        except Exception as e:
            print(f"  ⚠️  Qdrant query failed: {e}")
            return []
        results: list[SearchResult] = []
        for pt in response.points:
            payload = dict(pt.payload or {})
            text = payload.pop("text", "")
            results.append(SearchResult(
                text=text,
                score=float(pt.score),
                metadata=payload,
                method="dense",
            ))
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank + 1).

    Dùng text làm key để dedupe (giả định cùng text = cùng document).
    """
    rrf_scores: dict[str, dict] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            key = result.text
            if key not in rrf_scores:
                rrf_scores[key] = {"score": 0.0, "result": result}
            rrf_scores[key]["score"] += 1.0 / (k + rank + 1)

    # Sort by RRF score descending
    sorted_items = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)

    results: list[SearchResult] = []
    for item in sorted_items[:top_k]:
        orig: SearchResult = item["result"]
        results.append(SearchResult(
            text=orig.text,
            score=float(item["score"]),
            metadata=orig.metadata,
            method="hybrid",
        ))
    return results


class HybridSearch:
    """Combines BM25 + Dense + RRF."""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
