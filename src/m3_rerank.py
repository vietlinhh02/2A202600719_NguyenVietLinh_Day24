from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


_MODEL_CACHE: dict[str, object] = {}


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu"):
        # Mặc định dùng CPU để tránh CUDA OOM khi GPU chật (bge-reranker ~2GB).
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            # Cache model ở class-level để tránh load nhiều lần → tránh CUDA OOM
            global _MODEL_CACHE
            cache_key = f"{self.model_name}::{self.device}"
            if cache_key not in _MODEL_CACHE:
                from sentence_transformers import CrossEncoder
                _MODEL_CACHE[cache_key] = CrossEncoder(self.model_name, device=self.device)
            self._model = _MODEL_CACHE[cache_key]
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-N → top-k."""
        if not documents:
            return []
        model = self._load_model()
        if model is None:
            return []

        pairs = [(query, doc.get("text", "")) for doc in documents]
        try:
            scores = model.predict(pairs, show_progress_bar=False)
        except Exception as e:
            print(f"  ⚠️  CrossEncoder predict failed: {e}")
            return []
        # Đảm bảo là list[float]
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if isinstance(scores, (int, float)):
            scores = [scores]

        scored = list(zip(scores, documents))
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[RerankResult] = []
        for i, (score, doc) in enumerate(scored[:top_k]):
            results.append(RerankResult(
                text=doc.get("text", ""),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i,
            ))
        return results


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        try:
            from flashrank import Ranker, RerankRequest
            if self._model is None:
                self._model = Ranker()
            passages = [{"text": d.get("text", "")} for d in documents]
            req = RerankRequest(query=query, passages=passages)
            raw = self._model.rerank(req)
            # raw is list of dicts sorted by score desc
            scored = sorted(
                zip([r["score"] for r in raw], documents),
                key=lambda x: x[0], reverse=True,
            )
            results: list[RerankResult] = []
            for i, (score, doc) in enumerate(scored[:top_k]):
                results.append(RerankResult(
                    text=doc.get("text", ""),
                    original_score=float(doc.get("score", 0.0)),
                    rerank_score=float(score),
                    metadata=doc.get("metadata", {}),
                    rank=i,
                ))
            return results
        except Exception:
            return []


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
