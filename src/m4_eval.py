from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, OPENAI_API_KEY, call_llm


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _has_valid_llm() -> bool:
    """Check có LLM hợp lệ không."""
    if False:
        return True
    if OPENAI_API_KEY and len(OPENAI_API_KEY) >= 20 and not OPENAI_API_KEY.startswith("sk-..."):
        return True
    return False


def _llm_judge_score(prompt_system: str, prompt_user: str, max_tokens: int = 10) -> float:
    """Gọi LLM để chấm điểm 0-1. Trả về 0.0 nếu lỗi."""
    result = call_llm(prompt_system, prompt_user, max_tokens=max_tokens)
    if not result:
        return 0.0
    # Parse số từ response
    import re
    match = re.search(r'(\d+(?:\.\d+)?)', result)
    if match:
        try:
            val = float(match.group(1))
            return max(0.0, min(1.0, val))
        except ValueError:
            pass
    return 0.0


def _compute_context_precision(contexts: list[str], ground_truth: str) -> float:
    """Context precision: tỷ lệ contexts có chứa thông tin từ ground truth."""
    if not contexts:
        return 0.0
    gt_words = set(ground_truth.lower().split())
    if len(gt_words) < 3:
        return 0.0
    # Lấy các từ "quan trọng" từ ground truth (>4 chars, không phải stopwords)
    stopwords = {"là", "của", "và", "có", "được", "cho", "trong", "khi", "này", "một", "những", "các", "theo", "đến", "từ"}
    important = [w for w in gt_words if len(w) > 4 and w not in stopwords]
    if not important:
        return 0.0
    relevant_count = 0
    for ctx in contexts:
        ctx_lower = ctx.lower()
        matches = sum(1 for w in important if w in ctx_lower)
        if matches >= max(1, len(important) * 0.3):
            relevant_count += 1
    return relevant_count / len(contexts)


def _compute_context_recall(contexts: list[str], ground_truth: str) -> float:
    """Context recall: tỷ lệ thông tin ground truth xuất hiện trong contexts."""
    if not contexts or not ground_truth:
        return 0.0
    combined = " ".join(contexts).lower()
    gt_words = set(ground_truth.lower().split())
    if len(gt_words) < 3:
        return 0.0
    stopwords = {"là", "của", "và", "có", "được", "cho", "trong", "khi", "này", "một", "những", "các", "theo", "đến", "từ"}
    important = [w for w in gt_words if len(w) > 4 and w not in stopwords]
    if not important:
        return 0.0
    matched = sum(1 for w in important if w in combined)
    return matched / len(important)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Custom RAGAS-style evaluation dùng Google Gen AI.

    Metrics:
    - faithfulness: LLM judge xem answer có dựa trên context không
    - answer_relevancy: LLM judge xem answer có trả lời câu hỏi không
    - context_precision: Heuristic overlap giữa contexts và ground truth
    - context_recall: Heuristic overlap giữa ground truth và contexts
    """
    if not questions:
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0, "per_question": []}

    has_llm = _has_valid_llm()
    print(f"  📊 Evaluating {len(questions)} questions (LLM judge: {'YES' if has_llm else 'NO — heuristic only'})", flush=True)

    per_question: list[EvalResult] = []
    sum_f, sum_ar, sum_cp, sum_cr = 0.0, 0.0, 0.0, 0.0

    for i, (q, a, ctxs, gt) in enumerate(zip(questions, answers, contexts, ground_truths)):
        # 1) Faithfulness: LLM judge
        if has_llm and ctxs and a:
            context_str = "\n".join(ctxs[:3])
            f_score = _llm_judge_score(
                "Bạn là giám khảo. Chấm điểm 0.0-1.0 cho độ trung thực (faithfulness) của answer với context. Trả lời chỉ bằng MỘT con số. 1.0 = answer hoàn toàn dựa trên context, 0.0 = answer chứa thông tin sai/ngụy tạo.",
                f"Context:\n{context_str}\n\nAnswer: {a}\n\nFaithfulness score (0.0-1.0):",
                max_tokens=10,
            )
        else:
            f_score = 0.0

        # 2) Answer relevancy: LLM judge
        if has_llm and a and q:
            ar_score = _llm_judge_score(
                "Bạn là giám khảo. Chấm điểm 0.0-1.0 cho mức độ liên quan của answer với câu hỏi. Trả lời chỉ bằng MỘT con số. 1.0 = trả lời trực tiếp đúng focus câu hỏi, 0.0 = lệch chủ đề hoàn toàn.",
                f"Câu hỏi: {q}\n\nAnswer: {a}\n\nAnswer relevancy score (0.0-1.0):",
                max_tokens=10,
            )
        else:
            ar_score = 0.0

        # 3) Context precision: heuristic
        cp_score = _compute_context_precision(ctxs, gt)

        # 4) Context recall: heuristic
        cr_score = _compute_context_recall(ctxs, gt)

        sum_f += f_score
        sum_ar += ar_score
        sum_cp += cp_score
        sum_cr += cr_score

        per_question.append(EvalResult(
            question=q,
            answer=a,
            contexts=ctxs,
            ground_truth=gt,
            faithfulness=f_score,
            answer_relevancy=ar_score,
            context_precision=cp_score,
            context_recall=cr_score,
        ))

        if (i + 1) % 5 == 0 or (i + 1) == len(questions):
            print(f"    [{i+1}/{len(questions)}] done", flush=True)

    n = len(questions)
    return {
        "faithfulness": sum_f / n if n else 0.0,
        "answer_relevancy": sum_ar / n if n else 0.0,
        "context_precision": sum_cp / n if n else 0.0,
        "context_recall": sum_cr / n if n else 0.0,
        "per_question": per_question,
    }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Phân tích bottom-N câu hỏi tệ nhất dùng Diagnostic Tree.

    Với mỗi câu hỏi, xác định metric tệ nhất → tra diagnostic tree → đề xuất fix.
    """
    if not eval_results:
        return []

    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating — answer chứa thông tin ngoài context",
                         "Tighten prompt: chỉ trả lời dựa trên context. Lower temperature (0.0). Có guardrail 'Nếu không có → nói Không tìm thấy.'"),
        "context_recall": ("Missing relevant chunks — relevant context không được retrieve",
                           "Cải thiện chunking (parent-child), bật BM25 cho exact-match Vietnamese, hoặc tăng top_k trước rerank."),
        "context_precision": ("Too many irrelevant chunks trong context",
                              "Bật reranker (cross-encoder) hoặc thêm metadata filter để loại chunks không liên quan."),
        "answer_relevancy": ("Answer không match câu hỏi — lệch chủ đề hoặc quá ngắn/dài",
                             "Cải thiện prompt template, ép LLM trả lời đúng focus câu hỏi."),
    }
    metric_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    # Tính avg và worst metric cho mỗi câu
    scored = []
    for r in eval_results:
        scores = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        avg = sum(scores.values()) / len(scores)
        worst = min(scores, key=scores.get)
        scored.append((avg, worst, scores[worst], r))

    # Sort ascending (tệ nhất trước)
    scored.sort(key=lambda x: x[0])

    failures = []
    for avg, worst, worst_score, r in scored[:bottom_n]:
        diagnosis, fix = diagnostic_tree[worst]
        failures.append({
            "question": r.question,
            "ground_truth": r.ground_truth,
            "answer": r.answer,
            "avg_score": round(avg, 4),
            "worst_metric": worst,
            "worst_score": round(worst_score, 4),
            "scores": {k: round(getattr(r, k), 4) for k in metric_keys},
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })
    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON."""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
