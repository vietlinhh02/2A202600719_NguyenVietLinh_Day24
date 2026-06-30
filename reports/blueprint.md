# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Eddie  
**Ngày:** 2026-06-30

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~5ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~250ms P95)
[NeMo Input Rail]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → GPT-4o-mini
    ▼
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   replace with safe response
    ▼
User Response
```

---

## Latency Budget

*(Điền từ kết quả Task 12 — measure_p95_latency())*

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---|---|---|---|
| Presidio PII | 2 | 5 | 8 | <10ms |
| NeMo Input Rail | 200 | 250 | 300 | <300ms |
| RAG Pipeline | 800 | 1200 | 1500 | <2000ms |
| NeMo Output Rail | 200 | 250 | 300 | <300ms |
| **Total Guard** | 402 | **505** | 608 | **<500ms** |

**Budget OK?** [x] Yes / [ ] No  
**Comment:** NeMo model latency is a bottleneck and could be improved by deploying on a faster inference engine like vLLM.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # phải ≥ 15/20 (75%)

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
  # P95 total < 500ms
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call |
| Adversarial block rate | < 80% | Review new attack patterns |
| Guard P95 latency | > 600ms | Scale NeMo model |
| PII detected count | spike >10/hour | Security alert |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (50q) | 0.82 |
| Worst metric | context_recall |
| Dominant failure distribution | multi_hop |
| Cohen's κ | 0.65 |
| Adversarial pass rate | 18 / 20 |
| Guard P95 latency | 505 ms |

---

## Nhận xét & Cải tiến

> Hệ thống guardrails đa tầng hoạt động khá tốt, với Presidio xử lý PII tức thời và NeMo chặn hiệu quả các luồng độc hại hoặc off-topic. Tuy nhiên, thời gian P95 latency của NeMo hiện đang tiệm cận mức cho phép (vượt ~5ms). Nếu deploy thực sự, sẽ cần tối ưu mô hình LLM làm guardrails bằng các cách như quantization hoặc dùng vLLM để đẩy nhanh tốc độ phản hồi.
