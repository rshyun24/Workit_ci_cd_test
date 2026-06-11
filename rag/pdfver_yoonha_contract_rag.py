"""
Workit - 계약서 검토 RAG 파이프라인

흐름:
  계약서 텍스트 입력
      ↓
  조항 단위 청킹
      ↓
  각 조항 → Qdrant 검색
      ↓
  관련 법령 + risk_id/risk_name 반환
      ↓
  JSON 저장
"""

import re
import json
from jihye_contract_parser import extract_contract_text
from dataclasses import dataclass, field, asdict
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer


# ──────────────────────────────────────────
# 0. 설정
# ──────────────────────────────────────────
QDRANT_PATH = "./qdrant_storage"
COLLECTION = "law_kb"
EMBED_MODEL = "BAAI/bge-m3"

TOP_K = 10
MIN_SCORE = 0.40


# ──────────────────────────────────────────
# 1. 데이터 클래스
# ──────────────────────────────────────────
@dataclass
class LawRef:
    """검색된 법령 조문 1개"""
    law_name: str
    article_number: str
    article_title: str
    chunk_text: str
    score: float
    risk_ids: list[str]
    risk_names: list[str]
    source_full: str
    is_risk_ref: bool


@dataclass
class ClauseResult:
    """계약서 조항 1개의 검색 결과"""
    clause_number: str
    clause_text: str
    law_refs: list[LawRef] = field(default_factory=list)
    risk_ids: list[str] = field(default_factory=list)
    risk_names: list[str] = field(default_factory=list)


# ──────────────────────────────────────────
# 2. 계약서 조항 단위 청킹
# ──────────────────────────────────────────
def chunk_contract(text: str) -> list[dict]:
    """
    계약서 텍스트를 조항 단위로 분할.
    제1조, 제10조, 제2조의2 패턴 지원.
    조항 패턴이 없으면 단락 단위로 분할.
    """
    text = text.strip()

    pattern = r"(제\d+조(?:의\d+)?(?:\s*\([^)]*\))?)"
    parts = re.split(pattern, text)

    clauses = []
    i = 1

    while i < len(parts) - 1:
        raw_header = parts[i].strip()
        body = parts[i + 1].strip()

        match = re.match(r"(제\d+조(?:의\d+)?)", raw_header)
        clause_number = match.group(1) if match else raw_header

        clause_text = f"{raw_header} {body}".strip()

        if body:
            clauses.append(
                {
                    "clause_number": clause_number,
                    "clause_text": clause_text,
                }
            )

        i += 2

    if not clauses:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        clauses = [
            {
                "clause_number": f"단락{i + 1}",
                "clause_text": paragraph,
            }
            for i, paragraph in enumerate(paragraphs)
        ]

    return clauses


# ──────────────────────────────────────────
# 3. 단일 조항 → 법령 검색
# ──────────────────────────────────────────
def search_law_for_clause(
    clause_text: str,
    client: QdrantClient,
    model: SentenceTransformer,
    risk_only: bool = False,
    top_k: int = TOP_K,
    min_score: float = MIN_SCORE,
) -> list[LawRef]:
    """
    계약서 조항 텍스트로 법령 KB 검색.
    risk_only=True이면 is_risk_ref=True인 조문만 검색.
    """
    query_vector = model.encode(clause_text).tolist()

    search_filter = None
    if risk_only:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="is_risk_ref",
                    match=MatchValue(value=True),
                )
            ]
        )

    response = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        query_filter=search_filter,
        limit=top_k,
    )

    points = response.points

    law_refs: list[LawRef] = []

    for point in points:
        if point.score is not None and point.score < min_score:
            continue

        payload = point.payload or {}

        law_refs.append(
            LawRef(
                law_name=payload.get("law_name", ""),
                article_number=payload.get("article_number", ""),
                article_title=payload.get("article_title", ""),
                chunk_text=payload.get("chunk_text", ""),
                score=float(point.score or 0.0),
                risk_ids=payload.get("risk_ids", []) or [],
                risk_names=payload.get("risk_names", []) or [],
                source_full=payload.get("source_full", ""),
                is_risk_ref=bool(payload.get("is_risk_ref", False)),
            )
        )

    return law_refs


# ──────────────────────────────────────────
# 4. 전체 계약서 검토
# ──────────────────────────────────────────
def review_contract(
    contract_text: str,
    client: QdrantClient,
    model: SentenceTransformer,
    risk_only: bool = False,
    top_k: int = TOP_K,
    min_score: float = MIN_SCORE,
) -> list[ClauseResult]:
    """
    계약서 전체 텍스트를 받아 조항별 관련 법령 검색 결과 반환.
    """
    clauses = chunk_contract(contract_text)
    results: list[ClauseResult] = []

    print(f"  총 {len(clauses)}개 조항 검색 중...")

    for i, clause in enumerate(clauses, 1):
        print(
            f"  [{i}/{len(clauses)}] {clause['clause_number']} 검색 중...",
            end="\r",
        )

        law_refs = search_law_for_clause(
            clause_text=clause["clause_text"],
            client=client,
            model=model,
            risk_only=risk_only,
            top_k=top_k,
            min_score=min_score,
        )

        risk_ids = list(
            dict.fromkeys(
                risk_id
                for ref in law_refs
                for risk_id in ref.risk_ids
            )
        )

        risk_names = list(
            dict.fromkeys(
                risk_name
                for ref in law_refs
                for risk_name in ref.risk_names
            )
        )

        results.append(
            ClauseResult(
                clause_number=clause["clause_number"],
                clause_text=clause["clause_text"],
                law_refs=law_refs,
                risk_ids=risk_ids,
                risk_names=risk_names,
            )
        )

    print("\n  ✅ 검색 완료")
    return results


# ──────────────────────────────────────────
# 5. 결과 포매터
# ──────────────────────────────────────────
def format_results(results: list[ClauseResult]) -> str:
    """
    검색 결과를 콘솔 확인용 텍스트로 변환.
    """
    lines = []

    for result in results:
        if not result.law_refs:
            continue

        lines.append(f"\n{'═' * 60}")
        lines.append(f"📌 {result.clause_number}")
        preview = result.clause_text[:220]
        lines.append(f"{preview}{'...' if len(result.clause_text) > 220 else ''}")

        if result.risk_names:
            lines.append(
                f"\n⚠️ 탐지된 리스크 후보: {', '.join(result.risk_names)}"
            )

        lines.append(f"\n📚 관련 법령 Top {len(result.law_refs)}")

        for idx, ref in enumerate(result.law_refs, 1):
            risk_tag = ""
            if ref.risk_names:
                risk_tag = f" [{', '.join(ref.risk_names)}]"

            lines.append(
                f"  {idx}. [{ref.score:.3f}] {ref.source_full}{risk_tag}"
            )
            lines.append(f"     {ref.chunk_text[:150]}...")

    if not lines:
        return "관련 법령이 검색된 조항이 없습니다."

    return "\n".join(lines)


def results_to_json(results: list[ClauseResult]) -> list[dict]:
    """
    EXAONE 또는 후속 LLM 입력용 JSON 변환.
    """
    return [asdict(result) for result in results]


# ──────────────────────────────────────────
# 6. 메인
# ──────────────────────────────────────────
def main(pdf_path: str = "계약서.pdf") -> None:
    print("=" * 60)
    print("Workit 계약서 검토 RAG 파이프라인")
    print("=" * 60)

    print(f"\n📦 모델 로드: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    client = QdrantClient(path=QDRANT_PATH)

    try:
        count = client.count(collection_name=COLLECTION)
    except Exception as exc:
        raise RuntimeError(
            f"Qdrant 컬렉션을 찾을 수 없습니다: {COLLECTION}."
        ) from exc

    print(f"📚 법령 KB: {count.count}개 청크 로드됨")

    contract_text = extract_contract_text(pdf_path)

    print("\n🔍 계약서 검토 시작...")
    results = review_contract(
        contract_text=contract_text,
        client=client,
        model=model,
        risk_only=True,
        top_k=TOP_K,
        min_score=MIN_SCORE,
    )

    print(format_results(results))

    output_path = Path("contract_review_output.json")
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results_to_json(results), file, ensure_ascii=False, indent=2)

    print(f"\n💾 JSON 저장: {output_path}")


if __name__ == "__main__":
    import sys
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "계약서.pdf"
    main(pdf_path)