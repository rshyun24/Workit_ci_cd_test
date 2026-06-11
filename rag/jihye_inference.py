import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 경로 설정
BASE_MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
ADAPTER_PATH = "../data/jihye_models/jihye_model_output"

SYSTEM_PROMPT = "당신은 공공 SW 계약서의 위험 조항을 탐지하는 전문가입니다. 주어진 계약 조항과 참고 기준을 바탕으로 위험 여부를 판단하고 근거를 제시하십시오."

# 모델 로드
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return model, tokenizer

# 단일 조항 판정
def predict(clause_text: str, law_refs: list, model, tokenizer) -> str:
    # 참고기준 구성
    ref_text = "\n".join([
        f"{r['source_full']}: {r['chunk_text'][:200]}"
        for r in law_refs[:3]
    ])

    user_content = f"다음 계약 조항의 위험 여부를 판단하세요.\n\n[계약조항]\n{clause_text}\n\n[참고기준]\n{ref_text}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=256,
            do_sample=False,
        )

    generated = output[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

# RAG 결과 JSON → 판정 결과
def run_inference(rag_output_path: str, result_path: str = "workit_result.json"):
    with open(rag_output_path, "r", encoding="utf-8") as f:
        rag_results = json.load(f)

    print("모델 로드 중...")
    model, tokenizer = load_model()

    final_results = []
    for item in rag_results:
        clause_number = item["clause_number"]
        clause_text = item["clause_text"]
        law_refs = item["law_refs"]

        if not law_refs:
            continue

        print(f"판정 중: {clause_number}")
        prediction = predict(clause_text, law_refs, model, tokenizer)

        final_results.append({
            "clause_number": clause_number,
            "clause_text": clause_text,
            "risk_names": item["risk_names"],
            "prediction": prediction,
        })

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print(f"완료: {result_path}")
    return final_results


if __name__ == "__main__":
    run_inference("contract_review_output.json")