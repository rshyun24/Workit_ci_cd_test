# pip install pdfplumber

"""
Workit - 계약서 PDF 파싱 스크립트
input : PDF 파일 경로
output: 조항 단위 텍스트 (yoonha_contract_rag.py의 chunk_contract() 입력용)
"""

import re
import pdfplumber
from pathlib import Path


def parse_pdf(pdf_path: str) -> str:
    """PDF에서 텍스트 추출"""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def extract_contract_text(pdf_path: str) -> str:
    """
    계약서 PDF에서 조항 텍스트 추출
    yoonha_contract_rag.py의 review_contract() 입력으로 사용
    """
    text = parse_pdf(pdf_path)

    # 불필요한 공백/줄바꿈 정리
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법: python jihye_contract_parser.py <계약서.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    contract_text = extract_contract_text(pdf_path)
    print(contract_text)