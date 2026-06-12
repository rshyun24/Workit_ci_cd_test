import json
import os
from django.http import HttpResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from .models import Contract, ContractDocument, AIReviewResult


@login_required
def contract_list(request):
    contracts = Contract.objects.filter(created_by=request.user).order_by('-created_at')
    return render(request, 'contracts/contract_list.html', {'contracts': contracts})


@login_required
def contract_create(request):
    if request.method == 'POST':
        contract = Contract.objects.create(
            project_name=request.POST.get('project_name'),
            company_name=request.POST.get('company_name'),
            issuing_org=request.POST.get('issuing_org', ''),
            budget=request.POST.get('budget', ''),
            contact_person=request.POST.get('contact_person', ''),
            created_by=request.user,
            status='reviewing',
        )
        # Handle file uploads
        doc_fields = [
            ('requirements_doc', 'requirements'),
            ('rfp_doc', 'rfp'),
            ('contract_doc', 'contract'),
        ]
        for field_name, doc_type in doc_fields:
            f = request.FILES.get(field_name)
            if f:
                doc = ContractDocument.objects.create(
                    contract=contract,
                    doc_type=doc_type,
                    file=f,
                    original_filename=f.name,
                )
        return JsonResponse({'status': 'ok', 'id': contract.id, 'name': contract.project_name})
    return JsonResponse({'status': 'error'}, status=400)


@login_required
def contract_detail_api(request, pk):
    contract = get_object_or_404(Contract, pk=pk, created_by=request.user)
    docs = []
    for doc in contract.documents.all():
        docs.append({
            'id': doc.id,
            'doc_type': doc.doc_type,
            'doc_type_display': doc.get_doc_type_display(),
            'filename': doc.filename(),
            'review_status': doc.review_status,
            'url': doc.file.url,
        })
    return JsonResponse({
        'id': contract.id,
        'project_name': contract.project_name,
        'company_name': contract.company_name,
        'issuing_org': contract.issuing_org,
        'budget': contract.budget,
        'contact_person': contract.contact_person,
        'status': contract.status,
        'status_display': contract.get_status_display(),
        'created_at': contract.created_at.strftime('%Y-%m-%d'),
        'documents': docs,
    })


@login_required
def document_analyze(request, doc_id):
    doc = get_object_or_404(ContractDocument, pk=doc_id, contract__created_by=request.user)
    try:
        result = doc.review_result
    except AIReviewResult.DoesNotExist:
        result = None
    return render(request, 'contracts/document_analyze.html', {
        'doc': doc,
        'contract': doc.contract,
        'result': result,
    })


# @login_required
# @require_POST
# def document_ai_analyze(request, doc_id):
#     """RAG + sLLM(EXAONE Fine-tuned) 기반 계약서 AI 분석"""
#     import sys
#     import os
#     import traceback

#     doc = get_object_or_404(ContractDocument, pk=doc_id, contract__created_by=request.user)

#     # ── rag/, data/ 디렉토리를 import 경로에 추가 ──
#     BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     for extra_path in [
#         os.path.join(BASE_DIR, 'rag'),
#         os.path.join(BASE_DIR, 'data'),
#     ]:
#         if extra_path not in sys.path:
#             sys.path.insert(0, extra_path)

#     try:
#         from contracts.utils import extract_text, parse_to_workit

#         # ── Step 1. 파일에서 텍스트 추출 ──
#         file_text = extract_text(doc.file.path)
#         if not file_text.strip():
#             return JsonResponse(
#                 {'status': 'error', 'message': '텍스트를 추출할 수 없는 파일 형식입니다.'},
#                 status=400,
#             )

#         # ── Step 2. RAG: 조항 청킹 + Qdrant 법령 검색 ──
#         from sentence_transformers import SentenceTransformer
#         from qdrant_client import QdrantClient
#         from yoonha_contract_rag import review_contract, results_to_json

#         QDRANT_PATH = os.path.join(BASE_DIR, 'vectorstore', 'qdrant_storage')
#         embed_model = SentenceTransformer('BAAI/bge-m3')
#         qdrant_client = QdrantClient(path=QDRANT_PATH)

#         clause_results = review_contract(
#             contract_text=file_text,
#             client=qdrant_client,
#             model=embed_model,
#             risk_only=True,   # 위험 조항 관련 법령만 검색
#         )
#         # contract_review_output.json 과 동일한 구조의 list[dict]
#         rag_results = results_to_json(clause_results)

#         # ── Step 3. sLLM(EXAONE Fine-tuned) 추론 ──
#         from jihye_inference import load_model, predict

#         llm_model, tokenizer = load_model()

#         inference_results = []
#         for item in rag_results:
#             if not item.get('law_refs'):
#                 continue

#             print(f"[{rag_results.index(item)+1}/{len(rag_results)}] 판정 중: {item['clause_number']}", flush=True)

#             prediction = predict(
#                 clause_text=item['clause_text'],
#                 law_refs=item['law_refs'],
#                 model=llm_model,
#                 tokenizer=tokenizer,
#             )
#             inference_results.append({
#                 'clause_number': item['clause_number'],
#                 'clause_text':   item['clause_text'],
#                 'risk_names':    item.get('risk_names', []),
#                 'prediction':    prediction,
#             })

#         # ── Step 4. Workit 화면 형식으로 변환 ──
#         parsed = parse_to_workit(inference_results)

#         # ── Step 5. DB 저장 ──
#         AIReviewResult.objects.update_or_create(
#             document=doc,
#             defaults={
#                 'blanks':       parsed['blanks'],
#                 'typos':        parsed['typos'],
#                 'legal_issues': parsed['legal_issues'],
#             },
#         )

#         total = (
#             len(parsed['blanks'])
#             + len(parsed['typos'])
#             + len(parsed['legal_issues'])
#         )
#         return JsonResponse({
#             'status':       'ok',
#             'total':        total,
#             'blanks':       parsed['blanks'],
#             'typos':        parsed['typos'],
#             'legal_issues': parsed['legal_issues'],
#         })

#     except Exception as e:
#         traceback.print_exc()
#         return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

@login_required
@require_POST
def document_ai_analyze(request, doc_id):
    """AI 분석 태스크 시작"""
    doc = get_object_or_404(ContractDocument, pk=doc_id, contract__created_by=request.user)
    from contracts.tasks import analyze_document_task
    task = analyze_document_task.delay(doc_id)
    return JsonResponse({'status': 'started', 'task_id': task.id})


@login_required
def document_ai_status(request, task_id):
    """태스크 진행 상태 조회"""
    from celery.result import AsyncResult
    result = AsyncResult(task_id)

    if result.state == 'PENDING':
        return JsonResponse({'state': 'pending', 'current': 0, 'total': 1})

    elif result.state == 'PROGRESS':
        meta = result.info or {}
        return JsonResponse({
            'state': 'progress',
            'current': meta.get('current', 0),
            'total': meta.get('total', 1),
        })

    elif result.state == 'SUCCESS':
        data = result.result or {}
        return JsonResponse({'state': 'success', **data})

    else:
        return JsonResponse({'state': 'error', 'message': str(result.info)})


@login_required
@require_POST
def document_complete_review(request, doc_id):
    doc = get_object_or_404(ContractDocument, pk=doc_id, contract__created_by=request.user)
    doc.review_status = 'reviewed'
    doc.save()
    contract = doc.contract
    contract.status = 'in_progress'
    contract.save()
    return JsonResponse({'status': 'ok', 'redirect': '/performance/'})


@login_required
@require_POST
def contract_update_file(request, pk):
    contract = get_object_or_404(Contract, pk=pk, created_by=request.user)
    doc_type = request.POST.get('doc_type')
    f = request.FILES.get('file')
    if not f or not doc_type:
        return JsonResponse({'status': 'error', 'message': '파일 또는 문서 유형이 없습니다.'}, status=400)

    existing = contract.documents.filter(doc_type=doc_type).first()
    if existing:
        existing.file = f
        existing.original_filename = f.name
        existing.review_status = 'pending'
        existing.save()
    else:
        ContractDocument.objects.create(
            contract=contract,
            doc_type=doc_type,
            file=f,
            original_filename=f.name,
        )
    return JsonResponse({'status': 'ok', 'filename': f.name})

@login_required
def document_page_image(request, doc_id, page):
    doc = get_object_or_404(ContractDocument, pk=doc_id, contract__created_by=request.user)
    
    try:
        from pdf2image import convert_from_path
        import io, shutil, tempfile

        poppler_path = r"C:\poppler-24.08.0\Library\bin"

        # 한글 경로 문제 해결 - 임시 파일로 복사
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp_path = tmp.name
            # print(f"tmp_path: {tmp_path}")
            shutil.copy2(doc.file.path, tmp_path)

        images = convert_from_path(
            tmp_path,
            dpi=150,
            first_page=page,
            last_page=page,
            poppler_path=poppler_path,
        )

        os.unlink(tmp_path)  # 임시 파일 삭제

        if not images:
            return HttpResponse(status=404)

        buf = io.BytesIO()
        images[0].save(buf, format='PNG')
        buf.seek(0)
        return HttpResponse(buf.read(), content_type='image/png')

    except Exception as e:
        import traceback
        return HttpResponse(traceback.format_exc(), content_type='text/plain', status=500)

@login_required  
def document_page_count(request, doc_id):
    """PDF 총 페이지 수 반환"""
    doc = get_object_or_404(ContractDocument, pk=doc_id, contract__created_by=request.user)
    
    try:
        from pdf2image import pdfinfo_from_path
        import io
        
        poppler_path = r"C:\poppler-24.08.0\Library\bin"

        info = pdfinfo_from_path(
            doc.file.path,
            poppler_path=poppler_path if os.name == 'nt' else None,
        )
        return JsonResponse({'pages': info['Pages']})
    
    except Exception as e:
        return JsonResponse({'pages': 1})