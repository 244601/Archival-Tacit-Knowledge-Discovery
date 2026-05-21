import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from retrieval_system import CombatLogRetrievalSystem
from config_manager import ConfigManager
from ocr_handler import OCRProcessor
from text_splitter import TextSplitter
from tagging_processor import TaggingProcessor
from rewriting_processor import RewritingProcessor
from visualization import build_network, export_gephi
import numpy as np
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import threading
import webbrowser

app = FastAPI(title="档案信息智能检索系统 API")

# 配置管理器
config_manager = ConfigManager()

# 检索系统实例
retrieval_system = CombatLogRetrievalSystem(config_manager)

# OCR处理器
ocr_processor = OCRProcessor(config_manager)

# 文本切分器
text_splitter = TextSplitter(config_manager)

# 标签化处理器
tagging_processor = TaggingProcessor(config_manager)

# 转写处理器
rewriting_processor = RewritingProcessor(config_manager)

def convert_numpy(obj):
    """递归转换 numpy 类型为 Python 原生类型"""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    elif isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    else:
        return obj

# -------------------- 请求/响应模型 (保持不变) --------------------
class IntentRequest(BaseModel):
    query: str

class ParsedData(BaseModel):
    keywords: List[str] = []
    labels: List[str] = []
    keyword_and_groups: List[List[str]] = []
    keyword_or_groups: List[List[str]] = []
    label_and_groups: List[List[str]] = []
    label_or_groups: List[List[str]] = []
    summary: str = ""
    keywords_cleaned: List[str] = []
    labels_cleaned: List[str] = []

class LinearRequest(BaseModel):
    parsed_data: ParsedData
    query: str

class ModularRequest(BaseModel):
    parsed_data: ParsedData
    query: str
    modules: List[str]

class IntentResponse(BaseModel):
    keywords: List[str]
    labels: List[str]
    summary: str
    keywords_cleaned: List[str]
    labels_cleaned: List[str]
    keyword_and_groups: List[List[str]] = []
    keyword_or_groups: List[List[str]] = []

class RetrievalResult(BaseModel):
    results: List[Dict[str, Any]]
    filename: str
    count: int

class LinearResponse(BaseModel):
    results: Optional[List[Dict[str, Any]]] = None
    filename: Optional[str] = None
    count: int = 0
    need_full_scan: bool = False

class FullVectorResponse(BaseModel):
    results: List[Dict[str, Any]]
    filename: str
    count: int

class ModularResponse(BaseModel):
    keyword: Optional[RetrievalResult] = None
    skills: Optional[RetrievalResult] = None
    vector: Optional[RetrievalResult] = None

# -------------------- 检索API (保持不变) --------------------
@app.post("/api/intent/analyze", response_model=IntentResponse)
async def analyze_intent(req: IntentRequest):
    raw = retrieval_system.get_intent_analysis(req.query)
    if not raw:
        raise HTTPException(status_code=500, detail="意图分析失败")
    parsed = retrieval_system.parse_intent(raw)
    cleaned = retrieval_system.clean_keywords_labels(parsed, req.query)
    return IntentResponse(
        keywords=parsed["keywords"],
        labels=parsed["labels"],
        summary=parsed["summary"],
        keywords_cleaned=cleaned["keywords_cleaned"],
        labels_cleaned=cleaned["labels_cleaned"],
        keyword_and_groups=cleaned.get("keyword_and_groups", []),
        keyword_or_groups=cleaned.get("keyword_or_groups", []),
    )

@app.post("/api/retrieve/linear")
async def linear_retrieval(req: LinearRequest):
    parsed_dict = req.parsed_data.dict()
    parsed_dict["keyword_and_groups"] = parsed_dict.get("keyword_and_groups", [])
    parsed_dict["keyword_or_groups"] = parsed_dict.get("keyword_or_groups", [])
    parsed_dict["label_and_groups"] = parsed_dict.get("label_and_groups", [])
    parsed_dict["label_or_groups"] = parsed_dict.get("label_or_groups", [])
    initial = retrieval_system.stage1_retrieval(parsed_dict)
    N = len(initial)
    if N >= 100:
        final = sorted(initial, key=lambda x: x["confidence"], reverse=True)
        for item in final:
            item.pop("_row_idx", None)
        filename = "result_path1.xlsx"
        retrieval_system.save_results(final, filename)
        return LinearResponse(
            results=final,
            filename=filename,
            count=len(final),
            need_full_scan=False,
        )
    else:
        merged = retrieval_system.path2_agent_skills(initial, parsed_dict, req.query)
        final_N = len(merged)
        if final_N >= 100:
            for item in merged:
                item.pop("_row_idx", None)
            filename = "result_path2.xlsx"
            retrieval_system.save_results(merged, filename)
            return LinearResponse(
                results=merged,
                filename=filename,
                count=final_N,
                need_full_scan=False,
            )
        else:
            temp_save = [{k: v for k, v in item.items() if k != "_row_idx"} for item in merged]
            temp_filename = "result_partial.xlsx"
            retrieval_system.save_results(temp_save, temp_filename)
            return LinearResponse(
                count=final_N,
                need_full_scan=True,
            )

@app.post("/api/retrieve/full-vector")
async def full_vector_retrieval(req: LinearRequest):
    parsed_dict = req.parsed_data.dict()
    parsed_dict["keyword_and_groups"] = parsed_dict.get("keyword_and_groups", [])
    parsed_dict["keyword_or_groups"] = parsed_dict.get("keyword_or_groups", [])
    parsed_dict["label_and_groups"] = parsed_dict.get("label_and_groups", [])
    parsed_dict["label_or_groups"] = parsed_dict.get("label_or_groups", [])
    results = retrieval_system.path3_full_vector_search(parsed_dict, req.query)
    results = retrieval_system._rerank(req.query, results, top_k=50)
    for item in results:
        item.pop("_row_idx", None)
    # 转换 numpy 类型
    results = convert_numpy(results)
    filename = "result_full_scan.xlsx"
    retrieval_system.save_results(results, filename, confidence_col_name="语义相似度")
    return FullVectorResponse(
        results=results,
        filename=filename,
        count=len(results),
    )

@app.post("/api/retrieve/modular")
async def modular_retrieval(req: ModularRequest):
    parsed_dict = req.parsed_data.dict()
    parsed_dict["keyword_and_groups"] = parsed_dict.get("keyword_and_groups", [])
    parsed_dict["keyword_or_groups"] = parsed_dict.get("keyword_or_groups", [])
    parsed_dict["label_and_groups"] = parsed_dict.get("label_and_groups", [])
    parsed_dict["label_or_groups"] = parsed_dict.get("label_or_groups", [])
    if req.modules == ["full"]:
        kw = retrieval_system.keyword_retrieval(parsed_dict)
        retrieval_system.save_results(kw, "full_keyword_result.xlsx")
        sk = retrieval_system.skills_retrieval(parsed_dict, req.query)
        retrieval_system.save_results(sk, "full_skills_result.xlsx")
        vec = retrieval_system.vector_retrieval(parsed_dict, req.query)
        retrieval_system.save_results(vec, "full_vector_result.xlsx", confidence_col_name="语义相似度")
        response = {
            "keyword": {"results": kw, "filename": "full_keyword_result.xlsx", "count": len(kw)},
            "skills": {"results": sk, "filename": "full_skills_result.xlsx", "count": len(sk)},
            "vector": {"results": vec, "filename": "full_vector_result.xlsx", "count": len(vec)},
        }
    else:
        response = {}
        for mod in req.modules:
            if mod == "keyword":
                res = retrieval_system.keyword_retrieval(parsed_dict)
                fname = "modular_keyword_result.xlsx"
                retrieval_system.save_results(res, fname)
            elif mod == "skills":
                res = retrieval_system.skills_retrieval(parsed_dict, req.query)
                fname = "modular_skills_result.xlsx"
                retrieval_system.save_results(res, fname)
            elif mod == "vector":
                res = retrieval_system.vector_retrieval(parsed_dict, req.query)
                fname = "modular_vector_result.xlsx"
                retrieval_system.save_results(res, fname, confidence_col_name="语义相似度")
            else:
                continue
            for item in res:
                item.pop("_row_idx", None)
            response[mod] = {"results": res, "filename": fname, "count": len(res)}
    response = convert_numpy(response)
    return response

@app.get("/api/download")
async def download_file(file: str):
    output_dir = config_manager.get("output_dir")
    safe_filename = os.path.basename(file)
    filepath = os.path.join(output_dir, safe_filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(
        path=filepath,
        filename=safe_filename,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# -------------------- 配置管理API --------------------
class ConfigUpdate(BaseModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    corpus_path: Optional[str] = None
    intent_train_path: Optional[str] = None
    output_dir: Optional[str] = None
    ocr_api_url: Optional[str] = None
    ocr_token: Optional[str] = None
    intent_model: Optional[str] = None
    skill_model: Optional[str] = None
    tagging_model: Optional[str] = None
    rewriting_model: Optional[str] = None

@app.get("/api/config")
async def get_config():
    return config_manager.get_all()

@app.post("/api/config")
async def update_config(update: ConfigUpdate):
    new_config = update.dict(exclude_unset=True)
    config_manager.update(new_config)
    global retrieval_system
    # 仅当 corpus_path 改变且不为空时才重新加载
    if "corpus_path" in new_config and new_config["corpus_path"] != retrieval_system.corpus_path:
        retrieval_system.reload_corpus(new_config["corpus_path"])
    retrieval_system.update_config(**new_config)
    global ocr_processor
    ocr_processor = OCRProcessor(config_manager)
    global tagging_processor
    tagging_processor = TaggingProcessor(config_manager)
    global rewriting_processor
    rewriting_processor = RewritingProcessor(config_manager)
    return {"message": "配置更新成功"}

# -------------------- PDF批量识别API (保持不变) --------------------
class OcrTaskResponse(BaseModel):
    task_id: str
    filename: str
    total_pages: int

class OcrResultResponse(BaseModel):
    task_id: str
    filename: str
    pages: List[Dict]
    total_pages: int

class ConfirmOcrRequest(BaseModel):
    task_id: str
    confirmed_pages: List[Dict]

@app.post("/api/ocr/upload", response_model=OcrTaskResponse)
async def ocr_upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="仅支持PDF文件")
    result = await ocr_processor.process_pdf(file)
    return OcrTaskResponse(
        task_id=result["task_id"],
        filename=result["filename"],
        total_pages=result["total_pages"]
    )

@app.get("/api/ocr/result/{task_id}", response_model=OcrResultResponse)
async def ocr_result(task_id: str):
    result = ocr_processor.get_result(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return OcrResultResponse(**result)

@app.post("/api/ocr/confirm")
async def ocr_confirm(req: ConfirmOcrRequest):
    updated = ocr_processor.save_confirmed(req.task_id, req.confirmed_pages)
    if updated is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {"message": "确认成功", "task_id": req.task_id}

@app.get("/api/ocr/download/{task_id}")
async def ocr_download(task_id: str):
    result = ocr_processor.get_result(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return JSONResponse(content=result)

# -------------------- 文本切分API (保持不变) --------------------
class SplitTextRequest(BaseModel):
    text: Optional[str] = None
    ocr_task_id: Optional[str] = None
    filename: str = "语段库.xlsx"

@app.post("/api/split")
async def split_text(req: SplitTextRequest):
    if req.text:
        chunks = text_splitter.split_text(req.text)
    elif req.ocr_task_id:
        chunks = text_splitter.split_from_ocr_result(req.ocr_task_id)
    else:
        raise HTTPException(status_code=400, detail="必须提供text或ocr_task_id")
    return {"chunks": chunks, "count": len(chunks)}

@app.post("/api/split/build_corpus")
async def build_corpus(req: SplitTextRequest):
    if req.text:
        chunks = text_splitter.split_text(req.text)
    elif req.ocr_task_id:
        chunks = text_splitter.split_from_ocr_result(req.ocr_task_id)
    else:
        raise HTTPException(status_code=400, detail="必须提供text或ocr_task_id")
    save_path = text_splitter.build_corpus(chunks, req.filename)
    return {"filename": os.path.basename(save_path), "path": save_path, "count": len(chunks)}

@app.get("/api/split/download/{filename}")
async def split_download(filename: str):
    save_dir = config_manager.get("splitted_corpus_dir")
    safe_filename = os.path.basename(filename)
    filepath = os.path.join(save_dir, safe_filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(
        path=filepath,
        filename=safe_filename,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# -------------------- 文本标签化API --------------------
@app.post("/api/tagging/process")
async def tagging_process(
    file: Optional[UploadFile] = File(None),
    ocr_task_id: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    filename: str = Form("标签化结果.xlsx")
):
    if file:
        contents = await file.read()
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        result_path = tagging_processor.process_excel_file(tmp_path, filename)
        os.unlink(tmp_path)
    elif ocr_task_id:
        chunks = text_splitter.split_from_ocr_result(ocr_task_id)
        corpus_path = text_splitter.build_corpus(chunks, "temp_corpus.xlsx")
        result_path = tagging_processor.process_excel_file(corpus_path, filename)
        os.unlink(corpus_path)
    elif text:
        chunks = text_splitter.split_text(text)
        corpus_path = text_splitter.build_corpus(chunks, "temp_corpus.xlsx")
        result_path = tagging_processor.process_excel_file(corpus_path, filename)
        os.unlink(corpus_path)
    else:
        raise HTTPException(status_code=400, detail="必须提供file、ocr_task_id或text")
    return {"filename": os.path.basename(result_path), "path": result_path}

@app.get("/api/tagging/download/{filename}")
async def tagging_download(filename: str):
    output_dir = config_manager.get("output_dir")
    safe_filename = os.path.basename(filename)
    filepath = os.path.join(output_dir, safe_filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(
        path=filepath,
        filename=safe_filename,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# -------------------- 文本转写API --------------------
@app.post("/api/rewriting/process")
async def rewriting_process(
    file: Optional[UploadFile] = File(None),
    ocr_task_id: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    filename: str = Form("转写结果.xlsx")
):
    if file:
        contents = await file.read()
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        result_path = rewriting_processor.process_excel_file(tmp_path, filename)
        os.unlink(tmp_path)
    elif ocr_task_id:
        chunks = text_splitter.split_from_ocr_result(ocr_task_id)
        corpus_path = text_splitter.build_corpus(chunks, "temp_corpus.xlsx")
        result_path = rewriting_processor.process_excel_file(corpus_path, filename)
        os.unlink(corpus_path)
    elif text:
        chunks = text_splitter.split_text(text)
        corpus_path = text_splitter.build_corpus(chunks, "temp_corpus.xlsx")
        result_path = rewriting_processor.process_excel_file(corpus_path, filename)
        os.unlink(corpus_path)
    else:
        raise HTTPException(status_code=400, detail="必须提供file、ocr_task_id或text")
    return {"filename": os.path.basename(result_path), "path": result_path}

@app.get("/api/rewriting/download/{filename}")
async def rewriting_download(filename: str):
    output_dir = config_manager.get("output_dir")
    safe_filename = os.path.basename(filename)
    filepath = os.path.join(output_dir, safe_filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(
        path=filepath,
        filename=safe_filename,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# -------------------- 可视化分析API (保持不变) --------------------
class VisualizationUploadRequest(BaseModel):
    file: UploadFile

class VisualizationResponse(BaseModel):
    nodes: List[Dict]
    edges: List[Dict]
    gephi_file: str

@app.post("/api/visualization/upload", response_model=VisualizationResponse)
async def visualization_upload(file: UploadFile = File(...)):
    if not file.filename.endswith('.xlsx'):
        raise HTTPException(status_code=400, detail="仅支持xlsx文件")
    contents = await file.read()
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    try:
        nodes, edges = build_network(tmp_path)
        nodes = convert_numpy(nodes)
        edges = convert_numpy(edges)
        gephi_path = export_gephi(nodes, edges)
        return VisualizationResponse(
            nodes=nodes,
            edges=edges,
            gephi_file=os.path.basename(gephi_path)
        )
    finally:
        os.unlink(tmp_path)

@app.get("/api/visualization/download/{filename}")
async def visualization_download(filename: str):
    output_dir = config_manager.get("output_dir")
    safe_filename = os.path.basename(filename)
    filepath = os.path.join(output_dir, safe_filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(
        path=filepath,
        filename=safe_filename,
        media_type="application/xml"
    )

# 静态文件挂载
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

def open_browser():
    webbrowser.open("http://localhost:8000")

threading.Timer(1.5, open_browser).start()

from fastapi import Request
from fastapi.responses import JSONResponse

@app.get("/{full_path:path}")
async def catch_all(request: Request, full_path: str):
    # 如果请求路径不是 API 开头，且不是静态文件，则返回 index.html
    if not full_path.startswith("api") and not full_path.startswith("static"):
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))
    # 否则返回 404
    return JSONResponse(status_code=404, content={"detail": "Not Found"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)