import os
import json
import uuid
import httpx
import base64
import logging
from fastapi import UploadFile, HTTPException
from PyPDF2 import PdfReader
from io import BytesIO
from typing import List, Dict

logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self, config_manager):
        self.config_manager = config_manager

    async def process_pdf(self, file: UploadFile) -> Dict:
        """
        处理上传的PDF文件：
        1. 读取PDF内容
        2. 调用OCR API获取所有页的识别结果
        3. 合并PyPDF2提取的文本（如果有）和OCR结果
        """
        contents = await file.read()
        pdf_reader = PdfReader(BytesIO(contents))
        num_pages = len(pdf_reader.pages)
        if num_pages > 100:
            raise HTTPException(status_code=400, detail="PDF页数超过100页限制")
        if len(contents) > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="PDF文件大小超过50MB限制")

        # 调用OCR API获取结果
        ocr_results = await self._call_ocr_api(contents)

        pages_text = []
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            # 先尝试用PyPDF2提取文本
            text = page.extract_text()
            if text and text.strip():
                pages_text.append({
                    "page": page_num + 1,
                    "text": text.strip(),
                    "source": "pdf_extract"
                })
            else:
                # 使用OCR结果（按顺序对应页码）
                if page_num < len(ocr_results):
                    ocr_text = ocr_results[page_num].get("prunedResult", "")
                    if ocr_text:
                        pages_text.append({
                            "page": page_num + 1,
                            "text": ocr_text,
                            "source": "ocr"
                        })
                    else:
                        pages_text.append({
                            "page": page_num + 1,
                            "text": "[OCR结果为空]",
                            "source": "ocr_empty"
                        })
                else:
                    pages_text.append({
                        "page": page_num + 1,
                        "text": "[OCR结果缺失]",
                        "source": "ocr_missing"
                    })

        task_id = str(uuid.uuid4())
        result_data = {
            "task_id": task_id,
            "filename": file.filename,
            "pages": pages_text,
            "total_pages": num_pages
        }

        save_dir = self.config_manager.get("ocr_results_dir")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{task_id}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        return result_data

    async def _call_ocr_api(self, pdf_bytes: bytes) -> List[Dict]:
        """
        调用PPOCR V5 API，传入PDF二进制数据，返回每页的OCR结果列表
        """
        api_url = self.config_manager.get("ocr_api_url")
        token = self.config_manager.get("ocr_token")

        # 将PDF文件编码为base64
        file_base64 = base64.b64encode(pdf_bytes).decode("ascii")

        headers = {
            "Authorization": f"token {token}",
            "Content-Type": "application/json"
        }

        payload = {
            "file": file_base64,
            "fileType": 0,  # 0 表示 PDF 文件
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useTextlineOrientation": False,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(api_url, json=payload, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    # 根据示例，返回格式为 {"result": {"ocrResults": [...]}}
                    ocr_results = data.get("result", {}).get("ocrResults", [])
                    return ocr_results
                else:
                    logger.error(f"OCR API 调用失败，状态码: {response.status_code}")
                    # 返回空列表，后续会用 [OCR识别失败] 填充
                    return []
            except Exception as e:
                logger.error(f"OCR API 调用异常: {str(e)}")
                return []

    def get_result(self, task_id: str) -> Dict:
        """获取OCR识别结果"""
        save_dir = self.config_manager.get("ocr_results_dir")
        path = os.path.join(save_dir, f"{task_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_confirmed(self, task_id: str, confirmed_pages: List[Dict]):
        """保存用户确认后的识别结果"""
        save_dir = self.config_manager.get("ocr_results_dir")
        path = os.path.join(save_dir, f"{task_id}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for cp in confirmed_pages:
                for p in data["pages"]:
                    if p["page"] == cp["page"]:
                        p["text"] = cp["text"]
                        p["confirmed"] = True
                        break
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return data
        return None