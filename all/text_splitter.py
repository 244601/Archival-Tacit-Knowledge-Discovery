import os
import re
import json
import pandas as pd
from typing import List, Dict


class TextSplitter:
    def __init__(self, config_manager):
        self.config_manager = config_manager

    def split_text(self, text: str) -> List[Dict]:
        """
        将纯文本切分为语段：三句话组成一个文段
        句子以。！？；;!?结尾，保留标点。
        """
        # 使用正则分割句子，保留分隔符
        # 匹配中文和英文标点
        pattern = r'(?<=[。！？；;!?])'
        sentences = re.split(pattern, text)
        # 过滤空字符串
        sentences = [s.strip() for s in sentences if s.strip()]

        # 如果 sentences 为空，返回空列表
        if not sentences:
            return []

        chunks = []
        for i in range(0, len(sentences), 3):
            chunk_sentences = sentences[i:i + 3]
            chunk_text = "".join(chunk_sentences)
            # 确保文段以标点结尾（通常已经是）
            chunk = {
                "文段": chunk_text,
                "句子1": chunk_sentences[0] if len(chunk_sentences) > 0 else "",
                "句子2": chunk_sentences[1] if len(chunk_sentences) > 1 else "",
                "句子3": chunk_sentences[2] if len(chunk_sentences) > 2 else "",
                "来源": "用户上传文本"
            }
            chunks.append(chunk)
        return chunks

    def split_from_ocr_result(self, task_id: str) -> List[Dict]:
        ocr_dir = self.config_manager.get("ocr_results_dir")
        path = os.path.join(ocr_dir, f"{task_id}.json")
        if not os.path.exists(path):
            raise FileNotFoundError("OCR结果文件不存在")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        full_text = "\n".join([p["text"] for p in data["pages"]])
        chunks = self.split_text(full_text)
        for chunk in chunks:
            chunk["来源"] = f"OCR识别-{data['filename']}"
        return chunks

    def build_corpus(self, chunks: List[Dict], filename: str) -> str:
        df = pd.DataFrame(chunks)
        corpus_df = pd.DataFrame({
            "部队名称": "",
            "年月日": "",
            "原始文本": df["文段"],
            "语段标签库": "",
            "语段讲解库": ""
        })
        save_dir = self.config_manager.get("splitted_corpus_dir")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        corpus_df.to_excel(save_path, index=False, engine='openpyxl')
        return save_path