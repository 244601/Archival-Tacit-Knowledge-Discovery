import json
import os
from typing import Optional
import sys
def get_app_dir():
    """返回可读写文件的目录（exe 所在目录）"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

def get_data_dir():
    """返回只读数据文件的目录（打包后为临时目录）"""
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    else:
        return os.path.dirname(os.path.abspath(__file__))

# 保留原有 BASE_DIR 兼容性，但建议改为使用 get_app_dir
BASE_DIR = get_app_dir()
DEFAULT_CONFIG = {
    "api_key": "sk-170f1621c51d41e0b05f17cba3d7b812",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "intent_model": "qwen3-4b-instruct-2507-50f9a9b8d4d7",
    "skill_model": "MiniMax-M2.1",
    "tagging_model": "qwen3-4b-instruct-2507-ft-202602062020-3b4f-d962213e",
    "rewriting_model": "qwen3-4b-instruct-2507-c86b617576dc",
    "ocr_api_url": "https://letbm9tfi0vdtco6.aistudio-app.com/ocr",
    "ocr_token": "660b1e9fac8162960c377486cb45b9906db8b1c9",
    "corpus_path": os.path.join(BASE_DIR, "语段库.xlsx"),
    "intent_train_path": os.path.join(BASE_DIR, "意图识别微调数据.xlsx"),
    "output_dir": os.path.join(BASE_DIR, "generated_excels"),
    "ocr_results_dir": os.path.join(BASE_DIR, "ocr_results"),
    "splitted_corpus_dir": os.path.join(BASE_DIR, "splitted_corpus"),
    "poppler_path": ""
}

CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

class ConfigManager:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                self._config = json.load(f)
        else:
            self._config = DEFAULT_CONFIG.copy()
            self._save_config()
        # 确保目录存在
        os.makedirs(self._config["output_dir"], exist_ok=True)
        os.makedirs(self._config["ocr_results_dir"], exist_ok=True)
        os.makedirs(self._config["splitted_corpus_dir"], exist_ok=True)

    def _save_config(self):
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=4, ensure_ascii=False)

    def get(self, key: str):
        return self._config.get(key)

    def set(self, key: str, value):
        self._config[key] = value
        self._save_config()

    def get_all(self):
        return self._config.copy()

    def update(self, new_config: dict):
        self._config.update(new_config)
        self._save_config()