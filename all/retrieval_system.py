import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import re
import httpx
import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings
import json
from typing import List, Dict, Optional
from config_manager import get_data_dir

warnings.filterwarnings('ignore')

class CombatLogRetrievalSystem:
    """
    基于Skills的知识体系智能检索系统（支持线性/模块双模式）
    手动输入支持 * 语法，线性模式全量匹配增加语义相似度列
    适配动态配置管理
    """

    def __init__(self, config_manager=None):
        """
        初始化系统，从config_manager读取配置
        """
        self.config_manager = config_manager
        self.corpus_path = config_manager.get("corpus_path")
        self.intent_train_path = config_manager.get("intent_train_path")
        self.client = OpenAI(
            api_key=config_manager.get("api_key"),
            base_url=config_manager.get("base_url")
        )
        self.intent_model = config_manager.get("intent_model")
        self.skill_model = config_manager.get("skill_model")

        self.df = None
        self.embedding_model = None
        self.tfidf_vectorizer = None
        self.tfidf_feature_names = None
        self.tfidf_idf = None
        self.manual_mode = False

        self._load_corpus()
        self._prepare_tfidf()

    def _load_corpus(self):
        """加载语段库，若路径不存在则生成测试数据"""
        try:
            print(f"正在加载语段库: {self.corpus_path}")
            self.df = pd.read_excel(self.corpus_path)
            expected_cols = ["部队名称", "年月日", "原始文本", "语段标签库", "语段讲解库"]
            if list(self.df.columns) != expected_cols:
                self.df.columns = expected_cols[:len(self.df.columns)]
            self.df = self.df.fillna("")
            print(f"语段库加载完成，共 {len(self.df)} 条记录。")
        except Exception as e:
            print(f"错误：无法加载语段库Excel文件。{e}")
            # 生成简易测试数据
            test_data = {
                "部队名称": ["第四野战军", "第三野战军", "华东野战军"] * 10,
                "年月日": ["1949-04-21", "1948-07-15", "1948-11-02"] * 10,
                "原始文本": ["第四野战军在渡江战役中实施了战术迂回", "敌军指挥官出现决策失误", "兵力部署集中在沿江地带"] * 10,
                "语段标签库": ["渡江战役 战术", "指挥官 决策失误", "兵力部署"] * 10,
                "语段讲解库": ["四野渡江作战记录", "敌军指挥分析", "部署细节"] * 10
            }
            self.df = pd.DataFrame(test_data)
            print("已生成模拟语段库。")

    def reload_corpus(self, new_path=None):
        """重新加载语段库，支持动态切换"""
        if new_path:
            self.corpus_path = new_path
            if self.config_manager:
                self.config_manager.set("corpus_path", new_path)
        self._load_corpus()
        self._prepare_tfidf()

    def update_config(self, **kwargs):
        """更新配置参数（API key, URL等）并重建客户端"""
        if "api_key" in kwargs or "base_url" in kwargs:
            api_key = kwargs.get("api_key", self.config_manager.get("api_key"))
            base_url = kwargs.get("base_url", self.config_manager.get("base_url"))
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        if "intent_model" in kwargs:
            self.intent_model = kwargs["intent_model"]
        if "skill_model" in kwargs:
            self.skill_model = kwargs["skill_model"]
        if self.config_manager:
            self.config_manager.update(kwargs)

    def _load_embedding_model(self):
        if self.embedding_model is None:
            print("正在加载本地向量模型 (sentence-transformers/all-MiniLM-L6-v2)...")
            model_path = os.path.join(get_data_dir(), "sentence_transformers_model", "all-MiniLM-L6-v2")
            self.embedding_model = SentenceTransformer(model_path)

    def _prepare_tfidf(self):
        try:
            print(f"正在加载意图微调数据: {self.intent_train_path}")
            df_train = pd.read_excel(self.intent_train_path, header=None)
            raw_texts = df_train[1].astype(str).tolist()
            documents = [" ".join(text.split("；")) for text in raw_texts]
            documents = [doc.replace(";", " ") for doc in documents]

            self.tfidf_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=1)
            self.tfidf_vectorizer.fit(documents)
            self.tfidf_feature_names = self.tfidf_vectorizer.get_feature_names_out()
            self.tfidf_idf = self.tfidf_vectorizer.idf_
            print(f"TF-IDF模型构建完成，词汇表大小: {len(self.tfidf_feature_names)}")
        except Exception as e:
            print(f"警告：无法加载意图微调数据，将跳过TF-IDF清洗。错误: {e}")
            self.tfidf_vectorizer = None

    def _tfidf_prune(self, terms, keep_ratio=0.7):
        # 仍保留原方法，但后续会调用大模型审核
        if not terms or self.tfidf_vectorizer is None:
            return terms
        max_idf = max(self.tfidf_idf) if len(self.tfidf_idf) > 0 else 1.0
        term_idf = []
        for term in terms:
            if term in self.tfidf_feature_names:
                idx = list(self.tfidf_feature_names).index(term)
                idf_val = self.tfidf_idf[idx]
            else:
                idf_val = max_idf + 1.0
            term_idf.append((term, idf_val))
        term_idf.sort(key=lambda x: x[1], reverse=True)
        keep_count = max(1, int(len(term_idf) * keep_ratio))
        pruned = [t[0] for t in term_idf[:keep_count]]
        return pruned

    def _review_cleaning(self, original_keywords, cleaned_keywords, query, summary, labels):
        """调用大模型审核TF-IDF清洗结果，返回应保留的误删词"""
        if not original_keywords:
            return []
        prompt = f"""你是一个专业的清洗审核助手。请判断以下清洗过程中是否误删了重要的关键词。清洗目的是去除那些指代范围过于宽泛的词汇，例如“解放战争”、“中国共产党”、“中国”，仅保留具体的词汇。因此你认为应当保留的词汇也不应该过于宽泛。
原始用户查询：{query}
意图总结：{summary}
意图标签：{', '.join(labels)}
原始关键词列表：{', '.join(original_keywords)}
清洗后保留的关键词：{', '.join(cleaned_keywords)}

请找出在原始关键词中被误删的重要关键词（即应该保留但被删除了的词）。只输出这些词的列表，用顿号分隔，不要其他内容。如果没有误删，则输出“无”。
"""
        try:
            response = self.client.chat.completions.create(
                model=self.skill_model,  # 使用技能模型
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            content = response.choices[0].message.content.strip()
            if content == "无":
                return []
            # 解析关键词，支持顿号、逗号、空格分隔
            terms = re.split(r"[、，,;；\s]+", content)
            return [t.strip() for t in terms if t.strip()]
        except Exception as e:
            print(f"清洗审核调用失败: {e}")
            return []

    def get_intent_analysis(self, user_query):
        print("\n>>> 步骤1：意图识别模型调用（增强版）")
        combined_prompt = f"""你是一个专门用于从历史研究查询中提取结构化信息的分析工具。
你的核心功能是处理用户关于《解放战争作战日志》的查询，并严格按照规定格式输出分析结果。
请针对输入的课题或问题，完成以下四项任务，确保输出准确、结构清晰、内容专业：
一、关键词提取
从输入文本中提取全部名词性实体关键词，包括但不限于：地名、人名、时间、组织、事件等。要求提取完整、准确，提取出的关键词必须在原文中出现过。禁止出现“解放战争”“作战日志”等过于广泛的词。
二、意图标签生成
基于输入文本，归纳出若干简洁的意图标签，便于分类与检索。禁止出现“解放战争”“作战日志”等过于广泛的词。
三、逻辑关系解析
分析关键词之间的逻辑关系，输出逻辑组。使用“且”表示AND关系，使用“或”表示OR关系。格式：关键词A且关键词B；关键词C或关键词D。
四、意图总结
概括输入文本在《解放战争作战日志》中的检索意图，说明检索者希望获取哪些信息。总结应简洁自然，严格控制在150–300字。
输出格式（仅输出以下内容，无额外说明）：
关键词：关键词1、关键词2、关键词3……
标签：标签1、标签2、标签3……
逻辑组：A且B；C或D
总结：意图总结文本
示例：
输入：第四野战军在渡江战役中对敌军指挥官的战术判断与决策研究
输出：
关键词：第四野战军、渡江战役、敌军指挥官
标签：战术判断、决策制定、兵力部署、指令执行
逻辑组：第四野战军且渡江战役
总结：检索者旨在从《解放战争作战日志》中，获取第四野战军在渡江战役开展过程中，针对敌军指挥官所进行的战术判断相关作战记载，以及该部队依托相关判断完成的决策制定、兵力部署、指令执行等具体内容，为第四野战军在渡江战役中的战术判断与决策相关课题研究提供详实的史料依据。
待处理输入：{user_query}"""
        try:
            with httpx.Client(timeout=30.0) as http_client:
                client = OpenAI(
                    api_key=self.client.api_key,
                    base_url=self.client.base_url,
                    http_client=http_client
                )
                response = client.chat.completions.create(
                    model=self.intent_model,
                    messages=[{"role": "user", "content": combined_prompt}],
                    temperature=0.1,
                    extra_body={"enable_thinking": False}
                )
            content = response.choices[0].message.content.strip()
            return content
        except httpx.TimeoutException:
            print("错误：API请求超时（30秒），请检查网络或服务端负载。")
            return None
        except Exception as e:
            print(f"意图识别API调用失败: {type(e).__name__}: {e}")
            return None

    def parse_intent(self, raw_output):
        print("\n>>> 步骤2：解析增强意图输出")
        try:
            lines = raw_output.strip().split('\n')
            keywords = []
            labels = []
            logic_groups = []
            summary = ""
            for line in lines:
                if line.startswith("关键词："):
                    kw_text = line[4:].strip()
                    keywords = re.split(r"[、，,;；\s]+", kw_text) if kw_text else []
                    keywords = [k for k in keywords if k]
                elif line.startswith("标签："):
                    lb_text = line[3:].strip()
                    labels = re.split(r"[、，,;；\s]+", lb_text) if lb_text else []
                    labels = [l for l in labels if l]
                elif line.startswith("逻辑组："):
                    logic_text = line[4:].strip()
                    if logic_text:
                        groups = logic_text.split('；')
                        for g in groups:
                            if '且' in g:
                                parts = re.split(r'且', g)
                                and_group = [p.strip() for p in parts if p.strip()]
                                if and_group:
                                    logic_groups.append(('and', and_group))
                            elif '或' in g:
                                parts = re.split(r'或', g)
                                or_group = [p.strip() for p in parts if p.strip()]
                                if or_group:
                                    logic_groups.append(('or', or_group))
                elif line.startswith("总结："):
                    summary = line[3:].strip()

            print(f"原始关键词: {keywords}")
            print(f"原始标签: {labels}")
            print(f"逻辑组: {logic_groups}")
            print(f"意图总结: {summary[:50]}...")
            return {"keywords": keywords, "labels": labels, "logic_groups": logic_groups, "summary": summary}
        except Exception as e:
            print(f"解析意图结果出错: {e}")
            return {"keywords": [], "labels": [], "logic_groups": [], "summary": ""}

    def clean_keywords_labels(self, parsed_data, original_query):
        print("\n>>> 步骤3：TF-IDF清洗 + 大模型审核")
        original_keywords = parsed_data.get("keywords", [])
        original_labels = parsed_data.get("labels", [])

        cleaned_keywords = self._tfidf_prune(original_keywords, keep_ratio=0.7)
        cleaned_labels = self._tfidf_prune(original_labels, keep_ratio=0.7)

        # 调用大模型审核关键词
        reviewed_keywords = self._review_cleaning(
            original_keywords, cleaned_keywords,
            original_query, parsed_data.get("summary", ""), original_labels
        )
        # 将审核返回的词合并到清洗后列表中（去重）
        final_keywords = list(set(cleaned_keywords + reviewed_keywords))

        # 同样审核标签（简化：标签暂不审核，直接使用清洗后的）
        final_labels = cleaned_labels

        print(f"清洗后关键词: {cleaned_keywords}")
        print(f"审核后补充: {reviewed_keywords}")
        print(f"最终关键词: {final_keywords}")
        print(f"清洗后标签: {final_labels}")

        parsed_data["keywords_cleaned"] = final_keywords
        parsed_data["labels_cleaned"] = final_labels
        # 将逻辑组转换为内部格式
        keyword_and_groups = []
        keyword_or_groups = []
        for rel, group in parsed_data.get("logic_groups", []):
            if rel == 'and':
                keyword_and_groups.append(group)
            elif rel == 'or':
                keyword_or_groups.append(group)
        parsed_data["keyword_and_groups"] = keyword_and_groups
        parsed_data["keyword_or_groups"] = keyword_or_groups

        # 标签的逻辑组暂时未用，可留空
        parsed_data["label_and_groups"] = []
        parsed_data["label_or_groups"] = []
        return parsed_data

    @staticmethod
    def _parse_input_with_and(text):
        if not text:
            return [], []
        items = re.split(r"[+，,;；\s]+", text)
        ordinary = []
        and_groups = []
        for item in items:
            item = item.strip()
            if not item:
                continue
            if '*' in item:
                sub_words = [w.strip() for w in item.split('*') if w.strip()]
                if sub_words:
                    and_groups.append(sub_words)
            else:
                ordinary.append(item)
        return ordinary, and_groups

    def keyword_retrieval(self, parsed_data):
        print("\n>>> 【模块】关键词检索（支持AND/OR）")
        ordinary_keywords = parsed_data.get("keywords_cleaned", [])
        and_groups = parsed_data.get("keyword_and_groups", [])
        or_groups = parsed_data.get("keyword_or_groups", [])

        # 如果存在AND组，则忽略普通关键词（OR）
        use_and_only = len(and_groups) > 0
        if use_and_only:
            ordinary_keywords = []   # 强制忽略普通关键词

        if not ordinary_keywords and not and_groups and not or_groups:
            print("无有效关键词，返回空列表。")
            return []

        results = []
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="关键词检索中"):
            text_combined = f"{row['原始文本']} {row['语段标签库']} {row['语段讲解库']}"
            confidence = 0.0

            if use_and_only:
                # 仅计算AND组
                for group in and_groups:
                    group_sims = []
                    group_ok = True
                    for word in group:
                        if word in text_combined:
                            sim = 1.0
                        else:
                            sim = fuzz.partial_ratio(word, text_combined) / 100.0
                        if sim >= 0.85:
                            group_sims.append(sim)
                        else:
                            group_ok = False
                            break
                    if group_ok and group_sims:
                        group_conf = min(group_sims)
                        confidence = max(confidence, group_conf)
            else:
                # 计算OR普通关键词
                for kw in ordinary_keywords:
                    if kw in text_combined:
                        confidence = 1.0
                        break
                    else:
                        sim = fuzz.partial_ratio(kw, text_combined) / 100.0
                        if sim >= 0.85:
                            confidence = max(confidence, sim)
                # 计算OR组
                for group in or_groups:
                    group_sims = []
                    for word in group:
                        if word in text_combined:
                            sim = 1.0
                        else:
                            sim = fuzz.partial_ratio(word, text_combined) / 100.0
                        group_sims.append(sim)
                    # OR组：取最大值
                    group_conf = max(group_sims)
                    confidence = max(confidence, group_conf)

            if confidence > 0:
                item = row.to_dict()
                item["confidence"] = confidence
                item["_row_idx"] = idx
                results.append(item)

        print(f"关键词检索命中 {len(results)} 条")
        return results

    def _skills_retrieval_core(self, parsed_data, original_query):
        print("\n>>> 【模块内部】Skills检索核心")
        time_unit, force_name = self._extract_time_force(original_query, parsed_data)
        subset = None
        if time_unit or force_name:
            subset = self._filter_by_time_force(time_unit, force_name)
            print(f"大模型定位：时间范围={time_unit}, 部队={force_name}，筛选出 {len(subset)} 条记录")
        else:
            print("未提取到明确的时间/部队信息，转用向量语义定位...")
            self._load_embedding_model()
            summary = parsed_data["summary"]
            summary_vec = self.embedding_model.encode([summary])[0]
            explanations = self.df["语段讲解库"].tolist()
            exp_vecs = self.embedding_model.encode(explanations, batch_size=64, show_progress_bar=False)
            sims = cosine_similarity([summary_vec], exp_vecs)[0]
            mask = sims > 0.5  # 阈值降低至0.5
            subset = self.df[mask].copy()
            subset["vector_sim"] = sims[mask]
            print(f"向量语义定位：相似度>0.5 共 {len(subset)} 条记录")

        if subset is None or len(subset) == 0:
            print("第一步定位结果为空，Skills检索返回空列表。")
            return []

        ordinary_labels = parsed_data.get("labels_cleaned", [])
        and_groups = parsed_data.get("label_and_groups", [])
        or_groups = parsed_data.get("label_or_groups", [])
        if not ordinary_labels and not and_groups and not or_groups:
            print("无有效的意图标签，Skills检索返回空列表。")
            return []

        threshold = 0.6
        use_and_only = len(and_groups) > 0
        if use_and_only:
            ordinary_labels = []

        step2_results = []
        for idx, row in tqdm(subset.iterrows(), total=len(subset), desc="标签匹配中"):
            text_combined = f"{row['原始文本']} {row['语段标签库']} {row['语段讲解库']}"
            confidence = 0.0

            if use_and_only:
                # 仅计算AND组
                for group in and_groups:
                    group_sims = []
                    group_ok = True
                    for word in group:
                        if word in text_combined:
                            sim = 1.0
                        else:
                            sim = fuzz.partial_ratio(word, text_combined) / 100.0
                        if sim >= threshold:
                            group_sims.append(sim)
                        else:
                            group_ok = False
                            break
                    if group_ok and group_sims:
                        group_conf = min(group_sims)
                        confidence = max(confidence, group_conf)
            else:
                # OR普通标签
                for lb in ordinary_labels:
                    if lb in text_combined:
                        confidence = 1.0
                        break
                    else:
                        sim = fuzz.partial_ratio(lb, text_combined) / 100.0
                        if sim >= threshold:
                            confidence = max(confidence, sim)
                # OR组
                for group in or_groups:
                    group_sims = []
                    for word in group:
                        if word in text_combined:
                            sim = 1.0
                        else:
                            sim = fuzz.partial_ratio(word, text_combined) / 100.0
                        group_sims.append(sim)
                    group_conf = max(group_sims)
                    confidence = max(confidence, group_conf)

            if confidence >= threshold:
                item = row.to_dict()
                item["confidence"] = confidence
                item["_row_idx"] = idx
                step2_results.append(item)

        print(f"Skills标签匹配新增 {len(step2_results)} 条语段（置信度≥{threshold}）")
        return step2_results

    def skills_retrieval(self, parsed_data, original_query):
        print("\n>>> 【模块】Skills检索")
        return self._skills_retrieval_core(parsed_data, original_query)

    def _rewrite_query(self, query: str):
        """生成多个查询改写版本"""
        prompt = f"""请将以下用户查询改写成3个不同表述的版本，用于扩展检索，保留原意，可调整句式或使用同义词。
输出格式：每行一个版本，不包含序号。
用户查询：{query}"""
        try:
            response = self.client.chat.completions.create(
                model=self.intent_model,  # 可复用意图模型
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200,
                extra_body={"enable_thinking": False}
            )
            content = response.choices[0].message.content.strip()
            versions = [v.strip() for v in content.split('\n') if v.strip()]
            return versions[:5]  # 最多取5个
        except Exception as e:
            print(f"查询改写失败: {e}")
            return [query]  # 失败时返回原查询

    def _generate_vector_query(self, query, summary, keywords, labels):
        """生成优化后的向量查询文本"""
        prompt = f"""请根据用户查询和意图分析，生成一段精炼的文本，用于向量相似度检索，要求融合以下所有信息。
用户查询：{query}
意图总结：{summary}
关键词：{', '.join(keywords)}
标签：{', '.join(labels)}
输出文本：直接输出一句话。"""
        try:
            response = self.client.chat.completions.create(
                model=self.skill_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"生成向量查询失败: {e}")
            # 降级到原拼接方式
            return " ".join([query, summary] + keywords + labels)

    def vector_retrieval(self, parsed_data, original_query):
        print("\n>>> 【模块】向量检索（优化查询）")
        self._load_embedding_model()

        # 生成优化后的查询文本
        opt_query = self._generate_vector_query(
            original_query,
            parsed_data.get("summary", ""),
            parsed_data.get("keywords", []),
            parsed_data.get("labels", [])
        )
        print(f"优化后查询: {opt_query}")

        query_vec = self.embedding_model.encode([opt_query])[0]

        combined_texts = (self.df["原始文本"] + " " + self.df["语段标签库"] + " " + self.df["语段讲解库"]).tolist()
        doc_vecs = self.embedding_model.encode(combined_texts, batch_size=64, show_progress_bar=True)
        sim_scores = cosine_similarity([query_vec], doc_vecs)[0]

        results = []
        for idx, score in enumerate(sim_scores):
            row = self.df.iloc[idx].to_dict()
            row["confidence"] = score
            row["_row_idx"] = idx
            results.append(row)

        results.sort(key=lambda x: x["confidence"], reverse=True)
        print(f"向量检索完成，共 {len(results)} 条，已按相似度降序排列。")
        return results

    def _rerank(self, query: str, candidates: List[Dict], top_k: int = 50) -> List[Dict]:
        """对候选列表重排序"""
        if not candidates:
            return []
        # 只重排前top_k条
        to_rerank = candidates[:top_k]
        for item in tqdm(to_rerank, desc="重排序中"):
            text = item.get("原始文本", "") + " " + item.get("语段讲解库", "")
            if not text:
                continue
            prompt = f"""请根据原始查询对以下文段的相关性进行打分（1-5分，5为非常相关）。
原始查询：{query}
文段：{text}
只输出分数，不要其他内容。"""
            try:
                response = self.client.chat.completions.create(
                    model=self.skill_model,  # 使用技能模型
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=10,
                    extra_body={"enable_thinking": False}
                )
                score_str = response.choices[0].message.content.strip()
                score = float(score_str) if score_str.replace('.','').isdigit() else 0.0
                item["rerank_score"] = score
            except Exception as e:
                item["rerank_score"] = item.get("confidence", 0.0)
        # 按新分数排序
        to_rerank.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        # 将未参与重排的条目追加到后面（按原序）
        remaining = candidates[top_k:]
        return to_rerank + remaining

    def stage1_retrieval(self, parsed_data):
        print("\n>>> 路径一：关键词检索")
        return self.keyword_retrieval(parsed_data)

    def path2_agent_skills(self, initial_results, parsed_data, original_query):
        print("\n>>> 路径二：Agent Skills渐进式检索启动")
        skills_results = self._skills_retrieval_core(parsed_data, original_query)

        combined = initial_results + skills_results
        seen = set()
        unique = []
        for item in combined:
            text = item["原始文本"]
            if text not in seen:
                unique.append(item)
                seen.add(text)
        # 按置信度排序（问题3）
        unique.sort(key=lambda x: x["confidence"], reverse=True)
        return unique

    def path3_full_vector_search(self, parsed_data, original_query):
        return self.vector_retrieval(parsed_data, original_query)

    def _extract_time_force(self, query, parsed_data):
        prompt = f"""你是一个历史研究辅助助手，请从以下信息中提取【时间范围】和【部队名称】。
如果信息中没有明确的部队或时间，请回答“无”。
注意：
- 时间范围：如“1948年夏”应转化为具体月份范围（如1948年5-9月），若不能精确转化则保留原始描述。
- 部队名称：使用标准全称，如“第一野战军”、“华东野战军”等。

检索者原句：{query}
意图总结：{parsed_data.get("summary", "")}
意图标签：{parsed_data.get("labels", [])}
关键词：{parsed_data.get("keywords", [])}

输出格式：
时间范围：<提取结果或“无”>
部队名称：<提取结果或“无”>
"""
        try:
            response = self.client.chat.completions.create(
                model=self.skill_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100,
                extra_body={"enable_thinking": True}
            )
            content = response.choices[0].message.content.strip()
            time_match = re.search(r"时间范围[：:]\s*(.+)", content)
            force_match = re.search(r"部队名称[：:]\s*(.+)", content)
            time_info = time_match.group(1).strip() if time_match else None
            force_info = force_match.group(1).strip() if force_match else None
            if time_info == "无":
                time_info = None
            if force_info == "无":
                force_info = None
            return time_info, force_info
        except Exception as e:
            print(f"大模型提取时间/部队失败: {e}")
            return None, None

    def _filter_by_time_force(self, time_desc, force_name):
        df = self.df.copy()
        mask = pd.Series([True] * len(df))

        if force_name:
            force_aliases = {
                "一野": ["第一野战军", "一野"],
                "二野": ["第二野战军", "二野"],
                "三野": ["第三野战军", "三野"],
                "四野": ["第四野战军", "四野"],
                "华野": ["华东野战军", "华野"],
                "东野": ["东北野战军", "东野"],
                "中野": ["中原野战军", "中野"],
                "第一野战军": ["第一野战军", "一野"],
                "第二野战军": ["第二野战军", "二野"],
                "第三野战军": ["第三野战军", "三野"],
                "第四野战军": ["第四野战军", "四野"],
                "华东野战军": ["华东野战军", "华野"],
                "东北野战军": ["东北野战军", "东野"],
                "中原野战军": ["中原野战军", "中野"],
            }
            patterns = force_aliases.get(force_name, [force_name])
            pattern = '|'.join([re.escape(p) for p in patterns if p])
            if pattern:
                mask &= df["部队名称"].str.contains(pattern, na=False)

        if time_desc:
            years = re.findall(r"\d{4}", time_desc)
            if years:
                year_mask = df["年月日"].astype(str).str[:4].isin(years)
                month_range = re.search(r"(\d+)[-～至](\d+)月", time_desc)
                if month_range:
                    start_m, end_m = int(month_range.group(1)), int(month_range.group(2))
                    df["temp_month"] = df["年月日"].astype(str).str[5:7].str.lstrip('0')
                    df["temp_month"] = pd.to_numeric(df["temp_month"], errors='coerce')
                    month_mask = df["temp_month"].between(start_m, end_m)
                    year_mask &= month_mask.fillna(False)
                    df.drop(columns=["temp_month"], inplace=True)
                mask &= year_mask
            else:
                mask &= df["年月日"].astype(str).str.contains(time_desc, na=False)

        return df[mask]

    def save_results(self, results, filename, confidence_col_name=None):
        """
        保存结果到XLSX，输出目录由配置决定
        """
        if not results:
            print("没有结果可保存。")
            return
        output_dir = self.config_manager.get("output_dir")
        full_path = os.path.join(output_dir, filename)
        original_columns = self.df.columns.tolist()
        out_df = pd.DataFrame(results)
        out_df = out_df[original_columns] if all(col in out_df.columns for col in original_columns) else out_df
        if confidence_col_name and 'confidence' in out_df.columns:
            out_df[confidence_col_name] = out_df['confidence']
        if "检索来源" in out_df.columns:
            cols = original_columns + ["检索来源"]
            if confidence_col_name and confidence_col_name not in cols:
                cols.append(confidence_col_name)
            out_df = out_df[cols]
        out_df.to_excel(full_path, index=False, engine='openpyxl')
        print(f"结果已保存至: {full_path} (共 {len(out_df)} 条)")
        return full_path