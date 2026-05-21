import os
import pandas as pd
from openai import OpenAI
from config_manager import ConfigManager
from concurrent.futures import ThreadPoolExecutor, as_completed

class TaggingProcessor:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.client = OpenAI(
            api_key=config_manager.get("api_key"),
            base_url=config_manager.get("base_url"),
            timeout=30.0  # 每个请求30秒超时
        )
        self.model = config_manager.get("tagging_model")
        if not self.model:
            raise ValueError("配置中缺少 tagging_model，请在系统设置中配置标签化模型名称")

    def process_excel_file(self, input_path: str, output_filename: str) -> str:
        df = pd.read_excel(input_path, dtype=str)
        # 确保至少有两列
        if len(df.columns) < 2:
            df["语段标签库"] = ""
        else:
            df.columns.values[1] = "语段标签库"

        # 并发处理每一行
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_idx = {}
            for idx in range(len(df)):
                text = df.iloc[idx, 0]
                if pd.isna(text) or not str(text).strip():
                    df.iloc[idx, 1] = "文本为空"
                    continue
                future = executor.submit(self._get_tags, str(text))
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    tags = future.result()
                except Exception as e:
                    tags = f"【处理异常】: {str(e)}"
                df.iloc[idx, 1] = tags

        output_dir = self.config_manager.get("output_dir")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        df.to_excel(output_path, index=False, engine='openpyxl')
        return output_path

    def _get_tags(self, text: str) -> str:
        system_prompt = """你是一名中国近现代史文献研究专家，擅长对原始军事档案文本进行深度语义解析与关键信息凝练。你的核心能力是从非结构化的历史记录中，精准识别、抽取和归纳出具有研究价值的概念实体与主题标签，严格遵循学术规范与任务要求。"""

        user_template = """请执行以下文本标签化任务：
1.任务定义
对给定的《解放战争作战日志》原始文本段落，执行概念抽取与总结式标签化。该任务需同步完成两项操作：
-抽取：识别并列出文本中所有承载关键信息的具体名词性实体。
-总结：基于文本整体语义，凝练出反映核心事件、政策、工作或战略战术范畴的、具有学术研究价值的抽象概念标签。
2.输出要求与格式
-输出格式：仅输出最终标签列表，以中文顿号"、"分隔。不附加任何解释性、分析性、引导性或总结性文字。
-标签性质：
-抽取式标签：文本中明确出现的具体实体，如"华东野战军第9纵队"、"临沂"、"1947年5月14日"、"MG42机枪"。
-总结式标签：需基于文本内容进行概括和抽象，生成与中国近代史研究（军事、政治、社会史）高度相关的学术性概念，如城市接管、围点打援、后勤补给、土地改革、俘虏政策、军事整训。
-处理原则：
-同时涵盖具体实体与抽象概念。
-总结式标签应紧扣文本内涵，避免过度泛化或无依据的推断。
-确保所有标签均为名词或名词性短语。
3.示例参考（基于假设文本）
-输入文本示例："1948年11月7日，中原野战军一部协同地方武装，于徐州以东组织群众转运粮秣，并开展对敌铁路线的破袭，保障主力侧翼安全。同日，政治部下发关于对新解放城镇工商业者进行政策宣传的指示。"
-预期输出示例：1948年11月7日、中原野战军、地方武装、徐州、群众支前、粮秣转运、破袭战、铁路交通线、侧翼安全、政治工作、新解放区、城市政策、工商业者、政策宣传
请严格遵循以上指令，对输入文本进行处理。
输入文本："""

        user_prompt = user_template + f'"{text}"'

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                extra_body={"enable_thinking": False},
                temperature=0.1,
                max_tokens=500
            )
            result = completion.choices[0].message.content.strip()
            return result
        except Exception as e:
            return f"【调用失败】: {str(e)}"