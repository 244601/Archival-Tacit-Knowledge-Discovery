import os
import pandas as pd
from openai import OpenAI
from config_manager import ConfigManager
from concurrent.futures import ThreadPoolExecutor, as_completed

class RewritingProcessor:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.client = OpenAI(
            api_key=config_manager.get("api_key"),
            base_url=config_manager.get("base_url"),
            timeout=30.0
        )
        self.model = config_manager.get("rewriting_model")
        if not self.model:
            raise ValueError("配置中缺少 rewriting_model，请在系统设置中配置转写模型名称")

    def process_excel_file(self, input_path: str, output_filename: str) -> str:
        df = pd.read_excel(input_path, dtype=str)
        # 确保至少有3列
        while len(df.columns) < 3:
            df[f"临时列_{len(df.columns)}"] = ""

        # 列重命名逻辑（保持不变）
        if len(df.columns) == 1:
            df["部队名称"] = ""
            df["年月日"] = ""
            cols = ["部队名称", "年月日", df.columns[0]]
            df = df[cols]
        elif len(df.columns) == 2:
            df["原始文本"] = ""
            cols = [df.columns[0], df.columns[1], "原始文本"]
            df = df[cols]
        else:
            df.columns.values[0] = "部队名称"
            df.columns.values[1] = "年月日"
            df.columns.values[2] = "原始文本"

        if len(df.columns) < 4:
            df["语段讲解库"] = ""
        else:
            df.columns.values[3] = "语段讲解库"

        # 并发处理每一行
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_idx = {}
            for idx in range(len(df)):
                troop = df.iloc[idx, 0] if len(df.columns) > 0 else ""
                date = df.iloc[idx, 1] if len(df.columns) > 1 else ""
                text = df.iloc[idx, 2] if len(df.columns) > 2 else ""
                if pd.isna(text) or not str(text).strip():
                    df.iloc[idx, 3] = "原始文本为空"
                    continue
                future = executor.submit(self._get_explanation, str(troop), str(date), str(text))
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    explanation = future.result()
                except Exception as e:
                    explanation = f"【处理异常】: {str(e)}"
                df.iloc[idx, 3] = explanation

        output_dir = self.config_manager.get("output_dir")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        df.to_excel(output_path, index=False, engine='openpyxl')
        return output_path

    def _get_explanation(self, troop: str, date: str, text: str) -> str:
        system_prompt = """你是一名中国近现代史文献研究专家，擅长对原始军事档案文本进行深度语义解析与关键信息凝练。你的核心能力是对各种历史记录进行解读，严格遵循学术规范与任务要求。"""

        user_template = """核心任务：对《解放战争作战日志》中{部队名称}{年月日}的记录进行专业化学术分析，要求分层次展开，突出深层历史意义与战略内涵。
一、历史状况分析
-结合上下文，将此次行动置于{年月日}的历史时期，分析其战略目的、战略方案、战斗部署。
二、体现出的策略解读
-军事层面：是否体现了哪一组织某种战斗策略、军事部署、战斗特点？（若没有体现或体现不明显则忽略该方面）
-政治层面：是否体现了哪一组织某种政策或者措施？（若没有体现或体现不明显则忽略该方面）
-经济建设与后勤方面：是否体现了哪一组织的经济建设或者后勤状况？（若没有体现或体现不明显则忽略该方面）
-思想方面：是否体现了哪一组织的思想状况、精神面貌？（若没有体现或体现不明显则忽略该方面）
输出要求：
-只输出解读内容，不增加额外的说明，不复述原文内容、不引用其他内容；
-语言严谨、客观，避免主观臆断，要完全贴合原文；
-注意输出内容的学术性、专业性、严谨性；
-保持用语平实。
输入文段：{原始文本}"""

        user_prompt = user_template.format(部队名称=troop, 年月日=date, 原始文本=text)

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                extra_body={"enable_thinking": False},
                temperature=0.1,
                max_tokens=800
            )
            result = completion.choices[0].message.content.strip()
            return result
        except Exception as e:
            return f"【调用失败】: {str(e)}"