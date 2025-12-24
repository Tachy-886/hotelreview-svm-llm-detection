from __future__ import annotations
from typing import Dict, List
from dashscope import Generation
import dashscope
import json

dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"


SYSTEM_PROMPT = (
    "你是一名严谨的酒店评论综合分析与决策助手。\n"
    "你只能基于输入提供的评论内容、SVM 判定结果与置信度分数进行分析。\n"
    "你必须尊重 SVM 判定，不得反向推翻。\n"
    "你必须用中文输出。\n"
)

PROMPT_TEMPLATE = """
现在给你多个酒店的评论集合。每家酒店包含多条评论，
并且每条评论都给出了一个 SVM 的真假判定与置信度分数。

重要规则（必须遵守）：
1. 判定=真实：score 为负数，且越负代表置信度越高。
2. 判定=虚假：score 为正数，且越正代表置信度越高。
3. score 越接近 0，模型越不确定。
4. 如果某家酒店中存在多条高置信度“虚假”评论，应显著降低其可信度。
5. 不得编造评论中未出现的信息。

你的任务：
1. 对每家酒店：
   - 概括主要优点
   - 概括主要缺点
   - 分析评论整体可信度（结合所有评论的判定与 score）
   - 给出入住建议（推荐 / 谨慎推荐 / 不推荐）
2. 在所有酒店之间做最终对比，给出一个最终选择。

输出格式（必须严格遵守，全中文）：

【酒店逐条分析】
- 酒店：<酒店名>
  - 评论概述：...
  - 可信度分析：...
  - 入住建议：...
  - 理由：...

（对每家酒店重复）

【最终对比与选择】
- 最终选择：<酒店名 / 都不推荐>
- 选择理由：...

以下是酒店评论数据（JSON），只能基于这些数据分析：
{hotels_json}
"""


def analyze_hotels_and_choose(
    api_key: str,
    hotel_reviews: Dict[str, List[dict]],
    model: str = "qwen3-max"
) -> str:
    """
    hotel_reviews 结构示例：
    {
      "hotel1": [
        {"zh": "...", "label": "truthful", "score": -0.6},
        {"zh": "...", "label": "deceptive", "score": 1.4}
      ],
      "hotel2": [...]
    }
    """

    # 中文化结构
    payload = {}
    for hotel, reviews in hotel_reviews.items():
        payload[hotel] = []
        for r in reviews:
            payload[hotel].append({
                "评论": r["zh"],
                "判定": "真实" if r["label"] == "truthful" else "虚假",
                "置信度分数": round(float(r["score"]), 3),
                "规则说明": "真实：score 越负越可信；虚假：score 越正越可信"
            })

    hotels_json = json.dumps(payload, ensure_ascii=False, indent=2)
    prompt = PROMPT_TEMPLATE.format(hotels_json=hotels_json)

    user_content = f"{SYSTEM_PROMPT}\n\n{prompt}"

    response = Generation.call(
        api_key=api_key,
        model=model,
        messages=[
            {"role": "user", "content": user_content},
        ],
        result_format="message",
        enable_thinking=False,
    )

    if response.status_code != 200:
        raise RuntimeError(response.message)

    return response.output.choices[0].message.content.strip()
