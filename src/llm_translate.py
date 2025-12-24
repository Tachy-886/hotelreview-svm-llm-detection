"""
LLM Translation Module (DashScope / Qwen)

功能：
- translate_text_to_english: 输入任意语言文本 -> 输出英文译文（只返回译文，不带多余内容）
- translate_to_txt: 输入文本 -> 翻译 -> 保存为 .txt

设计目标：
- 作为后续 “图片->OCR->翻译” 流水线的基石模块
- 输出格式稳定（便于下游继续处理）
"""

# src/llm_translate.py（只展示需要替换的 translate_text_to_english 函数）
from __future__ import annotations
from pathlib import Path
from dashscope import Generation
import dashscope

dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

SYSTEM_PROMPT = (
    "You are a high-precision translation engine. "
    "You do not chat. You only translate."
)

TRANSLATE_PROMPT_TEMPLATE = """Translate the following text into English.

Rules (must follow):
- Output ONLY the English translation. Do not add any explanation, prefix, or quotes.
- Preserve the original formatting as much as possible (line breaks, bullet points, punctuation).
- Preserve numbers, dates, emojis, and special symbols.
- Preserve proper nouns (hotel names, place names, product names) by transliteration or keeping the original if appropriate.
- If the input is already English, output it unchanged.

Text:
\"\"\"{text}\"\"\"
"""

def translate_text_to_english(
    api_key: str,
    text: str,
    model: str = "qwen-mt-plus",
) -> str:
    prompt = TRANSLATE_PROMPT_TEMPLATE.format(text=str(text))

    # ✅ system 合并进 user，避免 system role 报错
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

    out = response.output.choices[0].message.content
    return out.strip()



def translate_to_txt(
    api_key: str,
    text: str,
    output_txt_path: str,
    model: str = "qwen-mt-plus",
    encoding: str = "utf-8",
) -> str:
    """
    翻译 text -> 保存为英文 .txt，返回英文译文
    """
    english = translate_text_to_english(api_key=api_key, text=text, model=model)
    out_path = Path(output_txt_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(english, encoding=encoding)
    return english
