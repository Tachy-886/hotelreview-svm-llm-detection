from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import dashscope
from dashscope import MultiModalConversation

# 需要时可切新加坡：
# dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"


SYSTEM_PROMPT = "你是一个严谨的OCR信息抽取系统，只做文本抽取，不要解释。"

EXTRACT_PROMPT = """
请从这张“酒店/商家评价截图”中抽取【评论正文文本】。

抽取规则（必须严格遵守）：
1) 只输出评论正文，不要输出：用户名/头像/评分/日期/地点/房型/按钮/价格/酒店回复/“展开”等UI文字。
2) 如果图片中有多条评论，按从上到下的顺序依次抽取，分别作为数组元素。
3) 保留原始语言（不要翻译），尽量保持原文内容与标点；可把同一条评论的换行合并为一行。
4) 如果正文被“...”截断，只输出截图里可见的部分。
5) 输出必须是严格 JSON：一个字符串数组，例如：
   ["评论正文1", "评论正文2"]
6) 如果没有识别到任何评论正文，输出：[]

现在开始输出JSON：
""".strip()


def _safe_json_list(text: str) -> List[str]:
    """
    尽量把模型输出解析成 List[str]。
    模型偶尔会夹杂多余文字，这里做容错。
    """
    text = text.strip()

    # 1) 直接解析
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass

    # 2) 尝试截取第一个 [...] 再解析
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        chunk = m.group(0)
        try:
            obj = json.loads(chunk)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            pass

    # 3) 兜底：按行抽取（不推荐，但保证不崩）
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[:10]  # 防止异常爆长


def extract_reviews_from_image(
    api_key: str,
    image_path: str,
    model: str = "qwen3-vl-plus",
    timeout_hint: bool = False,
) -> List[str]:
    """
    输入：单张截图路径（png/jpg/webp）
    输出：该图中识别到的评论正文列表（可能为空，可能多条）
    """
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 直接用 file:// 本地路径（更省事也更快）
    img_uri = f"file://{img_path.resolve()}"

    messages = [
        {
            "role": "system",
            "content": [{"text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"image": img_uri},
                {"text": EXTRACT_PROMPT},
            ],
        },
    ]

    resp = MultiModalConversation.call(
        api_key=api_key,
        model=model,
        messages=messages,
    )

    if getattr(resp, "status_code", 200) != 200:
        # dashscope SDK 不同版本字段略有差异，这里尽量兼容
        msg = getattr(resp, "message", str(resp))
        raise RuntimeError(msg)

    # 官方示例：resp.output.choices[0].message.content[0]["text"]
    raw = resp.output.choices[0].message.content[0]["text"]
    return _safe_json_list(raw)


def extract_reviews_from_folder(
    api_key: str,
    folder_path: str,
    model: str = "qwen3-vl-plus",
    max_workers: int = 4,
    exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),
) -> List[Tuple[str, List[str]]]:
    """
    批量处理文件夹中的截图，返回：
    [
      ("xxx.png", ["评论正文1", "评论正文2"]),
      ...
    ]
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    files = sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])

    results: List[Tuple[str, List[str]]] = []
    if not files:
        return results

    # 并发请求（注意别开太大，避免限流）
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_map = {
            pool.submit(extract_reviews_from_image, api_key, str(p), model): p.name
            for p in files
        }
        for fut in as_completed(fut_map):
            name = fut_map[fut]
            try:
                texts = fut.result()
            except Exception as e:
                texts = [f"[ERROR] {e}"]
            results.append((name, texts))

    # 保持输出按文件名排序（as_completed会乱序）
    results.sort(key=lambda x: x[0])
    return results


def format_reviews_numbered(
    folder_results: List[Tuple[str, List[str]]],
    start_index: int = 1,
) -> List[str]:
    """
    把批量结果展平成：
    评论1, "..."
    评论2, "..."
    """
    lines: List[str] = []
    k = start_index
    for _, texts in folder_results:
        for t in texts:
            t = t.replace("\n", " ").strip()
            t = t.replace('"', '\\"')  # 防止破坏引号
            if not t:
                continue
            lines.append(f'评论{k}, "{t}"')
            k += 1
    return lines


def save_lines(lines: List[str], out_path: str, encoding: str = "utf-8") -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines), encoding=encoding)
