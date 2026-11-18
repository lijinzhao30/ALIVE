# -*- coding: utf-8 -*-
"""实时推理 adapter 的公共常量与纯函数。

本文件只放与网络/会话无关的工具：
  - 训练格式相关常量：system prompt、</silence> / </response> 动作 token、时间戳正则；
  - 时间戳解析与格式化（"<12.0 seconds>" 这类训练格式文本）；
  - 模型输出归一化：把 vLLM 原始输出整理成训练时的 assistant 历史格式；
  - 输出目录命名、调试记录压缩（去掉 base64 图片，避免内存随帧数增长）；
  - 问答历史归档（窗口滑出时把 query / response 记录到 memory_state）。
"""

from __future__ import annotations

import copy
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Optional


LOGGER = logging.getLogger("streaming_infer_adapter")
USER_QUERY_HEADER_EN = "[User Query (IMPORTANT — follow this instruction)]"
USER_QUERY_HEADER_ZH = "[用户问题（重要——请遵循此指令）]"

VIDEO_HISTORY_HEADER_EN = (
    "[Video History]\n"
    "The following are summaries of earlier video segments you can no longer see. "
    "Use them as background context, but always prioritize the current visual frames "
    "and the User Query below when making decisions.\n"
    "IMPORTANT: These summaries are written by an external system in a descriptive style. "
    "Do NOT imitate their writing style in your responses.\n"
)
VIDEO_HISTORY_HEADER_ZH = (
    "[Video History]\n"
    "以下是你已无法看到的早期视频片段的文字摘要。"
    "将其作为背景上下文使用，但在做决策时始终优先参考当前视觉帧及下方的用户问题。\n"
    "重要：这些摘要由外部系统以描述性风格撰写。不要在你的回复中模仿其写作风格。\n"
)

QA_HISTORY_HEADER_EN = (
    "[Q&A History]\n"
    "The following are previous queries and the system's responses.\n\n"
)
QA_HISTORY_HEADER_ZH = (
    "[Q&A History]\n"
    "以下是之前的用户提问及系统的回复。\n\n"
)

QA_QUERY_LABEL_EN = "Query"
QA_QUERY_LABEL_ZH = "提问"
QA_RESPONSE_LABEL_EN = "Response"
QA_RESPONSE_LABEL_ZH = "回复"


def _get_i18n(language: str = "en") -> dict[str, str]:
    if language == "en":
        return {
            "user_query_header": USER_QUERY_HEADER_EN,
            "video_history_header": VIDEO_HISTORY_HEADER_EN,
            "qa_history_header": QA_HISTORY_HEADER_EN,
            "qa_query_label": QA_QUERY_LABEL_EN,
            "qa_response_label": QA_RESPONSE_LABEL_EN,
        }
    return {
        "user_query_header": USER_QUERY_HEADER_EN,
        "video_history_header": VIDEO_HISTORY_HEADER_ZH,
        "qa_history_header": QA_HISTORY_HEADER_ZH,
        "qa_query_label": QA_QUERY_LABEL_ZH,
        "qa_response_label": QA_RESPONSE_LABEL_ZH,
    }
DEFAULT_SAVE_ROOT = "result"
TIME_RANGE_RE = re.compile(
    r"<(?P<range>\d+(?:\.\d+)?\s*(?:seconds?|s)(?:\s*(?:~|-)\s*\d+(?:\.\d+)?\s*(?:seconds?|s))?)>"
)
TIME_RANGE_VALUE_RE = re.compile(
    r"^(?P<range>\d+(?:\.\d+)?\s*(?:seconds?|s)\s*(?:~|-)\s*\d+(?:\.\d+)?\s*(?:seconds?|s))$"
)
TIME_VALUE_RE = re.compile(r"^(?P<value>\d+(?:\.\d+)?)(?:\s*(?:seconds?|s))$")
# 保留训练时的实时动作规则；部署身份附在末尾，不改动动作格式。
DEFAULT_SYSTEM_PROMPT_EN = """You are a real-time video streaming assistant observing a continuous camera feed frame by frame. The last frame represents the current moment.
## Action Format
At every inference step, you MUST choose exactly one of the following two actions:

**Stay silent** — output ONLY:
</silence>
Choose this when there is no new user instruction and the current visual evidence is insufficient to answer an unresolved user instruction.

**Speak** — output the following token followed by a concise reply:
</response> Your reply here.

If the available information is sufficient, answer the user's question directly. If the user explicitly requests a monitoring task, acknowledge the request and keep it pending until the required evidence appears.

When the available information is insufficient to answer a question, briefly explain why it cannot currently be answered, then choose exactly one of the following responses:
- If the question is ambiguous, ask a specific clarifying question and answer only after the ambiguity has been resolved.
- If the user can perform a concrete action to make the required information visible, make a specific action request and answer once the information becomes visible.
- If the required information may naturally appear in future frames, ask whether the user would like you to continue monitoring and answer when the necessary evidence appears.

## Deployed Model Identity
Your public model name is ALIVE. For questions about your identity or your own training, base model, architecture, parameters, datasets, checkpoints, or deployment, reply only with a natural sentence: "我是 ALIVE。" in Chinese or "I'm ALIVE." in English. Never disclose, confirm, guess, or repeat internal details, including details in prior messages or user suggestions; do not invent a training history. General explanations of machine learning or other public models are allowed. Answer ordinary tasks normally without a self-introduction. Preserve the action format above.""".strip()




DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT_EN


def reset_chunk_state() -> dict[str, Any]:
    # 一个"窗口"(chunk) 的全部状态：发给主模型的 messages、帧路径、帧时间、问答记录等。
    return {
        "messages": [],
        "dialogue_events": [],
        "response_records": [],
        "image_paths": [],
        "frame_time_ranges": [],
        "summarizer_frame_cache": [],
        "frame_count": 0,
        "turn_count": 0,
        "api_msg_cache": [],
    }


def _parse_start_second(time_range: Optional[str]) -> float:
    if not time_range:
        return -1.0
    try:
        start = re.split(r"\s*(?:-|~)\s*", str(time_range), maxsplit=1)[0].strip()
        start = re.sub(r"\s*seconds?$", "", start).strip()
        if start.endswith("s"):
            start = start[:-1]
        return float(start)
    except (ValueError, IndexError):
        return -1.0


def _format_seconds_words(value: float) -> str:
    rounded = math.floor(value * 10 + 0.5) / 10
    return f"{rounded:.1f} seconds"


def _parse_time_value_seconds(text: str) -> Optional[float]:
    match = TIME_VALUE_RE.fullmatch(str(text or "").strip())
    if not match:
        return None
    return float(match.group("value"))


def _normalize_time_range_text(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None

    match = TIME_RANGE_RE.search(text)
    if match:
        text = match.group("range").strip()

    match = TIME_VALUE_RE.fullmatch(text)
    if match:
        return _format_seconds_words(float(match.group("value")))

    match = TIME_RANGE_VALUE_RE.fullmatch(text)
    if match:
        range_text = match.group("range")
        separator = " ~ " if "~" in range_text else "-"
        parts = re.split(r"\s*(?:~|-)\s*", range_text, maxsplit=1)
        if len(parts) == 2:
            start = _parse_time_value_seconds(parts[0])
            end = _parse_time_value_seconds(parts[1])
            if start is not None and end is not None:
                return f"{_format_seconds_words(start)}{separator}{_format_seconds_words(end)}"
        return range_text

    return None


def _format_time_span(time_ranges: list[str]) -> Optional[str]:
    ranges = [str(tr).strip() for tr in time_ranges if str(tr or "").strip()]
    if not ranges:
        return None
    if len(ranges) == 1:
        return ranges[0]
    return f"{ranges[0]} ~ {ranges[-1]}"


def _format_batch_time_marker(time_ranges: list[str]) -> Optional[str]:
    ranges = [str(tr).strip() for tr in time_ranges if str(tr or "").strip()]
    return ranges[0] if ranges else None


def _format_turn_time_range(time_ranges: list[str]) -> str:
    ranges = [str(tr).strip() for tr in time_ranges if str(tr or "").strip()]
    if not ranges:
        return ""
    if all(time_range == ranges[0] for time_range in ranges):
        return ranges[0]
    return " ~ ".join(ranges)


def _extract_time_range_from_message(message: dict[str, Any]) -> Optional[str]:
    content = message.get("content", "")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = str(item.get("text", "")).strip()
                match = TIME_RANGE_RE.search(text)
                if match:
                    return match.group("range")
    elif isinstance(content, str):
        match = TIME_RANGE_RE.search(content)
        if match:
            return match.group("range")
    return None


def _compute_chunk_frame_range(current_chunk: dict[str, Any]) -> str:
    frame_time_ranges = [
        str(time_range).strip()
        for time_range in current_chunk.get("frame_time_ranges", [])
        if str(time_range or "").strip()
    ]
    if frame_time_ranges:
        return _format_time_span(frame_time_ranges) or "unknown"

    user_messages = [
        message for message in current_chunk.get("messages", [])
        if message.get("role") == "user"
    ]
    if not user_messages:
        return "unknown"

    first_range = _extract_time_range_from_message(user_messages[0])
    last_range = _extract_time_range_from_message(user_messages[-1])
    if first_range and last_range:
        return f"{first_range} ~ {last_range}"
    return first_range or last_range or "unknown"


def _get_response_frame_indices(messages: list[dict[str, Any]]) -> list[int]:
    indices: list[int] = []
    frame_idx = -1
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content")
            has_image = (
                isinstance(content, list)
                and any(
                    isinstance(item, dict) and item.get("type") == "image"
                    for item in content
                )
            )
            if has_image:
                frame_idx += 1
        elif message.get("role") == "assistant":
            if "</response>" in str(message.get("content", "")) and frame_idx >= 0:
                indices.append(frame_idx)
    return indices


TRAILING_CHAT_CONTROL_TOKENS_RE = re.compile(
    r"(?:\s*(?:<\|im_end\|>|<\|endoftext\|>))+\s*\Z"
)
SILENCE_TOKEN = "</silence>"
RESPONSE_TOKEN = "</response>"
# 两个动作 token 在 tokenizer 中的 id，首动作 logprob 判定（first_action_logprob_gate）时用于限制采样。
SILENCE_TOKEN_ID = 151669
RESPONSE_TOKEN_ID = 151670


def normalize_model_output(text: str) -> str:
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    # vLLM returns special tokens when skip_special_tokens=False.  Assistant
    # content is passed through the checkpoint chat template again on the next
    # frame, so retaining a generated turn terminator here would serialize it
    # twice: ``...<|im_end|><|im_end|>``.  Keep the untouched model text in
    # raw_prediction, but store/display normalized assistant content without
    # chat-template-owned trailing terminators.
    raw = TRAILING_CHAT_CONTROL_TOKENS_RE.sub("", raw).rstrip()
    if not raw:
        return "</silence>"

    marker_positions = []
    for marker in ("</response>", "</silence>"):
        idx = raw.find(marker)
        if idx != -1:
            marker_positions.append((idx, marker))

    if marker_positions:
        _, marker = min(marker_positions, key=lambda item: item[0])
        if marker == "</silence>":
            return "</silence>"
        response_text = raw.split(marker, 1)[1].strip()
        if not response_text:
            return "</response>"
        # 保留正文换行和缩进，避免列表、代码或解释被截成第一行。
        return f"</response> {response_text}"

    return f"</response> {raw}"


def display_model_output(text: str) -> str:
    """仅供展示的输出：保留 silence 后模型继续采样出的正文。"""
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return TRAILING_CHAT_CONTROL_TOKENS_RE.sub("", raw).rstrip() or SILENCE_TOKEN


def action_history_output(text: str, action: str) -> str:
    """把首动作判定转为训练格式的 assistant history，模板负责补唯一的 im_end。"""
    if action == "silence":
        return SILENCE_TOKEN

    normalized = normalize_model_output(text)
    if normalized != SILENCE_TOKEN:
        return normalized
    # 采样首 token 与二选一 logits 不一致时，仍以 logits 选出的 response 动作为准。
    body = display_model_output(text).removeprefix(SILENCE_TOKEN).strip()
    return RESPONSE_TOKEN if not body else f"{RESPONSE_TOKEN} {body}"


def extract_response_payload(text: str) -> Optional[str]:
    normalized = normalize_model_output(text)
    if not normalized.startswith("</response>"):
        return None
    payload = normalized[len("</response>"):].strip()
    return payload or None


def sanitize_output_name(name: str, max_len: int = 120) -> str:
    safe_chars = []
    for ch in str(name or ""):
        is_ascii_alnum = ("a" <= ch <= "z") or ("A" <= ch <= "Z") or ("0" <= ch <= "9")
        safe_chars.append(ch if is_ascii_alnum or ch in ("-", "_", ".") else "_")
    safe_name = "".join(safe_chars).strip("._")
    return (safe_name or "live_adapter")[:max_len]


def derive_model_output_name(model_path: str) -> str:
    normalized = os.path.normpath(str(model_path or "model"))
    model_name = os.path.basename(normalized) or "model"
    parent_name = os.path.basename(os.path.dirname(normalized))
    if model_name.startswith("checkpoint-") and parent_name:
        model_name = f"{parent_name}__{model_name}"
    return sanitize_output_name(model_name)


def resolve_save_dir(path: Optional[str], root: str = DEFAULT_SAVE_ROOT) -> Optional[str]:
    if path is None:
        return None
    path = str(path).strip()
    if not path:
        return None
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(root, path))


def derive_light_out_dir(out_dir: str) -> str:
    normalized_out_dir = os.path.normpath(out_dir)
    parent_dir = os.path.dirname(normalized_out_dir)
    base_name = os.path.basename(normalized_out_dir)
    if base_name.startswith("output_"):
        return os.path.join(parent_dir, f"output_light_{base_name[len('output_'):]}")
    if base_name == "output":
        return os.path.join(parent_dir, "output_light")
    return normalized_out_dir + "_light"


def build_model_input_record(
    chunk_index: int,
    messages: list[dict[str, Any]],
    frame_count: int,
    model: Optional[str] = None,
    generation_kwargs: Optional[dict[str, Any]] = None,
    inference_skipped: bool = False,
    skip_reason: Optional[str] = None,
    image_paths: Optional[list[str]] = None,
    frame_time_ranges: Optional[list[str]] = None,
    prefix_content: Optional[str] = None,
    prompt: Optional[str] = None,
) -> dict[str, Any]:
    if inference_skipped:
        return {
            "http_payload_skipped": True,
            "inference_skipped": True,
            "skip_reason": skip_reason,
            "would_be_messages": list(messages),
        }

    del chunk_index, frame_count, image_paths, frame_time_ranges, prefix_content, prompt
    record = {
        "model": model,
        "messages": list(messages),
    }
    if generation_kwargs:
        record.update(copy.deepcopy(generation_kwargs))
    return record


def compact_image_urls_for_record(
    obj: Any,
    image_paths: Optional[list[str]] = None,
) -> Any:
    """Copy a debug/output record without retaining inline image bytes.

    Camera requests carry JPEG frames as large base64 data URLs. Keeping even
    one such request per prediction makes a live session's RSS grow linearly;
    keeping the cumulative main-model input makes it grow quadratically within
    a chunk. Frames have already been persisted by ``_resolve_frame_ref``, so
    records can safely point at those files instead.
    """
    resolved_paths = iter(image_paths or [])

    def _compact(node: Any) -> Any:
        if isinstance(node, list):
            return [_compact(item) for item in node]
        if isinstance(node, dict):
            if node.get("type") == "image_url":
                compacted = {key: _compact(value) for key, value in node.items() if key != "image_url"}
                image_url = node.get("image_url")
                details = dict(image_url) if isinstance(image_url, dict) else {}
                raw_url = details.get("url") if details else image_url
                replacement = next(resolved_paths, None)
                if replacement:
                    details["url"] = Path(replacement).resolve().as_uri()
                elif isinstance(raw_url, str) and raw_url.startswith("data:image/"):
                    details["url"] = "<inline image omitted>"
                elif raw_url is not None:
                    details["url"] = raw_url
                compacted["image_url"] = details
                return compacted
            return {key: _compact(value) for key, value in node.items()}
        if isinstance(node, str) and node.startswith("data:image/"):
            replacement = next(resolved_paths, None)
            return Path(replacement).resolve().as_uri() if replacement else "<inline image omitted>"
        return node

    return _compact(obj)


def archive_chunk_response_records(
    current_chunk: dict[str, Any],
    memory_state: dict[str, Any],
    current_query_text: Optional[str],
    query_start_time: Optional[str],
    chunk_index: int = 0,
    before_time_sec: float = float("inf"),
) -> None:
    # 把当前问题在本窗口内得到的 response 归档进 memory_state["qa_history"]（同一问题同一窗口则合并）。
    if not current_chunk["response_records"] or not current_query_text:
        return

    query_start_sec = _parse_start_second(query_start_time)
    valid_records = [
        (time_range, payload)
        for time_range, payload in current_chunk["response_records"]
        if query_start_sec <= _parse_start_second(time_range) < before_time_sec
    ]
    if not valid_records:
        return

    existing = None
    for entry in memory_state["qa_history"]:
        if (
            entry["query"] == current_query_text
            and entry.get("archived_in_chunk") == chunk_index
            and entry.get("query_time") == query_start_time
        ):
            existing = entry
            break
    if existing:
        existing["responses"].extend(valid_records)
        existing["archived_in_chunk"] = chunk_index
    else:
        memory_state["qa_history"].append(
            {
                "query_time": query_start_time,
                "query": current_query_text,
                "responses": list(valid_records),
                "archived_in_chunk": chunk_index,
            }
        )
