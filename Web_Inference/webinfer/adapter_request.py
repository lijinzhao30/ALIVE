# -*- coding: utf-8 -*-
"""HTTP 请求解析、图片编码与 OpenAI 响应构造。

数据流：
  WebUI 每秒发来一个 OpenAI chat 请求（messages 里带一张 base64 JPEG + 可选的用户问题）
    -> 这里解析出 session_id / 图片引用 / 时间戳 / 用户问题；
  adapter 把内部 {"type": "image", "image": 本地路径} 消息转换回 OpenAI 格式时，
    -> 这里读取图片、按短边 480 + max_pixels 等比缩放后重新编码成 data URL 发给 vLLM；
  推理结束后
    -> 这里把结果包装成标准 chat.completion 响应（附带 streamingharness 调试字段）。
"""

from __future__ import annotations

import base64
import functools
import io
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from aiohttp import web
from PIL import Image

from adapter_common import LOGGER, TIME_RANGE_RE, _normalize_time_range_text
from adapter_image_cache import ImageDataUrlCache


def _safe_session_id(session_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(session_id or "default"))
    return safe.strip("._")[:120] or "default"


def _request_session_id(request: web.Request, payload: dict[str, Any]) -> str:
    # WebUI 在 header 里带 x-streaming-session，每个浏览器会话对应 adapter 里一个独立的流式状态。
    return (
        request.headers.get("x-streaming-session")
        or request.headers.get("x-session-id")
        or str(payload.get("user") or "")
        or "default"
    )


def _extract_time_range_from_request(
    request: web.Request,
    payload: dict[str, Any],
) -> Optional[str]:
    candidates = (
        request.headers.get("x-frame-time-range"),
        request.headers.get("x-streaming-time-range"),
        str(payload.get("x_frame_time_range") or ""),
        str(payload.get("frame_time_range") or ""),
    )
    for candidate in candidates:
        normalized = _normalize_time_range_text(candidate)
        if normalized:
            return normalized
    return None


async def _read_json(request: web.Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise web.HTTPBadRequest(text=f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(text="JSON body must be an object")
    return payload


def _extract_first_image_ref(
    messages: list[dict[str, Any]],
    request: web.Request,
    payload: dict[str, Any],
) -> Optional[dict[str, str]]:
    local_path = _extract_local_image_path_from_request(request, payload)
    if local_path:
        return {"kind": "path", "value": local_path}

    for message in reversed(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image_url":
                continue
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url")
            else:
                url = image_url
            if isinstance(url, str) and url:
                file_path = _file_url_to_path(url)
                if file_path:
                    return {"kind": "path", "value": file_path}
                return {"kind": "data_url", "value": url}
    return None


def _extract_all_image_refs(
    messages: list[dict[str, Any]],
    request: web.Request,
    payload: dict[str, Any],
) -> list[dict[str, str]]:
    """Extract all image references from the request (supports batch frames)."""
    refs: list[dict[str, str]] = []

    local_path = _extract_local_image_path_from_request(request, payload)
    if local_path:
        refs.append({"kind": "path", "value": local_path})

    for message in reversed(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image_url":
                continue
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url")
            else:
                url = image_url
            if isinstance(url, str) and url:
                file_path = _file_url_to_path(url)
                if file_path:
                    refs.append({"kind": "path", "value": file_path})
                else:
                    refs.append({"kind": "data_url", "value": url})
        if refs:
            break
    return refs


def _extract_time_ranges_from_request(
    request: web.Request,
    payload: dict[str, Any],
) -> list[str]:
    """Extract multiple time ranges from request (for batch frames)."""
    ranges = payload.get("frame_time_ranges")
    if isinstance(ranges, list) and ranges:
        parsed: list[str] = []
        for r in ranges:
            normalized = _normalize_time_range_text(r)
            if normalized:
                parsed.append(normalized)
        if parsed:
            return parsed

    single = _extract_time_range_from_request(request, payload)
    return [single] if single else []


def _extract_local_image_path_from_request(
    request: web.Request,
    payload: dict[str, Any],
) -> Optional[str]:
    candidates = (
        request.headers.get("x-local-image-path"),
        request.headers.get("x-frame-image-path"),
        str(payload.get("x_local_image_path") or ""),
        str(payload.get("local_image_path") or ""),
        str(payload.get("frame_image_path") or ""),
    )
    for candidate in candidates:
        candidate = (candidate or "").strip()
        if candidate:
            return candidate
    return None


def _file_url_to_path(url: str) -> Optional[str]:
    if not url.startswith("file://"):
        return None
    from urllib.parse import unquote, urlparse

    parsed = urlparse(url)
    if parsed.netloc not in {"", "localhost"}:
        return None
    return unquote(parsed.path)


def _extract_user_prompt_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts = [
                str(item.get("text", "")).strip()
                for item in content
                if isinstance(item, dict)
                and item.get("type") == "text"
                and str(item.get("text", "")).strip()
            ]
            return "\n".join(text_parts).strip()
    return ""


def _extract_time_range_from_text(text: str) -> Optional[str]:
    return _normalize_time_range_text(text)


def _strip_time_range_from_text(text: str) -> str:
    lines = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if TIME_RANGE_RE.fullmatch(stripped):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _format_seconds(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return f"{int(round(value))}s"
    return f"{value:.3f}".rstrip("0").rstrip(".") + "s"


def _internal_message_has_image(message: dict[str, Any]) -> bool:
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(item, dict) and item.get("type") == "image" for item in content)


def _internal_message_to_openai(
    message: dict[str, Any], image_cache: Optional[ImageDataUrlCache] = None,
) -> dict[str, Any]:
    # History stores paths; only the bounded session cache retains encoded frames.
    content = message.get("content")
    if not isinstance(content, list):
        return {"role": message.get("role", "user"), "content": content or ""}

    converted: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "image":
            max_pixels = int(item.get("max_pixels") or 0)
            resize_short_edge = int(item.get("resize_short_edge") or 0)
            converted.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _file_to_data_url(
                        item["image"],
                        max_pixels=max_pixels,
                        resize_short_edge=resize_short_edge,
                        image_cache=image_cache,
                    )},
                }
            )
        elif item.get("type") == "text":
            converted.append({"type": "text", "text": str(item.get("text", ""))})
        else:
            converted.append(item)
    return {"role": message.get("role", "user"), "content": converted}


def _file_to_data_url(
    path: str,
    max_pixels: int = 0,
    resize_short_edge: int = 0,
    image_cache: Optional[ImageDataUrlCache] = None,
) -> str:
    if path.startswith("data:"):
        return _resize_data_url_if_needed(path, max_pixels, resize_short_edge)
    if image_cache is not None:
        return image_cache.encode(path, max_pixels, resize_short_edge, _encode_file_to_data_url)
    return _file_to_data_url_cached(path, max_pixels, resize_short_edge)


@functools.lru_cache(maxsize=128)
def _file_to_data_url_cached(
    path: str,
    max_pixels: int = 0,
    resize_short_edge: int = 0,
) -> str:
    # Legacy callers retain only a temporary cache, cleared after model calls.
    return _encode_file_to_data_url(path, max_pixels, resize_short_edge)


def _encode_file_to_data_url(path: str, max_pixels: int, resize_short_edge: int) -> str:
    ext = Path(path).suffix.lower()
    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
    }.get(ext, "image/jpeg")

    if max_pixels > 0 or resize_short_edge > 0:
        try:
            with Image.open(path) as image:
                image.load()
                resized = _resize_image_if_needed(image, max_pixels, resize_short_edge)
                if resized is not None:
                    return _image_to_data_url(resized, mime_type)
        except Exception as exc:
            LOGGER.warning(
                "failed to resize image %s for max_pixels=%s short_edge=%s: %s",
                path, max_pixels, resize_short_edge, exc,
            )

    with open(path, "rb") as file_obj:
        encoded = base64.b64encode(file_obj.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _resize_data_url_if_needed(
    data_url: str,
    max_pixels: int = 0,
    resize_short_edge: int = 0,
) -> str:
    if max_pixels <= 0 and resize_short_edge <= 0:
        return data_url
    match = re.match(r"^data:(image/[^;]+);base64,(.+)$", data_url, re.DOTALL)
    if not match:
        return data_url
    mime_type, encoded = match.groups()
    try:
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            image.load()
            resized = _resize_image_if_needed(image, max_pixels, resize_short_edge)
            if resized is None:
                return data_url
            return _image_to_data_url(resized, mime_type)
    except Exception as exc:
        LOGGER.warning(
            "failed to resize data URL image for max_pixels=%s short_edge=%s: %s",
            max_pixels, resize_short_edge, exc,
        )
        return data_url


def _resize_image_if_needed(
    image: Image.Image,
    max_pixels: int,
    resize_short_edge: int = 0,
) -> Optional[Image.Image]:
    # 仅当短边超过上限时等比缩小；小图不放大。随后按 max_pixels 限制总像素。
    width, height = image.size
    new_width, new_height = width, height
    if resize_short_edge > 0 and min(width, height) > resize_short_edge:
        scale = resize_short_edge / min(width, height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
    if max_pixels > 0 and new_width * new_height > max_pixels:
        scale = (max_pixels / (new_width * new_height)) ** 0.5
        new_width = max(1, int(new_width * scale))
        new_height = max(1, int(new_height * scale))
    if (new_width, new_height) == (width, height):
        return None
    return image.resize((new_width, new_height), Image.LANCZOS)


def _image_to_data_url(image: Image.Image, mime_type: str) -> str:
    buffer = io.BytesIO()
    if mime_type == "image/png":
        image.save(buffer, format="PNG")
    elif mime_type == "image/webp":
        image.save(buffer, format="WEBP")
    else:
        mime_type = "image/jpeg"
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        image.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _extract_extra_body(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "skip_special_tokens",
        "top_k",
        "repetition_penalty",
        "min_p",
        "stop_token_ids",
        "include_stop_str_in_output",
    )
    return {key: payload[key] for key in keys if key in payload}


def _chat_completion_response(
    model: str,
    content: str,
    usage: Optional[dict[str, Any]],
    raw_model: str,
    raw_text: str,
) -> dict[str, Any]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    response = {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": usage or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
    response["streamingharness"] = {
        "main_model": raw_model,
        "raw_content": raw_text,
    }
    return response


def _openai_error_response(message: str, status: int) -> web.Response:
    return web.json_response(
        {
            "error": {
                "message": message,
                "type": "streaming_infer_adapter_error",
                "param": None,
                "code": None,
            }
        },
        status=status,
    )


def _short(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
