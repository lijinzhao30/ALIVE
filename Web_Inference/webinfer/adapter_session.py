# -*- coding: utf-8 -*-
"""会话状态与会话级 I/O。

一个浏览器会话 = 一个 SessionState：
  - current_chunk：当前视觉窗口（最近 <= chunk 帧的 messages / 帧路径 / 时间戳）；
  - sparse_text_messages：滑出窗口的稀疏问答文本（旧帧图片不再发送）；
  - predictions：每一轮的输入/输出记录，会话结束或每轮异步写到 save_root 下的 output_* 目录。
SessionMixin 负责：创建/过期/淘汰会话、把 WebUI 发来的 base64 帧落盘、写结果 JSON 与调试快照。
"""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from aiohttp import web

from adapter_common import (
    LOGGER,
    archive_chunk_response_records,
    derive_light_out_dir,
    reset_chunk_state,
    sanitize_output_name,
)
from adapter_request import _file_to_data_url_cached, _safe_session_id
from adapter_image_cache import ImageDataUrlCache


@dataclass
class SessionState:
    session_id: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    image_cache: ImageDataUrlCache = field(default_factory=ImageDataUrlCache, repr=False)
    frame_count: int = 0
    turn_count: int = 0
    chunk_index: int = 1
    current_chunk: dict[str, Any] = field(default_factory=reset_chunk_state)
    sparse_text_messages: list[dict[str, str]] = field(default_factory=list)
    memory_state: dict[str, Any] = field(
        default_factory=lambda: {"long_term_memory": "", "qa_history": []}
    )
    current_query_text: Optional[str] = None
    query_start_time: Optional[str] = None
    query_in_current_chunk: bool = False
    mid_term_summaries: list[dict[str, Any]] = field(default_factory=list)
    mid_term_history: list[dict[str, Any]] = field(default_factory=list)
    long_term_history: list[dict[str, Any]] = field(default_factory=list)
    long_term_compression_next_index: int = 1
    async_summary_segment: dict[str, Any] = field(default_factory=reset_chunk_state)
    async_next_summary_target_turns: int = 0
    async_next_summary_index: int = 1
    async_pending_summary_jobs: list[dict[str, Any]] = field(default_factory=list)
    predictions: list[dict[str, Any]] = field(default_factory=list)
    session_started_at: float = field(default_factory=time.time)
    output_path: Optional[Path] = None
    light_output_path: Optional[Path] = None
    debug_input_dir: Optional[Path] = None
    session_out_dir: Optional[str] = None
    session_light_out_dir: Optional[str] = None
    session_frame_dir: Optional[Path] = None
    session_frame_counter: int = 0
    chunk_start_input_saved: set[int] = field(default_factory=set)
    last_access: float = field(default_factory=time.time)
    _pending_qa_archive: Optional[tuple[str, Optional[str]]] = field(default=None, repr=False)
    _pending_write_task: Optional[asyncio.Task] = field(default=None, repr=False)


class SessionMixin:
    """StreamingInferAdapter 的会话管理与落盘部分（依赖 self.config / self.sessions）。"""

    # ==================== 会话生命周期 ====================

    def get_session(self, session_id: str) -> SessionState:
        session_id = _safe_session_id(session_id or "default")
        state = self.sessions.get(session_id)
        if state is None:
            self._cleanup_expired_sessions()
            self._enforce_max_active_sessions(incoming_session_id=session_id)
            state = SessionState(session_id=session_id)
            if self.config.per_session_dirs and self.config.save_root:
                self._init_session_dirs(state)
            state.output_path = self._session_output_path(state, light=False)
            state.light_output_path = self._session_output_path(state, light=True)
            state.debug_input_dir = self._session_debug_input_dir(state)
            state.async_next_summary_target_turns = self._async_first_summary_turns()
            # Per-session frame directory
            frame_dir = Path(self.config.frame_save_dir) / session_id
            frame_dir.mkdir(parents=True, exist_ok=True)
            state.session_frame_dir = frame_dir
            self.sessions[session_id] = state
            LOGGER.info(
                "Created session %s (output=%s light=%s debug_input=%s frames=%s)",
                session_id,
                state.output_path,
                state.light_output_path,
                state.debug_input_dir,
                state.session_frame_dir,
            )
        state.last_access = time.time()
        return state

    def _drop_session_state(self, session_id: str, state: SessionState, reason: str) -> None:
        state.image_cache.clear()
        for job in state.async_pending_summary_jobs:
            job["task"].cancel()
        state.current_chunk = reset_chunk_state()
        state.sparse_text_messages.clear()
        state.async_summary_segment = reset_chunk_state()
        state.async_pending_summary_jobs.clear()
        state.predictions.clear()
        state.mid_term_summaries.clear()
        state.mid_term_history.clear()
        state.long_term_history.clear()
        state.memory_state = {"long_term_memory": "", "qa_history": []}
        _file_to_data_url_cached.cache_clear()
        LOGGER.info("Dropped session %s from memory (%s)", session_id, reason)

    def _enforce_max_active_sessions(self, incoming_session_id: str = "") -> None:
        limit = int(self.config.max_active_sessions or 0)
        if limit <= 0:
            return
        while len(self.sessions) >= limit:
            candidates = [
                (sid, state)
                for sid, state in self.sessions.items()
                if sid != incoming_session_id
            ]
            if not candidates:
                return
            sid, state = min(candidates, key=lambda item: item[1].last_access)
            self.sessions.pop(sid, None)
            self._drop_session_state(sid, state, "max_active_sessions")

    def _cleanup_expired_sessions(self) -> list[SessionState]:
        now = time.time()
        timeout = self.config.session_timeout_seconds
        expired = [
            sid for sid, s in self.sessions.items()
            if now - s.last_access > timeout
        ]
        expired_states = []
        for sid in expired:
            state = self.sessions.pop(sid, None)
            if state is not None:
                self._drop_session_state(sid, state, "session_timeout")
                expired_states.append(state)
                LOGGER.info("Expired session %s (idle %.0fs)", sid, now - state.last_access)
        return expired_states

    async def _session_cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(300)
            try:
                expired_states = self._cleanup_expired_sessions()
                for state in expired_states:
                    await self._flush_session_outputs(state)
            except Exception:
                LOGGER.exception("session cleanup error")

    def start_background_tasks(self) -> None:
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.ensure_future(self._session_cleanup_loop())

    def _init_session_dirs(self, state: SessionState) -> None:
        """Create per-session timestamped output/input directories."""
        session_ts = datetime.fromtimestamp(state.session_started_at).strftime("%Y%m%d_%H%M%S")
        model_name = self.config.output_model_name
        save_root = self.config.save_root

        state.session_out_dir = os.path.join(save_root, f"output_{session_ts}_{model_name}")
        state.session_light_out_dir = derive_light_out_dir(state.session_out_dir)

        Path(state.session_out_dir).mkdir(parents=True, exist_ok=True)
        Path(state.session_light_out_dir).mkdir(parents=True, exist_ok=True)

        if self.config.save_debug_inputs:
            state.debug_input_dir = Path(os.path.join(save_root, f"input_{session_ts}_{model_name}"))
            state.debug_input_dir.mkdir(parents=True, exist_ok=True)

    # ==================== 结果 / 调试文件写入 ====================

    def _session_output_path(self, state: SessionState, light: bool) -> Optional[Path]:
        if light:
            root = state.session_light_out_dir or self.config.light_out_dir
        else:
            root = state.session_out_dir or self.config.out_dir
        if not root:
            return None
        safe_session = sanitize_output_name(state.session_id)
        return Path(root) / "live" / f"{safe_session}.json"

    def _session_debug_input_dir(self, state: SessionState) -> Optional[Path]:
        if state.debug_input_dir:
            return state.debug_input_dir
        if not self.config.debug_input_dir:
            return None
        return Path(self.config.debug_input_dir)

    def _session_sample_data(self, state: SessionState) -> dict[str, Any]:
        return {
            "task_type": "live",
            "session_id": state.session_id,
            "adapter_model": self.config.adapter_model,
            "main_model": self.config.main_model,
            "main_api_base": self.config.main_api_base,
            "summarizer_model": self.config.summarizer_model,
            "summarizer_api_base": self.config.summarizer_api_base,
            "longterm_model": self.config.longterm_model,
            "longterm_api_base": self.config.longterm_api_base,
            "started_at": datetime.fromtimestamp(state.session_started_at).isoformat(
                timespec="seconds"
            ),
        }

    def _memory_trace(self, state: SessionState) -> dict[str, Any]:
        return {
            "mid_term_summaries": list(state.mid_term_history),
            "long_term_history": list(state.long_term_history),
            "qa_history": list(state.memory_state.get("qa_history", [])),
            "long_term_memory": state.memory_state.get("long_term_memory", ""),
        }

    def _write_json_file(self, path: Path, obj: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as file_obj:
            json.dump(obj, file_obj, ensure_ascii=False, indent=2)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(tmp_path, path)

    def _light_predictions(
        self,
        predictions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        light_keys = (
            "turn",
            "time_range",
            "query",
            "prediction",
            "action_prediction",
            "first_action_decision",
            "total_time",
            "inference_time",
            "fourb_mid_term_inference_time",
            "fourb_long_term_inference_time",
            "ground_truth",
        )
        return [
            {key: prediction[key] for key in light_keys if key in prediction}
            for prediction in predictions
        ]

    @staticmethod
    def _strip_base64_images(obj: Any) -> tuple[Any, dict[str, str]]:
        """Recursively strip inline base64 image data from an object.

        Returns a (stripped_obj, images_dict) tuple where images_dict maps
        placeholder keys to the original base64 strings.
        """
        images: dict[str, str] = {}
        counter = [0]

        def _strip(node: Any) -> Any:
            if isinstance(node, str):
                if node.startswith("data:image/") and len(node) > 200:
                    key = f"__image_{counter[0]}__"
                    counter[0] += 1
                    images[key] = node
                    return key
                return node
            if isinstance(node, list):
                return [_strip(item) for item in node]
            if isinstance(node, dict):
                return {k: _strip(v) for k, v in node.items()}
            return node

        stripped = _strip(obj)
        return stripped, images

    def _write_session_outputs_sync(
        self,
        output_path: Optional[Path],
        light_output_path: Optional[Path],
        full_result: Optional[dict[str, Any]],
        light_result: Optional[dict[str, Any]],
    ) -> None:
        if light_output_path and light_result:
            self._write_json_file(light_output_path, light_result)
        if output_path and full_result:
            stripped_result, images = self._strip_base64_images(full_result)
            self._write_json_file(output_path, stripped_result)
            if images:
                images_path = output_path.with_suffix(".images.json")
                self._write_json_file(images_path, images)

    def _write_session_outputs(self, state: SessionState) -> None:
        total_time = time.time() - state.session_started_at
        output_path = state.output_path
        light_output_path = state.light_output_path
        if not output_path and not light_output_path:
            return
        predictions_snapshot = copy.deepcopy(state.predictions)
        sample_data = self._session_sample_data(state)
        memory_trace = copy.deepcopy(self._memory_trace(state))
        full_result = None
        light_result = None
        if output_path:
            full_result = {
                "sample_data": sample_data,
                "total_time": total_time,
                "total_turns": len(predictions_snapshot),
                "predictions": predictions_snapshot,
                "memory": memory_trace,
            }
        if light_output_path:
            light_result = {
                "sample_data": sample_data,
                "total_time": total_time,
                "total_turns": len(predictions_snapshot),
                "predictions": self._light_predictions(predictions_snapshot),
                "memory": memory_trace,
            }
        if state._pending_write_task and not state._pending_write_task.done():
            state._pending_write_task.cancel()
        task = asyncio.ensure_future(
            asyncio.to_thread(
                self._write_session_outputs_sync,
                output_path,
                light_output_path,
                full_result,
                light_result,
            )
        )
        task.add_done_callback(self._on_write_task_done)
        state._pending_write_task = task

    @staticmethod
    def _on_write_task_done(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            LOGGER.error("session output write failed: %s", exc, exc_info=exc)

    async def _flush_session_outputs(self, state: SessionState) -> None:
        """Write final session outputs synchronously at session end."""
        self._execute_pending_qa_archive(state)
        if self.config.keep_qa_history and state.current_query_text:
            archive_chunk_response_records(
                state.current_chunk,
                state.memory_state,
                state.current_query_text,
                state.query_start_time,
                chunk_index=state.chunk_index,
            )
        total_time = time.time() - state.session_started_at
        output_path = state.output_path
        light_output_path = state.light_output_path
        if not output_path and not light_output_path:
            return
        if not state.predictions:
            return
        sample_data = self._session_sample_data(state)
        memory_trace = self._memory_trace(state)
        full_result = None
        light_result = None
        if output_path:
            full_result = {
                "sample_data": sample_data,
                "total_time": total_time,
                "total_turns": len(state.predictions),
                "predictions": state.predictions,
                "memory": memory_trace,
            }
        if light_output_path:
            light_result = {
                "sample_data": sample_data,
                "total_time": total_time,
                "total_turns": len(state.predictions),
                "predictions": self._light_predictions(state.predictions),
                "memory": memory_trace,
            }
        await asyncio.to_thread(
            self._write_session_outputs_sync,
            output_path,
            light_output_path,
            full_result,
            light_result,
        )
        LOGGER.info(
            "[%s] final session output written (%d turns)",
            state.session_id,
            len(state.predictions),
        )

    def _save_live_debug_input(
        self,
        state: SessionState,
        record: dict[str, Any],
        stem: str,
    ) -> Optional[str]:
        debug_dir = state.debug_input_dir or self.config.debug_input_dir
        if not debug_dir:
            return None
        path = (
            Path(debug_dir)
            / f"{sanitize_output_name(state.session_id)}__{stem}.json"
        )
        record = copy.deepcopy(record)
        record.setdefault("saved_at", datetime.now().isoformat(timespec="seconds"))
        record.setdefault("session_id", state.session_id)
        self._write_json_file(path, record)
        return str(path)

    def _maybe_save_chunk_start_model_input(
        self,
        state: SessionState,
        turn_count: int,
        time_range: str,
        model_input_record: dict[str, Any],
    ) -> Optional[str]:
        if not (state.debug_input_dir or self.config.debug_input_dir):
            return None
        if state.chunk_index in state.chunk_start_input_saved:
            return None
        record = copy.deepcopy(model_input_record)
        record["stage"] = "main_8b_chunk_start"
        record["turn"] = turn_count
        record["time_range"] = time_range
        path = self._save_live_debug_input(
            state,
            record,
            f"chunk_{state.chunk_index:04d}__turn_{turn_count:04d}",
        )
        state.chunk_start_input_saved.add(state.chunk_index)
        return path

    def _save_summarizer_debug_input(
        self,
        state: SessionState,
        stage: str,
        index: int,
        record: Optional[dict[str, Any]],
    ) -> Optional[str]:
        if not record:
            return None
        return self._save_live_debug_input(
            state,
            record,
            f"{stage}__{index:04d}",
        )

    # ==================== 帧落盘 ====================

    def _resolve_frame_ref(
        self,
        image_ref: dict[str, str],
        state: "SessionState",
    ) -> str:
        if image_ref.get("kind") == "path":
            return str(self._validate_local_image_path(image_ref.get("value", "")))
        if image_ref.get("kind") == "data_url":
            return self._save_base64_frame(image_ref.get("value", ""), state)
        raise web.HTTPBadRequest(text="unsupported image reference kind")

    def _save_base64_frame(self, data_url: str, state: "SessionState") -> str:
        # WebUI 的每帧 base64 JPEG 先落盘为 frame_000001.jpg ...，之后会话内只保存路径。
        match = re.match(r"^data:(image/[\w.+-]+);base64,(.+)$", data_url, re.DOTALL)
        if not match:
            raise web.HTTPBadRequest(text="invalid data URL format")
        state.session_frame_counter += 1
        mime_type, encoded = match.groups()
        suffix = {
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/bmp": ".bmp",
        }.get(mime_type.lower(), ".jpg")
        frame_dir = state.session_frame_dir or (Path(self.config.frame_save_dir) / state.session_id)
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frame_dir / f"frame_{state.session_frame_counter:06d}{suffix}"
        try:
            frame_path.write_bytes(base64.b64decode(encoded))
        except Exception as exc:
            raise web.HTTPBadRequest(text=f"invalid base64 image data: {exc}") from exc
        return str(frame_path)

    def _validate_local_image_path(self, raw_path: str) -> Path:
        if not self.config.allowed_local_image_roots:
            raise web.HTTPBadRequest(text="local image paths are disabled")

        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise web.HTTPBadRequest(text=f"local image path does not exist: {path}")
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            raise web.HTTPBadRequest(text=f"unsupported local image extension: {path.suffix}")

        for root in self.config.allowed_local_image_roots:
            root_path = Path(root).expanduser().resolve()
            try:
                path.relative_to(root_path)
                return path
            except ValueError:
                continue

        allowed = ", ".join(self.config.allowed_local_image_roots)
        raise web.HTTPBadRequest(text=f"local image path is outside allowed roots: {allowed}")
