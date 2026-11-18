# -*- coding: utf-8 -*-
"""StreamingInferAdapter 主体：HTTP 接口 + 每帧推理主流程。

每个 /v1/chat/completions 请求（= WebUI 送来的一帧）在 _handle_chat_payload 中依次完成：
  1. 解析图片 / 用户问题 / 时间戳，帧落盘；有新问题则更新 current_query_text；
  2. 窗口满 chunk 帧时裁剪到最近 retain_frames 帧（或 summarizer 模式下刷新窗口并做摘要）；
  3. 追加 user 消息；若还没有任何问题且 force_silence_before_query，直接输出 </silence> 不调模型；
  4. 否则构造 [system + 稀疏历史 + 窗口消息] 调 vLLM，归一化输出为 </silence> 或 </response> ...；
  5. 把 assistant 输出写回窗口历史，记录 prediction，返回 OpenAI 格式响应给 WebUI。
类方法按功能拆在三个 mixin 中：SessionMixin(会话/落盘)、ModelMixin(prompt/调模型)、SummaryMixin(摘要)。
"""

from __future__ import annotations

import asyncio
import copy
import time
from pathlib import Path
from typing import Any, Optional

from aiohttp import web
from openai import AsyncOpenAI

from adapter_common import (
    LOGGER,
    _format_turn_time_range,
    _parse_start_second,
    action_history_output,
    archive_chunk_response_records,
    build_model_input_record,
    compact_image_urls_for_record,
    display_model_output,
    extract_response_payload,
    normalize_model_output,
    reset_chunk_state,
)
from adapter_config import AdapterConfig
from adapter_model import ModelMixin
from adapter_identity import identity_action, is_identity_question
from adapter_request import (
    _chat_completion_response,
    _extract_all_image_refs,
    _extract_time_range_from_request,
    _extract_time_range_from_text,
    _extract_time_ranges_from_request,
    _extract_user_prompt_text,
    _openai_error_response,
    _read_json,
    _request_session_id,
    _safe_session_id,
    _short,
    _strip_time_range_from_text,
)
from adapter_session import SessionMixin, SessionState
from adapter_summary import SummaryMixin
from memory_summarizer import SummarizerModel


class StreamingInferAdapter(SessionMixin, ModelMixin, SummaryMixin):
    def __init__(self, config: AdapterConfig):
        self.config = config
        self.sessions: dict[str, SessionState] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        Path(config.frame_save_dir).mkdir(parents=True, exist_ok=True)
        self.main_client = AsyncOpenAI(
            base_url=config.main_api_base,
            api_key=config.api_key,
            timeout=config.request_timeout_seconds,
        )
        self.main_clients: dict[str, tuple[AsyncOpenAI, str]] = {}
        if config.main_backends:
            for backend in config.main_backends:
                name = backend["name"]
                self.main_clients[name] = (
                    AsyncOpenAI(
                        base_url=backend["api_base"],
                        api_key=config.api_key,
                        timeout=config.request_timeout_seconds,
                    ),
                    backend.get("model", name),
                )
        else:
            self.main_clients[config.main_model] = (self.main_client, config.main_model)
        self.summarizer: Optional[SummarizerModel] = None
        if config.enable_summarizer:
            self.summarizer = SummarizerModel(
                model_name=config.summarizer_model,
                api_base=config.summarizer_api_base,
                longterm_model_name=config.longterm_model,
                longterm_api_base=config.longterm_api_base,
                mid_term_max_tokens=config.mid_term_max_tokens,
                mid_term_target_tokens=config.mid_term_target_tokens,
                long_term_max_tokens=config.long_term_max_tokens,
                long_term_target_tokens=config.long_term_target_tokens,
                key_frames_per_chunk=config.summarizer_key_frames,
                max_pixels=config.summarizer_max_pixels,
                prompt_phase_seconds=config.summarizer_phase_seconds,
                mid_term_temperature=config.mid_term_temperature,
                mid_term_top_p=config.mid_term_top_p,
                mid_term_top_k=config.mid_term_top_k,
                mid_term_repetition_penalty=config.mid_term_repetition_penalty,
                mid_term_presence_penalty=config.mid_term_presence_penalty,
                long_term_temperature=config.long_term_temperature,
                long_term_top_p=config.long_term_top_p,
                long_term_top_k=config.long_term_top_k,
                long_term_repetition_penalty=config.long_term_repetition_penalty,
                long_term_presence_penalty=config.long_term_presence_penalty,
                debug=config.summarizer_debug,
            )
        if config.out_dir:
            Path(config.out_dir).mkdir(parents=True, exist_ok=True)
        if config.light_out_dir:
            Path(config.light_out_dir).mkdir(parents=True, exist_ok=True)
        if config.debug_input_dir:
            Path(config.debug_input_dir).mkdir(parents=True, exist_ok=True)

    def _resolve_backend(self, model_name: Optional[str] = None) -> tuple[AsyncOpenAI, str]:
        if model_name and model_name in self.main_clients:
            return self.main_clients[model_name]
        return self.main_client, self.config.main_model

    # ==================== HTTP 接口 ====================

    async def handle_models(self, request: web.Request) -> web.Response:
        del request
        now = int(time.time())
        data = [
            {
                "id": name,
                "object": "model",
                "created": now,
                "owned_by": "streamingharness",
            }
            for name in self.main_clients
        ]
        return web.json_response({"object": "list", "data": data})

    async def handle_health(self, request: web.Request) -> web.Response:
        del request
        return web.json_response(
            {
                "ok": True,
                "model": self.config.adapter_model,
                "backends": list(self.main_clients.keys()),
                "summarizer_enabled": self.summarizer is not None,
                "sessions": len(self.sessions),
            }
        )

    async def handle_reset(self, request: web.Request) -> web.Response:
        payload = await _read_json(request)
        session_id = _request_session_id(request, payload)
        session_id = _safe_session_id(session_id)
        removed_state = self.sessions.pop(session_id, None)
        if removed_state is not None:
            removed_state.image_cache.clear()
            for job in removed_state.async_pending_summary_jobs:
                job["task"].cancel()
            await self._flush_session_outputs(removed_state)
        removed = removed_state is not None
        return web.json_response({"ok": True, "session_id": session_id, "removed": removed})

    async def handle_chat_completions(self, request: web.Request) -> web.Response:
        payload = await _read_json(request)
        session_id = _request_session_id(request, payload)
        requested_model = payload.get("model")
        client, model_name = self._resolve_backend(requested_model)
        state = self.get_session(session_id)
        # 同一会话的帧串行处理，保证窗口历史按时间顺序追加。
        async with state.lock:
            try:
                result = await self._handle_chat_payload(state, payload, request, client=client, model_name=model_name)
            except web.HTTPException:
                raise
            except Exception as exc:
                LOGGER.exception("chat completion failed")
                return _openai_error_response(str(exc), status=502)
        return web.json_response(result)

    # ==================== 每帧推理主流程 ====================

    async def _handle_chat_payload(
        self,
        state: SessionState,
        payload: dict[str, Any],
        request: web.Request,
        *,
        client: Optional[AsyncOpenAI] = None,
        model_name: Optional[str] = None,
    ) -> dict[str, Any]:
        client = client or self.main_client
        model_name = model_name or self.config.main_model
        t_start = time.perf_counter()
        messages = payload.get("messages") or []
        if not isinstance(messages, list):
            raise web.HTTPBadRequest(text="messages must be a list")

        # ---- 1. 解析本帧：图片引用 / 用户问题 / 时间戳 ----
        image_refs = _extract_all_image_refs(messages, request, payload)
        if not image_refs:
            return await self._forward_text_only(payload, client=client, model_name=model_name)
        if self.config.input_format != "legacy" and len(image_refs) != 1:
            raise web.HTTPBadRequest(text="training input format requires one frame per request")

        # The live counter is authoritative and does not depend on how much
        # diagnostic prediction history is retained in memory.
        turn_count = state.turn_count + 1
        raw_prompt_text = _extract_user_prompt_text(messages)
        prompt_text = _strip_time_range_from_text(raw_prompt_text)

        # Resolve time ranges for all images
        incoming_time_ranges = _extract_time_ranges_from_request(request, payload)
        if not incoming_time_ranges:
            single = _extract_time_range_from_request(request, payload)
            if single is None:
                single = _extract_time_range_from_text(raw_prompt_text)
            if single:
                incoming_time_ranges = [single]
        time_ranges: list[str] = []
        for i in range(len(image_refs)):
            if i < len(incoming_time_ranges) and incoming_time_ranges[i]:
                time_ranges.append(incoming_time_ranges[i])
            else:
                time_ranges.append(self._time_range_for_frame(state.frame_count + i))
        time_range = _format_turn_time_range(time_ranges)

        image_paths = [self._resolve_frame_ref(ref, state) for ref in image_refs]
        LOGGER.info(
            "[%s] turn=%d frames=%d(+%d) chunk=%d time=%s prompt=%r",
            state.session_id,
            turn_count,
            state.frame_count,
            len(image_refs),
            state.chunk_index,
            time_range,
            _short(prompt_text, 80),
        )

        query_text = self._update_query_state(state, prompt_text, time_ranges[0])
        # WebUI 的问题只发送一次；非空文本就是新提交，允许连续问同一句。
        frame_query_text = prompt_text or None

        await self._commit_required_async_summaries(
            state, state.turn_count, non_blocking=True,
        )

        # ---- 2. 窗口满 chunk 帧：训练格式下裁剪到最近 retain_frames 帧；legacy 下刷新窗口（可做摘要） ----
        if (
            self.config.input_format != "legacy"
            and self.config.chunk > 0
            and state.current_chunk["turn_count"] >= self.config.chunk
        ):
            self._trim_current_chunk_to_retained_frames(state)
        elif (
            self.config.chunk > 0
            and state.current_chunk["turn_count"] >= self.config.chunk
        ):
            self._execute_pending_qa_archive(state)
            carry_response_records = []
            if self.config.keep_qa_history and state.current_query_text:
                qa_cutoff = float("inf")
                if (
                    self._async_summary_enabled()
                    and state.async_summary_segment["frame_time_ranges"]
                ):
                    qa_cutoff = _parse_start_second(
                        state.async_summary_segment["frame_time_ranges"][0]
                    )
                    carry_response_records = [
                        (tr, payload)
                        for tr, payload in state.current_chunk["response_records"]
                        if _parse_start_second(tr) >= qa_cutoff
                    ]
                archive_chunk_response_records(
                    state.current_chunk,
                    state.memory_state,
                    state.current_query_text,
                    state.query_start_time,
                    chunk_index=state.chunk_index,
                    before_time_sec=qa_cutoff,
                )
            await self._flush_chunk(state, use_async_summary=self._async_summary_enabled())
            if (
                self._async_summary_enabled()
                and state.async_summary_segment["turn_count"] > 0
            ):
                carry = copy.deepcopy(state.async_summary_segment)
                carry_frames = carry["frame_count"]
                carry_turns = carry["turn_count"]
                carry["frame_count"] = 0
                carry["turn_count"] = 0
                carry["response_records"] = carry_response_records
                carry["api_msg_cache"] = []
                state.current_chunk = carry
                LOGGER.info(
                    "[%s] carried over %d unsummarized turn(s), %d frame(s) to new chunk",
                    state.session_id,
                    carry_turns,
                    carry_frames,
                )
            else:
                state.current_chunk = reset_chunk_state()
            state.chunk_index += 1
            state.query_in_current_chunk = bool(query_text)

        # ---- 3. 本帧写入窗口：帧路径 / 时间戳 / user 消息 ----
        for tr, ip in zip(time_ranges, image_paths):
            state.frame_count += 1
            state.current_chunk["image_paths"].append(str(ip))
            state.current_chunk["frame_time_ranges"].append(tr)
            state.current_chunk["summarizer_frame_cache"].append({"path": str(ip)})
            state.current_chunk["frame_count"] += 1

        state.turn_count += 1
        state.current_chunk["turn_count"] += 1

        user_message = self._build_internal_user_message(
            time_ranges=time_ranges,
            image_paths=[str(ip) for ip in image_paths],
            query_text=query_text if self.config.input_format == "legacy" else frame_query_text,
        )
        state.current_chunk["messages"].append(user_message)
        if (
            self.config.input_format != "legacy"
            and self.summarizer is None
            and self.config.keep_qa_history
        ):
            state.current_chunk["dialogue_events"].append(
                {"time_range": time_ranges[0], "query": prompt_text}
            )
        if self._async_summary_enabled():
            self._append_async_summary_user_message(
                state,
                time_ranges=time_ranges,
                image_paths=[str(ip) for ip in image_paths],
                query_text=query_text,
            )

        turn_input_record = {
            "source_message": compact_image_urls_for_record(
                messages[-1] if messages else None,
                [str(ip) for ip in image_paths],
            ),
            "vllm_message": user_message,
            "chunk_index": state.chunk_index,
            "has_image": True,
            "image_path": str(image_paths[-1]),
            "image_paths_batch": [str(ip) for ip in image_paths],
            "num_chunk_turns": state.current_chunk["turn_count"],
            "num_chunk_frames": state.current_chunk["frame_count"],
            "image_paths": list(state.current_chunk["image_paths"]),
            "frame_time_ranges": list(state.current_chunk["frame_time_ranges"]),
        }
        is_forced_silence = (
            self.config.force_silence_before_query and not state.current_query_text
        )
        inference_start = None
        inference_time = 0.0
        chunk_start_model_input_path = None
        turn_model_input_record = None
        model_input_record = None

        # ---- 4. 推理：尚无问题时强制 </silence>；否则构造 messages 调主模型 ----
        if is_forced_silence:
            generated_text = "</silence>"
            raw_text = ""
            usage = None
            turn_model_input_record = build_model_input_record(
                chunk_index=state.chunk_index,
                messages=state.current_chunk["messages"],
                frame_count=state.current_chunk["frame_count"],
                inference_skipped=True,
                skip_reason="force_silence_before_query",
                image_paths=state.current_chunk["image_paths"],
                frame_time_ranges=state.current_chunk["frame_time_ranges"],
            )
            if self.config.save_model_inputs:
                model_input_record = turn_model_input_record
        else:
            t_prompt_build_start = time.perf_counter()
            cache_hits_before = state.image_cache.hits
            cache_misses_before = state.image_cache.misses
            internal_messages, prefix_content = self._build_main_internal_messages(state)
            t_prompt_history_end = time.perf_counter()
            api_messages = self._build_cached_api_messages(state, internal_messages)
            t_prompt_images_end = time.perf_counter()
            generation_kwargs = self._main_generation_kwargs(payload)
            http_messages = self._build_main_http_messages(api_messages)
            turn_model_input_record = build_model_input_record(
                chunk_index=state.chunk_index,
                messages=compact_image_urls_for_record(
                    http_messages,
                    list(state.current_chunk["image_paths"]),
                ),
                frame_count=state.current_chunk["frame_count"],
                model=model_name,
                generation_kwargs=generation_kwargs,
                image_paths=state.current_chunk["image_paths"],
                frame_time_ranges=state.current_chunk["frame_time_ranges"],
                prefix_content=prefix_content,
            )
            if self.config.save_model_inputs:
                model_input_record = turn_model_input_record
            chunk_start_model_input_path = self._maybe_save_chunk_start_model_input(
                state,
                turn_count,
                time_range,
                turn_model_input_record,
            )
            t_prompt_build_end = time.perf_counter()
            inference_start = time.time()
            model_result = await self._call_main_model(
                payload,
                api_messages,
                client=client,
                model_name=model_name,
                session_state=state,
                generation_kwargs=generation_kwargs,
                http_messages=http_messages,
            )
            raw_text, usage, first_action = (
                model_result if len(model_result) == 3 else (*model_result, None)
            )
            inference_time = time.time() - inference_start
            t_inference_end = time.perf_counter()
            generated_text = (
                display_model_output(raw_text)
                if self.config.first_action_logprob_gate
                else normalize_model_output(raw_text)
                if self.config.normalize_output
                else (raw_text or "").strip()
            )

        # ---- 5. 输出写回窗口历史（训练格式的 assistant 轮），记录 prediction ----
        if is_forced_silence:
            first_action = None
        history_text = (
            action_history_output(raw_text, first_action["action"])
            if first_action is not None
            else generated_text
        )
        self._execute_pending_qa_archive(state)

        response_payload = extract_response_payload(history_text)
        if response_payload and state.current_query_text:
            state.current_chunk["response_records"].append((time_range, response_payload))

        state.current_chunk["messages"].append(
            {"role": "assistant", "content": history_text}
        )
        if (
            self.config.input_format != "legacy"
            and self.summarizer is None
            and self.config.keep_qa_history
        ):
            state.current_chunk["dialogue_events"][-1]["answer"] = history_text
        if self._async_summary_enabled():
            state.async_summary_segment["messages"].append(
                {"role": "assistant", "content": history_text}
            )
            self._submit_async_summary_if_needed(state)

        turn_output_record = {}
        if is_forced_silence:
            turn_output_record["inference_skipped"] = True
            turn_output_record["skip_reason"] = "force_silence_before_query"

        t_end = time.perf_counter()
        total_time = t_end - t_start

        prediction = {
            "turn": turn_count,
            "time_range": time_range,
            "query": query_text,
            "input": turn_input_record,
            "output": turn_output_record,
            "prediction": generated_text,
            "raw_prediction": raw_text,
            "action_prediction": history_text,
            "total_time": round(total_time, 3),
            "inference_time": round(inference_time, 3),
        }
        if model_input_record is not None:
            turn_input_record["model_input"] = model_input_record
        if chunk_start_model_input_path:
            prediction["chunk_start_model_input_path"] = chunk_start_model_input_path
        if first_action is not None:
            prediction["first_action_decision"] = first_action
        state.predictions.append(prediction)

        t_end = time.perf_counter()
        adapter_timing = {
            "adapter_total_ms": round((t_end - t_start) * 1000, 1),
        }
        if not is_forced_silence:
            adapter_timing["prompt_build_ms"] = round((t_prompt_build_end - t_prompt_build_start) * 1000, 1)
            adapter_timing["prompt_history_ms"] = round((t_prompt_history_end - t_prompt_build_start) * 1000, 1)
            adapter_timing["prompt_images_ms"] = round((t_prompt_images_end - t_prompt_history_end) * 1000, 1)
            adapter_timing["prompt_record_ms"] = round((t_prompt_build_end - t_prompt_images_end) * 1000, 1)
            adapter_timing["vllm_inference_ms"] = round(inference_time * 1000, 1)
            adapter_timing["post_process_ms"] = round((t_end - t_inference_end) * 1000, 1)
            adapter_timing["pre_inference_ms"] = round((t_prompt_build_start - t_start) * 1000, 1)

        if not is_forced_silence:
            LOGGER.info(
                "[%s] turn=%d timing: total=%.1fms pre=%.1fms prompt_build=%.1fms vllm=%.1fms post=%.1fms",
                state.session_id,
                turn_count,
                adapter_timing["adapter_total_ms"],
                adapter_timing["pre_inference_ms"],
                adapter_timing["prompt_build_ms"],
                adapter_timing["vllm_inference_ms"],
                adapter_timing["post_process_ms"],
            )
        else:
            LOGGER.info(
                "[%s] turn=%d timing: total=%.1fms (forced silence, inference skipped)",
                state.session_id,
                turn_count,
                adapter_timing["adapter_total_ms"],
            )

        # ---- 6. 组装 OpenAI 响应返回 WebUI ----
        result = _chat_completion_response(
            model=self.config.adapter_model,
            content=generated_text,
            usage=usage,
            raw_model=model_name,
            raw_text=raw_text,
        )
        result["streamingharness"]["action_content"] = history_text
        if first_action is not None:
            result["streamingharness"]["first_action_decision"] = first_action
        result["streamingharness"]["timing"] = adapter_timing
        if not is_forced_silence:
            result["streamingharness"]["image_cache"] = {
                "hits": state.image_cache.hits - cache_hits_before,
                "misses": state.image_cache.misses - cache_misses_before,
                "entries": len(state.image_cache),
                "bytes": state.image_cache.bytes,
                "max_bytes": state.image_cache.max_bytes,
                "max_entries": state.image_cache.max_entries,
            }
        summarizer_timing = {}
        if state.mid_term_history:
            last_mid = state.mid_term_history[-1]
            summarizer_timing["last_mid_term_ms"] = round(last_mid.get("inference_time", 0) * 1000, 1)
            summarizer_timing["last_mid_term_chunk"] = last_mid.get("chunk_index")
            if last_mid.get("barrier_wait_time") is not None:
                summarizer_timing["barrier_wait_ms"] = round(last_mid["barrier_wait_time"] * 1000, 1)
        if state.long_term_history:
            last_long = state.long_term_history[-1]
            summarizer_timing["last_long_term_ms"] = round(last_long.get("inference_time", 0) * 1000, 1)
        if self.summarizer is not None:
            result["streamingharness"]["summarizer_timing"] = summarizer_timing
            result["streamingharness"]["memory"] = {
                "mid_term_summaries": [
                    {"chunk_index": e["chunk_index"], "frame_range": e["frame_range"], "summary_text": e["summary_text"]}
                    for e in state.mid_term_summaries
                ],
                "long_term_memory": state.memory_state.get("long_term_memory", ""),
            }
        return result

    async def _forward_text_only(
        self,
        payload: dict[str, Any],
        *,
        client: Optional[AsyncOpenAI] = None,
        model_name: Optional[str] = None,
    ) -> dict[str, Any]:
        client = client or self.main_client
        model_name = model_name or self.config.main_model
        prompt_text = _extract_user_prompt_text(payload.get("messages") or [])
        if is_identity_question(prompt_text):
            return _chat_completion_response(
                model=self.config.adapter_model, content=identity_action(prompt_text),
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                raw_model=model_name, raw_text=identity_action(prompt_text),
            )
        generation_kwargs = self._main_generation_kwargs(payload)
        response = await client.chat.completions.create(
            model=model_name,
            messages=self._build_main_http_messages(payload.get("messages") or []),
            **generation_kwargs,
        )
        raw_text = response.choices[0].message.content if response.choices else ""
        usage = response.usage.model_dump() if getattr(response, "usage", None) else None
        return _chat_completion_response(
            model=self.config.adapter_model,
            content=raw_text or "",
            usage=usage,
            raw_model=model_name,
            raw_text=raw_text or "",
        )

    def _time_range_for_frame(self, frame_index: int) -> str:
        start = frame_index * self.config.frame_seconds
        return f"{start:.1f} seconds"
