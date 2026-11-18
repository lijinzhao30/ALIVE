# -*- coding: utf-8 -*-
"""主模型输入构造与 vLLM 调用。

每来一帧，发给主模型的 messages 由三部分组成（默认 no_time 训练格式）：
  [system prompt]
  + sparse_text_messages ：已滑出窗口的稀疏问答（只保留文本：有提问的 user 轮 / 有 </response> 的 assistant 轮）
  + current_chunk.messages：窗口内逐帧的 user(问题+图片) / assistant(</silence> 或 </response> ...) 交替消息
窗口满 chunk(=30) 帧后，一次性裁剪到最近 retain_frames(=10) 帧（_trim_current_chunk_to_retained_frames），
之后 20 帧内消息前缀稳定，可以命中 vLLM 的 prefix cache。
"""

from __future__ import annotations

import math
from typing import Any, Optional

from openai import AsyncOpenAI

from adapter_common import (
    LOGGER,
    RESPONSE_TOKEN,
    RESPONSE_TOKEN_ID,
    SILENCE_TOKEN,
    SILENCE_TOKEN_ID,
    TIME_RANGE_RE,
    _extract_time_range_from_message,
    _format_batch_time_marker,
    _format_seconds_words,
    _get_i18n,
    _parse_start_second,
    archive_chunk_response_records,
)
from adapter_request import (
    _extract_extra_body,
    _extract_user_prompt_text,
    _file_to_data_url_cached,
    _internal_message_has_image,
    _internal_message_to_openai,
)
from adapter_session import SessionState
from adapter_identity import identity_action, is_identity_question


def build_static_system_content(
    extra_system_messages: Optional[list[str]] = None,
    memory_state: Optional[dict[str, Any]] = None,
    mid_term_summaries: Optional[list[dict[str, Any]]] = None,
    language: str = "en",
) -> str:
    # 仅 summarizer / legacy 模式会有内容：长期记忆 + 中期摘要拼成 [Video History]。
    i18n = _get_i18n(language)
    sections: list[str] = []
    for message in extra_system_messages or []:
        if message and message.strip():
            sections.append(message.strip())

    history_parts: list[str] = []
    if memory_state is not None and memory_state.get("long_term_memory"):
        history_parts.append(memory_state["long_term_memory"])
    if mid_term_summaries:
        for entry in mid_term_summaries:
            history_parts.append(f"<{entry['frame_range']}>\n{entry['summary_text']}")

    if history_parts:
        sections.append(
            i18n["video_history_header"]
            + "\n\n".join(history_parts)
        )

    return "\n\n".join(sections) if sections else ""


def build_dynamic_system_content(
    current_query_text: Optional[str] = None,
    memory_state: Optional[dict[str, Any]] = None,
    include_qa_history: bool = True,
    current_chunk_index: int = 0,
    language: str = "en",
) -> str:
    # 仅 summarizer / legacy 模式会有内容：旧版 [Q&A History] 与 [User Query] 文本块。
    i18n = _get_i18n(language)
    sections: list[str] = []

    if include_qa_history and memory_state is not None and memory_state.get("qa_history"):
        qa_entries = [
            entry for entry in memory_state["qa_history"]
            if entry.get("archived_in_chunk", 0) < current_chunk_index
        ]
        if qa_entries:
            qa_lines: list[str] = []
            for idx, entry in enumerate(qa_entries, 1):
                q_time = entry["query_time"] or "N/A"
                parts = [f"#{idx} [{i18n['qa_query_label']}@{q_time}] {entry['query']}"]
                for response_time, payload in entry.get("responses", []):
                    parts.append(f"[{i18n['qa_response_label']}@{response_time}] {payload}")
                qa_lines.append("\n".join(parts))
            sections.append(
                i18n["qa_history_header"]
                + "\n".join(qa_lines)
            )

    if current_query_text:
        sections.append(i18n["user_query_header"] + "\n" + current_query_text.strip())

    return "\n\n".join(sections) if sections else ""


class ModelMixin:
    """StreamingInferAdapter 的 prompt 构造与主模型调用部分（依赖 self.config / self.summarizer）。"""

    # ==================== 用户问题状态 ====================

    def _update_query_state(
        self,
        state: SessionState,
        prompt_text: str,
        time_range: str,
    ) -> Optional[str]:
        if not self.config.use_prompt_as_query:
            return None

        normalized_prompt = (prompt_text or "").strip()
        if not normalized_prompt:
            return None

        # 同样的文字再次提交也是新问题，旧问答先归档，再更新提问时刻。
        if state.current_query_text is not None:
            state._pending_qa_archive = (state.current_query_text, state.query_start_time)
        state.current_query_text = normalized_prompt
        state.query_start_time = time_range
        state.query_in_current_chunk = True
        return normalized_prompt

    def _execute_pending_qa_archive(self, state: SessionState) -> None:
        if state._pending_qa_archive is None:
            return
        old_query, old_start_time = state._pending_qa_archive
        archive_chunk_response_records(
            state.current_chunk,
            state.memory_state,
            old_query,
            old_start_time,
            chunk_index=state.chunk_index,
        )
        state.current_chunk["response_records"] = []
        state._pending_qa_archive = None

    # ==================== 窗口消息构造 ====================

    def _build_internal_user_message(
        self,
        time_range=None,
        image_path=None,
        query_text=None,
        *,
        time_ranges=None,
        image_paths=None,
    ) -> dict[str, Any]:
        i18n = _get_i18n(self.config.language)
        if time_ranges is None:
            time_ranges = [time_range] if time_range else []
        if image_paths is None:
            image_paths = [image_path] if image_path else []
        content: list[dict[str, Any]] = []
        if self.config.input_format == "legacy":
            if query_text:
                content.append({"type": "text", "text": i18n["user_query_header"] + "\n" + query_text})
            batch_time_marker = _format_batch_time_marker(time_ranges)
            if batch_time_marker:
                content.append({"type": "text", "text": f"<{batch_time_marker}>"})
        else:
            # 两版训练数据只差时间前缀；问题仅放在实际提问的这一帧。
            text = f"<{time_ranges[0]}>\n" if self.config.input_format == "timed" else ""
            if query_text:
                text += query_text + "\n"
            if text:
                content.append({"type": "text", "text": text})
        for ip in image_paths:
            content.append(
                {
                    "type": "image",
                    "image": ip,
                    "max_pixels": self.config.max_pixels,
                    "resize_short_edge": self.config.resize_short_edge,
                }
            )
        return {"role": "user", "content": content}

    def _archive_dialogue_events(
        self, state: SessionState, events: Optional[list[dict[str, str]]] = None
    ) -> None:
        for event in events if events is not None else state.current_chunk["dialogue_events"]:
            query = event["query"]
            # 事件记录的是该帧实际提交的问题，滑出窗口时也不能按文字去重。
            is_new_query = bool(query)
            answer = event["answer"]
            if not is_new_query and not answer.startswith("</response>"):
                continue
            # 训练集的旧窗口仅保留问答时刻；旧帧和纯静默轮次均不进入下一窗口。
            if self.config.input_format == "no_time":
                user_content = query if is_new_query else ""
            else:
                timestamp = f"<{_format_seconds_words(_parse_start_second(event['time_range']))}>"
                user_content = timestamp + (f"\n{query}" if is_new_query else "")
            state.sparse_text_messages.extend(
                [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": answer},
                ]
            )

    def _trim_current_chunk_to_retained_frames(self, state: SessionState) -> None:
        """Archive old no-summary turns and keep only the most recent retain_frames."""

        chunk = max(1, int(self.config.chunk or 1))
        retain = max(0, min(int(self.config.retain_frames or 0), chunk - 1))
        current_turns = int(state.current_chunk.get("turn_count") or 0)
        remove_turns = max(0, current_turns - retain)
        if remove_turns <= 0:
            return

        removed_events = state.current_chunk["dialogue_events"][:remove_turns]
        if removed_events:
            # 旧视觉帧不再进入窗口，只把需要保留的稀疏问答文本归档。
            self._archive_dialogue_events(state, removed_events)
            del state.current_chunk["dialogue_events"][:remove_turns]

        del state.current_chunk["messages"][: remove_turns * 2]
        for key in ("image_paths", "frame_time_ranges", "summarizer_frame_cache"):
            del state.current_chunk[key][:remove_turns]
        state.current_chunk["frame_count"] = max(
            0, int(state.current_chunk.get("frame_count") or 0) - remove_turns
        )
        state.current_chunk["turn_count"] = max(0, current_turns - remove_turns)
        state.current_chunk["api_msg_cache"] = []
        state.image_cache.retain_paths(state.current_chunk["image_paths"])
        LOGGER.info(
            "[%s] chunk window reached %d turn(s); archived %d old turn(s), retained %d recent turn(s)",
            state.session_id,
            chunk,
            remove_turns,
            state.current_chunk["turn_count"],
        )

    @staticmethod
    def _normalize_model_timestamps(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        first_time = next(
            (
                _parse_start_second(time_range)
                for message in messages
                if message.get("role") == "user"
                for time_range in [_extract_time_range_from_message(message)]
                if time_range
            ),
            0.0,
        )
        if first_time <= 0:
            return messages

        normalized = []
        for message in messages:
            if message.get("role") != "user":
                normalized.append(message)
                continue
            content = message["content"]
            if isinstance(content, str):
                content = TIME_RANGE_RE.sub(
                    lambda match: f"<{_format_seconds_words(_parse_start_second(match.group('range')) - first_time)}>",
                    content,
                    count=1,
                )
            else:
                # 当前帧的时间戳和问题可能在同一个 text 中，只改开头的时间戳。
                content = [
                    {
                        **item,
                        "text": TIME_RANGE_RE.sub(
                            lambda match: f"<{_format_seconds_words(_parse_start_second(match.group('range')) - first_time)}>",
                            item["text"],
                            count=1,
                        ),
                    }
                    if item.get("type") == "text" and TIME_RANGE_RE.match(item.get("text", ""))
                    else item
                    for item in content
                ]
            normalized.append({**message, "content": content})
        return normalized

    def _build_main_internal_messages(
        self,
        state: SessionState,
    ) -> tuple[list[dict[str, Any]], str]:
        memory_state = state.memory_state if state.current_query_text else None
        static_content = build_static_system_content(
            memory_state=memory_state,
            mid_term_summaries=state.mid_term_summaries,
            language=self.config.language,
        )
        inject_query = (
            state.current_query_text
            if self.config.input_format == "legacy" and not state.query_in_current_chunk
            else None
        )
        dynamic_content = build_dynamic_system_content(
            current_query_text=inject_query,
            memory_state=memory_state,
            include_qa_history=self.config.keep_qa_history and (
                self.config.input_format == "legacy" or self.summarizer is not None
            ),
            current_chunk_index=state.chunk_index,
            language=self.config.language,
        )
        prefix_content = "\n\n".join(
            part for part in (static_content, dynamic_content) if part
        )
        # legacy 保留旧版 Q&A History；稀疏问答和时间归一化仅用于新训练格式。
        training_history = self.config.input_format != "legacy" and self.summarizer is None
        history = state.sparse_text_messages if training_history else []
        all_messages = list(history) + list(state.current_chunk["messages"])
        if training_history:
            all_messages = self._normalize_model_timestamps(all_messages)

        if prefix_content:
            for idx in range(len(history), len(all_messages)):
                message = all_messages[idx]
                if message.get("role") != "user":
                    continue
                new_message = dict(message)
                content = message.get("content")
                if isinstance(content, list):
                    new_message["content"] = [
                        {"type": "text", "text": prefix_content}
                    ] + list(content)
                elif isinstance(content, str):
                    new_message["content"] = prefix_content + "\n\n" + content
                else:
                    new_message["content"] = prefix_content
                all_messages[idx] = new_message
                break

        return all_messages, prefix_content

    def _build_main_api_messages(self, state: SessionState) -> list[dict[str, Any]]:
        all_messages, _ = self._build_main_internal_messages(state)
        return [_internal_message_to_openai(message) for message in all_messages]

    def _build_cached_api_messages(
        self,
        state: SessionState,
        internal_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        # Rebuild text/prefixes, but reuse encoded frames in a bounded session cache.
        # Keep payloads out of message history and diagnostic prediction records.
        state.image_cache.retain_paths(state.current_chunk["image_paths"])
        if any(_internal_message_has_image(message) for message in internal_messages):
            state.current_chunk["api_msg_cache"] = []
            return [
                _internal_message_to_openai(message, image_cache=state.image_cache)
                for message in internal_messages
            ]

        cache = state.current_chunk["api_msg_cache"]
        chunk_msgs = state.current_chunk["messages"]
        # Incrementally convert new chunk messages and append to cache.
        # cache[i] corresponds to chunk_msgs[i] (without prefix injection).
        while len(cache) < len(chunk_msgs):
            cache.append(_internal_message_to_openai(chunk_msgs[len(cache)]))
        # internal_messages[0] has prefix injected, so always re-convert it.
        # internal_messages[1:] are identical to chunk_msgs[1:], so reuse cache.
        first_msg = _internal_message_to_openai(internal_messages[0])
        remaining = cache[1:len(internal_messages)]
        return [first_msg] + remaining

    def _build_main_http_messages(
        self,
        api_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        messages = list(api_messages)
        system_prompt = self.config.system_prompt
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        return messages

    # ==================== 主模型调用 ====================

    async def _call_main_model(
        self,
        inbound_payload: dict[str, Any],
        api_messages: list[dict[str, Any]],
        *,
        client: Optional[AsyncOpenAI] = None,
        model_name: Optional[str] = None,
        session_state: Optional["SessionState"] = None,
        generation_kwargs: Optional[dict[str, Any]] = None,
        http_messages: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[str, Optional[dict[str, Any]], Optional[dict[str, Any]]]:
        # WebUI sends each question once; later camera frames must stay silent
        # instead of regenerating an answer from an old identity/training query.
        current_prompt = _extract_user_prompt_text(inbound_payload.get("messages") or [])
        if is_identity_question(current_prompt):
            return identity_action(current_prompt), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, None
        if not current_prompt and session_state is not None and is_identity_question(session_state.current_query_text or ""):
            return SILENCE_TOKEN, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, None
        client = client or self.main_client
        model_name = model_name or self.config.main_model
        generation_kwargs = generation_kwargs or self._main_generation_kwargs(inbound_payload)
        api_messages = http_messages or self._build_main_http_messages(api_messages)
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=api_messages,
                **generation_kwargs,
            )
        finally:
            _file_to_data_url_cached.cache_clear()
        raw_text = response.choices[0].message.content if response.choices else ""
        usage = response.usage.model_dump() if getattr(response, "usage", None) else None
        first_action = None
        if self.config.first_action_logprob_gate:
            first_action = await self._first_action_decision(
                client=client,
                model_name=model_name,
                messages=api_messages,
            )
        return raw_text or "", usage, first_action

    async def _first_action_decision(
        self,
        *,
        client: AsyncOpenAI,
        model_name: str,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """只允许两个动作 token，贪心返回者即为首位置概率较高的动作。"""
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            logprobs=True,
            # allowed_token_ids 只约束采样；vLLM 的候选概率仍按原始全词表
            # 排序。多取少量候选，才能稳定覆盖两个已训练的动作 token。
            top_logprobs=20,
            extra_body={
                "skip_special_tokens": False,
                "greedy": True,
                "ignore_eos": True,
                "return_tokens_as_token_ids": True,
                "allowed_token_ids": [SILENCE_TOKEN_ID, RESPONSE_TOKEN_ID],
            },
        )
        token = (response.choices[0].message.content or "").strip()
        if token == SILENCE_TOKEN:
            action = "silence"
        elif token == RESPONSE_TOKEN:
            action = "response"
        else:
            raise RuntimeError(f"首动作二选一返回了非法 token: {token!r}")
        token_labels = {
            SILENCE_TOKEN: "silence",
            RESPONSE_TOKEN: "response",
            f"token_id:{SILENCE_TOKEN_ID}": "silence",
            f"token_id:{RESPONSE_TOKEN_ID}": "response",
            str(SILENCE_TOKEN_ID): "silence",
            str(RESPONSE_TOKEN_ID): "response",
        }
        token_logprob = response.choices[0].logprobs.content[0]
        # vLLM 可能只把“未选中的候选”放入 top_logprobs；实际选中的
        # token 及其概率保存在当前 content 项本身。两处合并后才能稳定
        # 得到 silence/response 的完整二选一概率。
        candidate_logprobs = [token_logprob, *token_logprob.top_logprobs]
        action_logprobs = {
            token_labels[item.token.strip()]: float(item.logprob)
            for item in candidate_logprobs
            if item.token.strip() in token_labels
        }
        # allowed_token_ids 只约束最终采样，原始全词表 Top-K 未必同时包含
        # 两个动作。对缺失动作单独强制采样一次；vLLM 返回的 selected-token
        # logprob 是施加 allowed-token mask 前的原始 logprob，可直接比较。
        action_tokens = {
            "silence": (SILENCE_TOKEN, SILENCE_TOKEN_ID),
            "response": (RESPONSE_TOKEN, RESPONSE_TOKEN_ID),
        }
        for name in ("silence", "response"):
            if name in action_logprobs:
                continue
            expected_token, token_id = action_tokens[name]
            forced = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=1,
                temperature=0.0,
                top_p=1.0,
                logprobs=True,
                top_logprobs=0,
                extra_body={
                    "skip_special_tokens": False,
                    "greedy": True,
                    "ignore_eos": True,
                    "return_tokens_as_token_ids": True,
                    "allowed_token_ids": [token_id],
                },
            )
            forced_token = (forced.choices[0].message.content or "").strip()
            if forced_token != expected_token:
                raise RuntimeError(
                    f"强制读取{name}概率时返回非法 token: {forced_token!r}"
                )
            action_logprobs[name] = float(
                forced.choices[0].logprobs.content[0].logprob
            )
        if set(action_logprobs) != {"silence", "response"}:
            raise RuntimeError(f"首动作概率缺少候选 token: {action_logprobs!r}")
        weights = {name: math.exp(value) for name, value in action_logprobs.items()}
        normalizer = sum(weights.values())
        probabilities = {
            name: round(value / normalizer, 8) for name, value in weights.items()
        }
        action = max(("silence", "response"), key=probabilities.__getitem__)
        return {
            "action": action,
            "selected_token": SILENCE_TOKEN if action == "silence" else RESPONSE_TOKEN,
            "method": "restricted_greedy_argmax",
            "probabilities": probabilities,
            "logprobs": action_logprobs,
        }

    def _main_generation_kwargs(self, inbound_payload: dict[str, Any]) -> dict[str, Any]:
        # 默认忽略 WebUI 请求里的采样参数，统一使用 adapter 启动参数（与离线 infer 对齐）。
        extra_body = _extract_extra_body(inbound_payload)
        extra_body.setdefault("skip_special_tokens", False)
        extra_body.setdefault("greedy", False)
        if self.config.honor_inbound_generation_params:
            extra_body.setdefault("top_k", self.config.main_top_k)
            extra_body.setdefault("repetition_penalty", self.config.main_repetition_penalty)
            return {
                "max_tokens": inbound_payload.get("max_tokens", self.config.main_max_tokens),
                "temperature": inbound_payload.get("temperature", self.config.main_temperature),
                "top_p": inbound_payload.get("top_p", self.config.main_top_p),
                "presence_penalty": inbound_payload.get("presence_penalty", self.config.main_presence_penalty),
                "extra_body": extra_body,
            }

        extra_body["top_k"] = self.config.main_top_k
        extra_body["repetition_penalty"] = self.config.main_repetition_penalty
        return {
            "max_tokens": self.config.main_max_tokens,
            "temperature": self.config.main_temperature,
            "top_p": self.config.main_top_p,
            "presence_penalty": self.config.main_presence_penalty,
            "extra_body": extra_body,
        }
