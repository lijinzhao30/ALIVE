# -*- coding: utf-8 -*-
"""Adapter 配置与命令行解析。

所有参数既可以用命令行传，也可以用同名环境变量覆盖（命令行优先）。
关键约束（parse_args 末尾校验）：
  - 训练输入格式（no_time / timed）必须关闭 summarizer、保留 QA history、chunk=30；
  - retain_frames 必须小于 chunk：窗口攒满 30 帧后裁剪到最近 retain_frames 帧，再继续累积。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from adapter_common import (
    DEFAULT_SAVE_ROOT,
    DEFAULT_SYSTEM_PROMPT_EN,
    derive_light_out_dir,
    derive_model_output_name,
    resolve_save_dir,
    sanitize_output_name,
)


@dataclass
class AdapterConfig:
    host: str = "127.0.0.1"
    port: int = 8070
    adapter_model: str = "streaming-infer-adapter"
    main_api_base: str = "http://127.0.0.1:7060/v1"
    main_model: str = "ALIVE"
    main_backends: tuple[dict[str, str], ...] = ()
    api_key: str = "EMPTY"
    allowed_local_image_roots: tuple[str, ...] = ()
    frame_seconds: float = 1.0
    input_format: str = "no_time"
    max_pixels: int = 786432
    resize_short_edge: int = 480
    main_max_tokens: int = 256
    main_temperature: float = 0.8
    main_top_p: float = 0.9
    main_top_k: int = 40
    main_repetition_penalty: float = 1.1
    main_presence_penalty: float = 1.5
    honor_inbound_generation_params: bool = False
    chunk: int = 30
    retain_frames: int = 10
    compress_every_n_chunks: int = 5
    async_summary_lead_frames: int = 10
    use_prompt_as_query: bool = True
    force_silence_before_query: bool = True
    keep_qa_history: bool = True
    normalize_output: bool = True
    first_action_logprob_gate: bool = False
    enable_summarizer: bool = False
    summarizer_model: str = "models/summary"
    summarizer_api_base: str = "http://127.0.0.1:8065/v1"
    longterm_model: str = "models/summary"
    longterm_api_base: str = "http://127.0.0.1:8065/v1"
    summarizer_max_pixels: int = 786432
    summarizer_key_frames: int = 0
    summarizer_phase_seconds: float = 10.0
    mid_term_max_tokens: int = 4000
    mid_term_target_tokens: int = 3000
    long_term_max_tokens: int = 2000
    long_term_target_tokens: int = 1000
    mid_term_temperature: float = 0.8
    mid_term_top_p: float = 0.9
    mid_term_top_k: int = 40
    mid_term_repetition_penalty: float = 1.1
    mid_term_presence_penalty: float = 0.0
    long_term_temperature: float = 1.0
    long_term_top_p: float = 1.0
    long_term_top_k: int = 80
    long_term_repetition_penalty: float = 1.1
    long_term_presence_penalty: float = 0.0
    long_term_memory_window: int = 40
    request_timeout_seconds: float = 300.0
    session_timeout_seconds: float = 3600.0
    max_active_sessions: int = 0
    out_dir: Optional[str] = None
    light_out_dir: Optional[str] = None
    debug_input_dir: Optional[str] = None
    save_root: Optional[str] = None
    output_model_name: str = ""
    per_session_dirs: bool = True
    save_model_inputs: bool = False
    save_debug_inputs: bool = False
    summarizer_debug: bool = False
    frame_save_dir: str = ".runtime/frames"
    language: str = "en"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT_EN


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _split_paths(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    normalized = value.replace(",", os.pathsep)
    return tuple(item.strip() for item in normalized.split(os.pathsep) if item.strip())


def parse_args() -> AdapterConfig:
    parser = argparse.ArgumentParser(description="StreamingHarness live OpenAI adapter")
    parser.add_argument("--host", default=os.environ.get("ADAPTER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=_env_int("ADAPTER_PORT", 8070))
    parser.add_argument(
        "--adapter-model",
        default=os.environ.get("ADAPTER_MODEL", "streaming-infer-adapter"),
    )
    parser.add_argument(
        "--main-api-base",
        default=os.environ.get("MAIN_API_BASE", "http://127.0.0.1:7060/v1"),
    )
    parser.add_argument(
        "--main-model",
        default=os.environ.get("MAIN_MODEL", "ALIVE"),
    )
    parser.add_argument(
        "--main-backends",
        default=os.environ.get("MAIN_BACKENDS", ""),
        help='JSON array of backends: [{"name":"...","api_base":"...","model":"..."},...]',
    )
    parser.add_argument("--api-key", default=os.environ.get("MODEL_API_KEY", "EMPTY"))
    parser.add_argument("--frame-seconds", type=float, default=_env_float("FRAME_SECONDS", 1.0))
    parser.add_argument(
        "--input-format",
        default=os.environ.get("INPUT_FORMAT", "no_time"),
        choices=("legacy", "timed", "no_time"),
    )
    parser.add_argument("--max-pixels", type=int, default=_env_int("MAX_PIXELS", 786432))
    parser.add_argument(
        "--resize-short-edge",
        type=int,
        default=_env_int("RESIZE_SHORT_EDGE", 480),
        help="Resize each frame proportionally so its shorter edge has this many pixels before applying max_pixels; 0 disables it.",
    )
    parser.add_argument("--main-max-tokens", type=int, default=_env_int("MAIN_MAX_TOKENS", 256))
    parser.add_argument(
        "--main-temperature",
        type=float,
        default=_env_float("MAIN_TEMPERATURE", 0.8),
    )
    parser.add_argument("--main-top-p", type=float, default=_env_float("MAIN_TOP_P", 0.9))
    parser.add_argument("--main-top-k", type=int, default=_env_int("MAIN_TOP_K", 40))
    parser.add_argument(
        "--main-repetition-penalty",
        type=float,
        default=_env_float("MAIN_REPETITION_PENALTY", 1.1),
    )
    parser.add_argument(
        "--main-presence-penalty",
        type=float,
        default=_env_float("MAIN_PRESENCE_PENALTY", 1.5),
    )
    parser.add_argument(
        "--honor-inbound-generation-params",
        action="store_true",
        default=_env_bool("HONOR_INBOUND_GENERATION_PARAMS", False),
        help="Use max_tokens/temperature/top_p from incoming WebUI requests instead of infer.sh-style defaults.",
    )
    parser.add_argument("--chunk", type=int, default=_env_int("CHUNK", 30))
    parser.add_argument(
        "--retain-frames",
        type=int,
        default=_env_int("RETAIN_FRAMES", 10),
        help="For training input formats without summarizer, when the frame window reaches --chunk, archive old turns and keep only this many recent frames before accumulating again.",
    )
    parser.add_argument(
        "--compress-every-n-chunks",
        type=int,
        default=_env_int("COMPRESS_EVERY_N_CHUNKS", 5),
    )
    parser.add_argument(
        "--async-summary-lead-frames",
        type=int,
        default=_env_int("ASYNC_SUMMARY_LEAD_FRAMES", 10),
        help="Generate async summaries this many turns before the chunk boundary. Name kept for compatibility.",
    )
    parser.add_argument(
        "--no-prompt-as-query",
        action="store_true",
        default=not _env_bool("USE_PROMPT_AS_QUERY", True),
    )
    parser.add_argument(
        "--force-silence-before-query",
        action="store_true",
        default=_env_bool("FORCE_SILENCE_BEFORE_QUERY", True),
    )
    parser.add_argument(
        "--no-force-silence-before-query",
        action="store_false",
        dest="force_silence_before_query",
        help="Disable infer.sh-style forced </silence> before the first query.",
    )
    parser.add_argument(
        "--no-qa-history",
        action="store_true",
        default=not _env_bool("KEEP_QA_HISTORY", True),
    )
    parser.add_argument(
        "--no-normalize-output",
        action="store_true",
        default=not _env_bool("NORMALIZE_OUTPUT", True),
    )
    parser.add_argument(
        "--first-action-logprob-gate",
        action="store_true",
        default=_env_bool("FIRST_ACTION_LOGPROB_GATE", False),
        help="用 </silence> 与 </response> 的首 token 概率决定 history 与评分动作。",
    )
    parser.add_argument(
        "--disable-summarizer",
        action="store_true",
        default=not _env_bool("ENABLE_SUMMARIZER", False),
    )
    parser.add_argument(
        "--summarizer-model",
        default=os.environ.get(
            "SUMMARIZER_MODEL",
            "models/summary",
        ),
    )
    parser.add_argument(
        "--summarizer-api-base",
        default=os.environ.get("SUMMARIZER_API_BASE", "http://127.0.0.1:8065/v1"),
    )
    parser.add_argument(
        "--longterm-model",
        default=os.environ.get(
            "LONGTERM_SUMMARIZER_MODEL",
            os.environ.get("SUMMARIZER_MODEL", "models/summary"),
        ),
    )
    parser.add_argument(
        "--longterm-api-base",
        default=os.environ.get(
            "LONGTERM_SUMMARIZER_API_BASE",
            os.environ.get("SUMMARIZER_API_BASE", "http://127.0.0.1:8065/v1"),
        ),
    )
    parser.add_argument(
        "--summarizer-max-pixels",
        type=int,
        default=_env_int("SUMMARIZER_MAX_PIXELS", 786432),
    )
    parser.add_argument(
        "--summarizer-key-frames",
        type=int,
        default=_env_int("SUMMARIZER_KEY_FRAMES", 0),
    )
    parser.add_argument(
        "--summarizer-phase-seconds",
        type=float,
        default=_env_float("SUMMARIZER_PHASE_SECONDS", 10.0),
    )
    parser.add_argument(
        "--mid-term-max-tokens",
        type=int,
        default=_env_int("MID_TERM_MAX_TOKENS", 4000),
    )
    parser.add_argument(
        "--mid-term-target-tokens",
        type=int,
        default=_env_int("MID_TERM_TARGET_TOKEN_COUNT", 3000),
    )
    parser.add_argument(
        "--long-term-max-tokens",
        type=int,
        default=_env_int("LONG_TERM_MAX_TOKENS", 2000),
    )
    parser.add_argument(
        "--long-term-target-tokens",
        type=int,
        default=_env_int("LONG_TERM_TARGET_TOKEN_COUNT", 1000),
    )
    parser.add_argument(
        "--mid-term-temperature",
        type=float,
        default=_env_float("MID_TERM_TEMPERATURE", 0.8),
    )
    parser.add_argument(
        "--mid-term-top-p",
        type=float,
        default=_env_float("MID_TERM_TOP_P", 0.9),
    )
    parser.add_argument(
        "--mid-term-top-k",
        type=int,
        default=_env_int("MID_TERM_TOP_K", 40),
    )
    parser.add_argument(
        "--mid-term-repetition-penalty",
        type=float,
        default=_env_float("MID_TERM_REPETITION_PENALTY", 1.0),
    )
    parser.add_argument(
        "--mid-term-presence-penalty",
        type=float,
        default=_env_float("MID_TERM_PRESENCE_PENALTY", 0.0),
    )
    parser.add_argument(
        "--long-term-temperature",
        type=float,
        default=_env_float("LONG_TERM_TEMPERATURE", 1.0),
    )
    parser.add_argument(
        "--long-term-top-p",
        type=float,
        default=_env_float("LONG_TERM_TOP_P", 1.0),
    )
    parser.add_argument(
        "--long-term-top-k",
        type=int,
        default=_env_int("LONG_TERM_TOP_K", 80),
    )
    parser.add_argument(
        "--long-term-repetition-penalty",
        type=float,
        default=_env_float("LONG_TERM_REPETITION_PENALTY", 1.0),
    )
    parser.add_argument(
        "--long-term-presence-penalty",
        type=float,
        default=_env_float("LONG_TERM_PRESENCE_PENALTY", 0.0),
    )
    parser.add_argument(
        "--long-term-memory-window",
        type=int,
        default=_env_int("LONG_TERM_MEMORY_WINDOW", 40),
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=_env_float("REQUEST_TIMEOUT_SECONDS", 300.0),
    )
    parser.add_argument(
        "--session-timeout-seconds",
        type=float,
        default=_env_float("SESSION_TIMEOUT_SECONDS", 3600.0),
        help="Drop idle sessions after this many seconds to release CPU memory.",
    )
    parser.add_argument(
        "--max-active-sessions",
        type=int,
        default=_env_int("MAX_ACTIVE_SESSIONS", 0),
        help="Maximum live sessions kept in adapter memory; 0 disables the limit.",
    )
    parser.add_argument(
        "--allowed-local-image-roots",
        default=os.environ.get("ALLOWED_LOCAL_IMAGE_ROOTS", ""),
        help="Comma- or colon-separated directories whose image files may be referenced directly by requests.",
    )
    parser.add_argument(
        "--frame-save-dir",
        default=os.environ.get("FRAME_SAVE_DIR", ".runtime/frames"),
        help="Directory to save base64 frames received from WebUI.",
    )
    parser.add_argument(
        "--save-root",
        default=os.environ.get(
            "LIVE_ADAPTER_SAVE_ROOT",
            os.environ.get("SAVE_ROOT", DEFAULT_SAVE_ROOT),
        ),
        help="Root used for auto-generated output_*, output_light_*, and input_* dirs.",
    )
    parser.add_argument(
        "--run-timestamp",
        default=os.environ.get("LIVE_ADAPTER_RUN_TIMESTAMP", ""),
        help="Timestamp suffix for auto-generated live save dirs.",
    )
    parser.add_argument(
        "--output-model-name",
        default=os.environ.get("LIVE_ADAPTER_OUTPUT_MODEL_NAME", ""),
        help="Model-name suffix for auto-generated live save dirs.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("LIVE_ADAPTER_OUT_DIR") or os.environ.get("OUT_DIR"),
        help="Full live output directory. Relative paths resolve under save-root.",
    )
    parser.add_argument(
        "--light-out-dir",
        default=os.environ.get("LIVE_ADAPTER_LIGHT_OUT_DIR")
        or os.environ.get("LIGHT_OUT_DIR"),
        help="Light live output directory. Defaults to output_light_* derived from out-dir.",
    )
    parser.add_argument(
        "--debug-input-dir",
        default=os.environ.get("LIVE_ADAPTER_DEBUG_INPUT_DIR")
        or os.environ.get("DEBUG_INPUT_DIR"),
        help="Directory for live input_* debug snapshots. Relative paths resolve under save-root.",
    )
    parser.add_argument(
        "--no-live-save",
        action="store_true",
        default=not _env_bool("LIVE_SAVE_OUTPUTS", False),
        help="Disable live output_*/output_light_* writing.",
    )
    parser.add_argument(
        "--no-debug-inputs",
        action="store_true",
        default=not _env_bool("LIVE_SAVE_DEBUG_INPUTS", False),
        help="Disable live input_* debug snapshot writing.",
    )
    parser.add_argument(
        "--no-save-model-inputs",
        action="store_true",
        default=not _env_bool("SAVE_MODEL_INPUTS", False),
        help="Do not embed per-turn model_input records in output_*.",
    )
    parser.add_argument(
        "--no-summarizer-debug",
        action="store_true",
        default=not _env_bool("SUMMARIZER_DEBUG", True),
        help="Do not keep/save mid/long-term summary debug inputs.",
    )
    parser.add_argument(
        "--system-prompt",
        default=os.environ.get("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT_EN),
        help="System prompt prepended to every 8B main-model request. Set SYSTEM_PROMPT='' to disable the built-in default.",
    )
    parser.add_argument(
        "--language",
        default=os.environ.get("ADAPTER_LANGUAGE", "en"),
        choices=["zh", "en"],
        help="Language for context injection text (Video History header, Q&A History header, User Query header). 'zh' for Chinese, 'en' for English.",
    )
    args = parser.parse_args()
    if args.input_format != "legacy" and (
        not args.disable_summarizer or args.no_qa_history or args.chunk != 30
    ):
        parser.error("training input formats require --disable-summarizer, QA history, and --chunk 30")
    if args.input_format != "legacy" and args.retain_frames >= args.chunk:
        parser.error("training input formats require --retain-frames to be smaller than --chunk")

    # 输出目录：默认按会话在 save_root 下建 output_<时间>_<模型名>/ 与 output_light_*/；显式给 out_dir 时改为全局单目录。
    raw_save_root = (args.save_root or DEFAULT_SAVE_ROOT).strip() or DEFAULT_SAVE_ROOT
    save_root = os.path.normpath(os.path.expanduser(raw_save_root))
    run_timestamp = args.run_timestamp.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_model_name = (
        sanitize_output_name(args.output_model_name)
        if args.output_model_name.strip()
        else derive_model_output_name(args.main_model)
    )
    no_live_save = args.no_live_save
    no_debug_inputs = args.no_debug_inputs
    per_session_dirs = not no_live_save
    explicit_out_dir = resolve_save_dir(args.out_dir, save_root)
    explicit_light_out_dir = resolve_save_dir(args.light_out_dir, save_root)
    explicit_debug_input_dir = resolve_save_dir(args.debug_input_dir, save_root)
    if explicit_out_dir or explicit_light_out_dir or explicit_debug_input_dir:
        per_session_dirs = False

    if no_live_save:
        out_dir = None
        light_out_dir = None
    elif per_session_dirs:
        out_dir = None
        light_out_dir = None
    else:
        out_dir = explicit_out_dir
        if out_dir is None:
            out_dir = os.path.join(
                save_root,
                f"output_{run_timestamp}_{output_model_name}",
            )
        light_out_dir = explicit_light_out_dir
        if light_out_dir is None:
            light_out_dir = derive_light_out_dir(out_dir)

    if no_debug_inputs:
        debug_input_dir = None
    elif per_session_dirs:
        debug_input_dir = None
    else:
        debug_input_dir = explicit_debug_input_dir
        if debug_input_dir is None:
            debug_input_dir = os.path.join(
                save_root,
                f"input_{run_timestamp}_{output_model_name}",
            )

    return AdapterConfig(
        host=args.host,
        port=args.port,
        adapter_model=args.adapter_model,
        main_api_base=args.main_api_base,
        main_model=args.main_model,
        main_backends=tuple(json.loads(args.main_backends)) if args.main_backends else (),
        api_key=args.api_key,
        allowed_local_image_roots=_split_paths(args.allowed_local_image_roots),
        frame_seconds=args.frame_seconds,
        input_format=args.input_format,
        max_pixels=args.max_pixels,
        resize_short_edge=args.resize_short_edge,
        main_max_tokens=args.main_max_tokens,
        main_temperature=args.main_temperature,
        main_top_p=args.main_top_p,
        main_top_k=args.main_top_k,
        main_repetition_penalty=args.main_repetition_penalty,
        main_presence_penalty=args.main_presence_penalty,
        honor_inbound_generation_params=args.honor_inbound_generation_params,
        chunk=args.chunk,
        retain_frames=args.retain_frames,
        compress_every_n_chunks=args.compress_every_n_chunks,
        async_summary_lead_frames=args.async_summary_lead_frames,
        use_prompt_as_query=not args.no_prompt_as_query,
        force_silence_before_query=args.force_silence_before_query,
        keep_qa_history=not args.no_qa_history,
        normalize_output=not args.no_normalize_output,
        first_action_logprob_gate=args.first_action_logprob_gate,
        enable_summarizer=not args.disable_summarizer,
        summarizer_model=args.summarizer_model,
        summarizer_api_base=args.summarizer_api_base,
        longterm_model=args.longterm_model,
        longterm_api_base=args.longterm_api_base,
        summarizer_max_pixels=args.summarizer_max_pixels,
        summarizer_key_frames=args.summarizer_key_frames,
        summarizer_phase_seconds=args.summarizer_phase_seconds,
        mid_term_max_tokens=args.mid_term_max_tokens,
        mid_term_target_tokens=args.mid_term_target_tokens,
        long_term_max_tokens=args.long_term_max_tokens,
        long_term_target_tokens=args.long_term_target_tokens,
        mid_term_temperature=args.mid_term_temperature,
        mid_term_top_p=args.mid_term_top_p,
        mid_term_top_k=args.mid_term_top_k,
        mid_term_repetition_penalty=args.mid_term_repetition_penalty,
        mid_term_presence_penalty=args.mid_term_presence_penalty,
        long_term_temperature=args.long_term_temperature,
        long_term_top_p=args.long_term_top_p,
        long_term_top_k=args.long_term_top_k,
        long_term_repetition_penalty=args.long_term_repetition_penalty,
        long_term_presence_penalty=args.long_term_presence_penalty,
        long_term_memory_window=args.long_term_memory_window,
        request_timeout_seconds=args.request_timeout_seconds,
        session_timeout_seconds=args.session_timeout_seconds,
        max_active_sessions=args.max_active_sessions,
        out_dir=out_dir,
        light_out_dir=light_out_dir,
        debug_input_dir=debug_input_dir,
        save_root=save_root if per_session_dirs else None,
        output_model_name=output_model_name,
        per_session_dirs=per_session_dirs,
        save_model_inputs=not args.no_save_model_inputs,
        save_debug_inputs=not no_debug_inputs,
        summarizer_debug=not args.no_summarizer_debug,
        frame_save_dir=args.frame_save_dir,
        language=args.language,
        system_prompt=args.system_prompt,
    )
