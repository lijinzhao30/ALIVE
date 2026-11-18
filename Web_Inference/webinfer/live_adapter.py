# -*- coding: utf-8 -*-
"""
OpenAI-compatible live adapter for StreamingHarness inference.

The adapter does not load model weights. It accepts the WebUI's ordinary
OpenAI chat-completion requests, stores incoming base64 frames on disk, keeps
StreamingHarness-style session state, and calls the existing vLLM OpenAI API
servers:

  - main 8B inference API
  - optional summary API for mid-term summaries and long-term compression

入口文件。模块划分：
  adapter_common.py   常量、时间戳/输出归一化等纯函数
  adapter_request.py  HTTP 请求解析、图片编码、OpenAI 响应构造
  adapter_config.py   AdapterConfig 与命令行/环境变量解析
  adapter_session.py  SessionState 与会话管理、帧落盘、结果写入
  adapter_model.py    滑窗 messages 构造与 vLLM 主模型调用
  adapter_summary.py  可选的中期/长期摘要记忆
  adapter_core.py     StreamingInferAdapter：HTTP 接口 + 每帧推理主流程
"""

from __future__ import annotations

import logging
import os

from aiohttp import web

from adapter_common import LOGGER
from adapter_config import AdapterConfig, parse_args
from adapter_core import StreamingInferAdapter


def create_app(config: AdapterConfig) -> web.Application:
    adapter = StreamingInferAdapter(config)
    app = web.Application(client_max_size=128 * 1024 * 1024)
    app["adapter"] = adapter
    async def _on_startup(_app: web.Application) -> None:
        adapter.start_background_tasks()
    app.on_startup.append(_on_startup)
    app.router.add_get("/health", adapter.handle_health)
    app.router.add_get("/v1/models", adapter.handle_models)
    app.router.add_post("/v1/chat/completions", adapter.handle_chat_completions)
    app.router.add_post("/v1/streaming/reset", adapter.handle_reset)
    return app


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = parse_args()
    LOGGER.info("Starting adapter on %s:%s", config.host, config.port)
    LOGGER.info("Input format: %s", config.input_format)
    LOGGER.info("Adapter model: %s", config.adapter_model)
    if config.main_backends:
        LOGGER.info("Multi-backend mode: %d backends", len(config.main_backends))
        for b in config.main_backends:
            LOGGER.info("  Backend: %s -> %s (model=%s)", b["name"], b["api_base"], b.get("model", b["name"]))
    else:
        LOGGER.info("Main model: %s at %s", config.main_model, config.main_api_base)
    if config.per_session_dirs:
        LOGGER.info("Live save mode: per-session directories under %s", config.save_root)
    else:
        LOGGER.info("Live output dir: %s", config.out_dir or "disabled")
        LOGGER.info("Live light output dir: %s", config.light_out_dir or "disabled")
        LOGGER.info("Live debug input dir: %s", config.debug_input_dir or "disabled")
    if config.enable_summarizer:
        LOGGER.info(
            "Summarizer APIs: mid=%s long=%s",
            config.summarizer_api_base,
            config.longterm_api_base,
        )
    else:
        LOGGER.info("Summarizer disabled")
    web.run_app(create_app(config), host=config.host, port=config.port)


if __name__ == "__main__":
    main()
