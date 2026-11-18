# -*- coding: utf-8 -*-
"""为 vLLM 准备可加载的 checkpoint 目录（原 Web_Inference/webinfer/run.sh 中 normalize_model_path_for_vllm 的 Python 段）。

原理：新版 transformers 保存的 tokenizer_config.json 里 extra_special_tokens 是 list，
而 vLLM 环境中的 transformers 要求它是 {name: token} 的 dict，直接加载会报错。
这里不改原 checkpoint：在 runtime_dir 下建一个镜像目录，其余文件全部软链接回原目录，
只重写 tokenizer_config.json，把 list 转成 dict。最后打印 vLLM 应该使用的模型路径。

用法: python fix_tokenizer_config.py <model_path> <runtime_dir>
"""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1]).expanduser().resolve()
runtime_dir = Path(sys.argv[2]).expanduser().resolve()
tokenizer_config = src / "tokenizer_config.json"

# 没有 tokenizer_config 或 extra_special_tokens 已经不是 list：原目录可直接用
if not src.is_dir() or not tokenizer_config.is_file():
    print(src)
    raise SystemExit(0)

with tokenizer_config.open() as f:
    cfg = json.load(f)

extra_special_tokens = cfg.get("extra_special_tokens")
if not isinstance(extra_special_tokens, list):
    print(src)
    raise SystemExit(0)

# 镜像目录名带原路径哈希，不同 checkpoint 互不覆盖
digest = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:10]
dst = runtime_dir / f"{src.name}-tokenizer-fixed-{digest}"
dst.mkdir(parents=True, exist_ok=True)

for item in src.iterdir():
    target = dst / item.name
    if item.name == "tokenizer_config.json":
        continue
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    os.symlink(item, target, target_is_directory=item.is_dir())

# "<|video_pad|>" -> key "video_token"，"<|im_start|>" -> key "im_start_token"
mapping = {}
for token in extra_special_tokens:
    key = str(token).strip("<|>").replace("|", "_")
    if key.endswith("_pad"):
        key = key[:-4] + "_token"
    else:
        key = key + "_token"
    mapping[key] = token

cfg["extra_special_tokens"] = mapping
with (dst / "tokenizer_config.json").open("w") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
    f.write("\n")

with (dst / ".source_model_path").open("w") as f:
    f.write(str(src) + "\n")

print(dst)
