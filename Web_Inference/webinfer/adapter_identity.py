"""Narrow deterministic replies for questions about this deployed model.

This does not rewrite arbitrary answers or block discussion of public models.
The system prompt covers paraphrases outside these explicit patterns.
"""

import re

PUBLIC_IDENTITY = "ALIVE"


def identity_action(text: str) -> str:
    """Return a natural, language-matched public self-introduction."""
    if re.search(r"[\u3400-\u9fff]", text or ""):
        return "</response> 我是 ALIVE。"
    return "</response> I'm ALIVE."

_DETAIL_ZH = r"(?:训练|训.{0,8}出|微调|基模|基座|底座|基础模型|架构|参数|权重|检查点|数据集|语料|技术来源|技术细节|版本|哪一版|开发|研发|模型|名字|名称|谁|来历)"
_DETAIL_EN = r"(?:train(?:ing|ed)?|fine[- ]?tun\w*|base\s+model|foundation|backbone|architecture|parameters?|weights?|checkpoints?|datasets?|training\s+data|lineage|model|name|develop\w*|creat\w*|version|built|based\s+on)"
_PATTERNS = tuple(re.compile(pattern, re.I) for pattern in (
    r"(?:你|您)(?:是|叫|姓|叫什么|到底是|究竟是).{0,12}(?:谁|什么|哪个|哪家|啥|alive|qwen|通义|千问|gpt|gemini)",
    r"(?:你|您)(?:的|自己的|自身的).{0,18}" + _DETAIL_ZH,
    r"(?:你|您)(?:是|如何|怎么|怎样|在|用|基于|由|有|属于|到底|究竟).{0,30}" + _DETAIL_ZH,
    r"(?:这个|该|当前|此)(?:模型|服务|系统).{0,20}" + _DETAIL_ZH,
    r"(?:你|您)(?:刚才|之前|前面|曾经|先前).{0,40}(?:说|提到).{0,40}" + _DETAIL_ZH,
    r"(?:介绍|说明|说说)(?:一下)?(?:你自己|你自身|你这个模型)",
    r"\b(?:who|what)\s+(?:exactly\s+)?are\s+you\b",
    r"\b(?:what|which)\s+(?:AI\s+)?(?:model|version)\s+are\s+you\b",
    r"\byour\s+(?:own\s+|underlying\s+|original\s+|exact\s+|technical\s+){0,3}" + _DETAIL_EN + r"\b",
    r"\b(?:are|were|have)\s+you\b[^.!?\n]{0,45}\b" + _DETAIL_EN + r"\b",
    r"\b(?:how|who)\s+(?:did|does|was|has)\s+(?:your\s+team|your\s+company|you)\b[^.!?\n]{0,35}\b" + _DETAIL_EN + r"\b",
    r"\b(?:who\s+(?:developed|created|trained|built)\s+you|introduce\s+yourself)\b",
    r"\byou\s+(?:said|mentioned|claimed|stated)\b[^.!?\n]{0,70}\b" + _DETAIL_EN + r"\b",
))


def is_identity_question(text: str) -> bool:
    """Match direct questions about this service, not general ML questions."""
    return any(pattern.search(text or "") for pattern in _PATTERNS)
