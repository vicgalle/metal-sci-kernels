"""LLM bridge — Claude (via claude_agent_sdk), Gemini (via google-genai),
and OpenAI (via openai SDK).

Adapted from llm-policies-social-dilemmas/llm_self_play.py. Single
``call_llm(system, user, model)`` entry point that returns
``(full_text, reasoning)``.
"""

from __future__ import annotations

import os
import sys

# When running inside a Claude Code session, the SDK refuses to launch
# itself unless this env var is unset.
os.environ.pop("CLAUDECODE", None)


def log(msg: str = "") -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def is_gemini(model: str) -> bool:
    return model.startswith("gemini") or model.startswith("gemma")


def is_openai(model: str) -> bool:
    m = model.lower()
    if m.startswith("openai/"):
        return True
    if m.startswith("gpt") or m.startswith("chatgpt"):
        return True
    # OpenAI reasoning families: o1, o3, o4, ...
    if len(m) >= 2 and m[0] == "o" and m[1].isdigit():
        return True
    return False


def _is_openai_reasoning(model: str) -> bool:
    m = model.lower().removeprefix("openai/")
    if len(m) >= 2 and m[0] == "o" and m[1].isdigit():
        return True
    # gpt-5 and later families are reasoning-capable.
    if m.startswith("gpt-5") or m.startswith("gpt-6"):
        return True
    return False


async def call_llm(system: str, user: str, model: str) -> tuple[str, str]:
    if is_gemini(model):
        return await _call_gemini(system, user, model)
    if is_openai(model):
        return await _call_openai(system, user, model)
    return await _call_claude(system, user, model)


async def _call_claude(system: str, user: str, model: str) -> tuple[str, str]:
    from claude_agent_sdk import (
        query, ClaudeAgentOptions,
        AssistantMessage, ResultMessage, TextBlock, ThinkingBlock,
    )
    from claude_agent_sdk._errors import MessageParseError

    options = ClaudeAgentOptions(
        system_prompt=system,
        max_turns=1,
        model=model,
        effort="high",
    )

    text_parts: list[str] = []
    thinking_parts: list[str] = []

    try:
        async for msg in query(prompt=user, options=options):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)
                    elif isinstance(block, ThinkingBlock):
                        thinking_parts.append(block.thinking)
            elif isinstance(msg, ResultMessage):
                if getattr(msg, "total_cost_usd", None):
                    log(f"  [llm] Claude cost: ${msg.total_cost_usd:.4f}")
    except MessageParseError as e:
        log(f"  [llm] SDK parse error (skipped): {e}")
        if not text_parts:
            raise

    return "\n".join(text_parts), "\n".join(thinking_parts)


async def _call_gemini(system: str, user: str, model: str) -> tuple[str, str]:
    from google import genai
    from google.genai import types

    client = genai.Client()
    response = await client.aio.models.generate_content(
        model=model,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            thinking_config=types.ThinkingConfig(thinking_level="high"),
        ),
    )

    full_text = ""
    reasoning = ""
    if response.candidates and response.candidates[0].content:
        for part in response.candidates[0].content.parts:
            if getattr(part, "thought", False):
                reasoning += getattr(part, "text", "") or ""
            else:
                full_text += getattr(part, "text", "") or ""
    return full_text, reasoning


async def _call_openai(system: str, user: str, model: str) -> tuple[str, str]:
    from openai import AsyncOpenAI

    # Strip optional "openai/" namespace prefix used in the dispatcher.
    api_model = model.removeprefix("openai/")

    client = AsyncOpenAI()

    kwargs: dict = {
        "model": api_model,
        "instructions": system,
        "input": user,
    }
    if _is_openai_reasoning(api_model):
        kwargs["reasoning"] = {"effort": "high", "summary": "auto"}

    response = await client.responses.create(**kwargs)

    full_text = getattr(response, "output_text", "") or ""

    reasoning_parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) == "reasoning":
            for s in getattr(item, "summary", None) or []:
                t = getattr(s, "text", None)
                if t:
                    reasoning_parts.append(t)
    reasoning = "\n".join(reasoning_parts)

    usage = getattr(response, "usage", None)
    if usage is not None:
        in_tok = getattr(usage, "input_tokens", None)
        out_tok = getattr(usage, "output_tokens", None)
        if in_tok is not None and out_tok is not None:
            log(f"  [llm] OpenAI tokens: in={in_tok}, out={out_tok}")

    return full_text, reasoning
