"""Thin, provider-direct LLM clients with identical tool-calling semantics.

No router, no litellm. One class per vendor, one neutral message format, exact usage
accounting (prompt, cached-prompt, completion tokens, latency, USD).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from .. import config


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    calls: int = 0

    def add(self, other: "Usage") -> "Usage":
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cached_tokens += other.cached_tokens
        self.cost_usd += other.cost_usd
        self.latency_ms += other.latency_ms
        self.calls += other.calls
        return self


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    assistant_message: dict[str, Any]  # neutral form, appended to history by the caller


# Neutral message format:
#   {"role": "system"|"user", "content": str}
#   {"role": "assistant", "content": str, "tool_calls": [{"id","name","arguments"}], "_raw": provider-specific}
#   {"role": "tool", "tool_call_id": str, "name": str, "content": str}


class LLM:
    def __init__(self, model: str) -> None:
        self.model = model
        self.vendor = "anthropic" if model.startswith("claude") else "openai"
        if self.vendor == "openai":
            from openai import OpenAI

            self._c = OpenAI(api_key=config.env("OPENAI_API_KEY", required=True), max_retries=4, timeout=120)
        else:
            import anthropic

            headers = {}
            ws = config.env("ANTHROPIC_WORKSPACE_ID")
            if ws:
                headers["anthropic-workspace-id"] = ws
            self._c = anthropic.Anthropic(api_key=config.env("ANTHROPIC_API_KEY", required=True),
                                          default_headers=headers, max_retries=4, timeout=180)

    # ------------------------------------------------------------------ public
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
             *, tool_choice: str | None = None, max_tokens: int = 700) -> Turn:
        t0 = time.perf_counter()
        if self.vendor == "openai":
            turn = self._chat_openai(messages, tools, tool_choice, max_tokens)
        else:
            turn = self._chat_anthropic(messages, tools, tool_choice, max_tokens)
        turn.usage.latency_ms = (time.perf_counter() - t0) * 1000
        turn.usage.calls = 1
        turn.usage.cost_usd = config.cost_usd(self.model, turn.usage.prompt_tokens,
                                              turn.usage.completion_tokens, turn.usage.cached_tokens)
        return turn

    def complete_text(self, prompt: str, *, max_tokens: int = 600) -> tuple[str, Usage]:
        turn = self.chat([{"role": "user", "content": prompt}], None, max_tokens=max(max_tokens, 16))
        return turn.text, turn.usage

    # ------------------------------------------------------------------ openai
    def _chat_openai(self, messages, tools, tool_choice, max_tokens) -> Turn:
        """Responses API (required for function tools + reasoning on gpt-5.x)."""
        instructions = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
        items: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "system":
                continue
            if m["role"] == "user":
                items.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                if m.get("_raw"):
                    items.extend(m["_raw"])  # replay reasoning + message + function_call items verbatim
                else:
                    if m.get("content"):
                        items.append({"role": "assistant", "content": m["content"]})
                    for tc in m.get("tool_calls") or []:
                        items.append({"type": "function_call", "call_id": tc["id"], "name": tc["name"],
                                      "arguments": json.dumps(tc["arguments"])})
            elif m["role"] == "tool":
                items.append({"type": "function_call_output", "call_id": m["tool_call_id"], "output": m["content"]})
        kwargs: dict[str, Any] = {"model": self.model, "input": items, "max_output_tokens": max_tokens, "store": False}
        if instructions:
            kwargs["instructions"] = instructions
        if tools:
            kwargs["tools"] = [{"type": "function", "name": t["name"], "description": t.get("description", ""),
                                "parameters": t["parameters"]} for t in tools]
            if tool_choice == "required":
                kwargs["tool_choice"] = "required"
            elif tool_choice and tool_choice != "auto":
                kwargs["tool_choice"] = {"type": "function", "name": tool_choice}
        if self.model.startswith(("gpt-5", "o")):
            kwargs["reasoning"] = {"effort": config.OPENAI_REASONING_EFFORT}
            kwargs["include"] = ["reasoning.encrypted_content"]
        r = self._c.responses.create(**kwargs)
        calls, text_parts, raw = [], [], []
        for item in r.output:
            d = item.model_dump(exclude_none=True)
            d.pop("status", None)
            raw.append(d)
            if item.type == "function_call":
                try:
                    args = json.loads(item.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": item.arguments}
                calls.append(ToolCall(id=item.call_id, name=item.name, arguments=args))
            elif item.type == "message":
                for c in item.content:
                    if getattr(c, "type", "") == "output_text":
                        text_parts.append(c.text)
        u = r.usage
        cached = 0
        if getattr(u, "input_tokens_details", None) is not None:
            cached = getattr(u.input_tokens_details, "cached_tokens", 0) or 0
        usage = Usage(prompt_tokens=u.input_tokens, completion_tokens=u.output_tokens, cached_tokens=cached)
        assistant = {"role": "assistant", "content": "".join(text_parts),
                     "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in calls],
                     "_raw": raw}
        return Turn(text="".join(text_parts), tool_calls=calls, usage=usage, assistant_message=assistant)

    # --------------------------------------------------------------- anthropic
    def _chat_anthropic(self, messages, tools, tool_choice, max_tokens) -> Turn:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
        an_msgs: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "system":
                continue
            if m["role"] == "user":
                an_msgs.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                if m.get("_raw"):
                    an_msgs.append({"role": "assistant", "content": m["_raw"]})
                else:
                    blocks: list[dict[str, Any]] = []
                    if m.get("content"):
                        blocks.append({"type": "text", "text": m["content"]})
                    for tc in m.get("tool_calls") or []:
                        blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]})
                    an_msgs.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(no content)"}]})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if an_msgs and an_msgs[-1]["role"] == "user" and isinstance(an_msgs[-1]["content"], list):
                    an_msgs[-1]["content"].append(block)
                else:
                    an_msgs.append({"role": "user", "content": [block]})
        kwargs: dict[str, Any] = {"model": self.model, "messages": an_msgs, "max_tokens": max_tokens,
                                  "output_config": {"effort": config.ANTHROPIC_EFFORT}}
        if system:
            kwargs["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        if tools:
            an_tools = [{"name": t["name"], "description": t.get("description", ""),
                         "input_schema": t["parameters"]} for t in tools]
            an_tools[-1]["cache_control"] = {"type": "ephemeral"}
            kwargs["tools"] = an_tools
            if tool_choice == "required":
                kwargs["tool_choice"] = {"type": "any"}
            elif tool_choice and tool_choice != "auto":
                kwargs["tool_choice"] = {"type": "tool", "name": tool_choice}
        r = self._c.messages.create(**kwargs)
        text_parts, calls, raw = [], [], []
        for b in r.content:
            raw.append(b.model_dump())
            if b.type == "text":
                text_parts.append(b.text)
            elif b.type == "tool_use":
                calls.append(ToolCall(id=b.id, name=b.name, arguments=dict(b.input or {})))
        u = r.usage
        cached = getattr(u, "cache_read_input_tokens", 0) or 0
        prompt = (u.input_tokens or 0) + cached + (getattr(u, "cache_creation_input_tokens", 0) or 0)
        usage = Usage(prompt_tokens=prompt, completion_tokens=u.output_tokens or 0, cached_tokens=cached)
        assistant = {"role": "assistant", "content": "".join(text_parts),
                     "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in calls],
                     "_raw": raw}
        return Turn(text="".join(text_parts), tool_calls=calls, usage=usage, assistant_message=assistant)


_CACHE: dict[str, LLM] = {}


def get_llm(model: str) -> LLM:
    if model not in _CACHE:
        _CACHE[model] = LLM(model)
    return _CACHE[model]
