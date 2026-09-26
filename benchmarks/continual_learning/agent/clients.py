"""Thin, provider-direct LLM clients with identical tool-calling semantics.

No router, no litellm. One class per vendor, one neutral message format, exact usage
accounting (prompt, cached-prompt, completion tokens, latency, USD).
"""

from __future__ import annotations

import json
import threading
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
    # The gateway's annotations on a served decision (``amfs`` block: trace_id, retrieval path,
    # the context and avoid rows it rendered, the prompt bytes when echoed). Frontier clients
    # leave it ``None``.
    amfs: dict[str, Any] | None = None


# Neutral message format:
#   {"role": "system"|"user", "content": str}
#   {"role": "assistant", "content": str, "tool_calls": [{"id","name","arguments"}], "_raw": provider-specific}
#   {"role": "tool", "tool_call_id": str, "name": str, "content": str}


class LLM:
    # Frontier clients run the full tool-calling loop. Clients that answer one decision per call
    # from a rendered prompt (the Managed Models students and the ``@decision`` control) set this
    # and the harness takes the single-shot branch of ``run_episode``.
    decision_mode: bool = False
    note_writer: str | None = None

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


# ---------------------------------------------------------------------------------------------
# Managed Models students (served through the SenseLab inference gateway) and the frontier
# ``@decision`` control that answers the identical rendered prompt.
#
# Model strings the runner accepts, beyond the frontier ids above:
#   <alias>                         a student from results/students/registry.json, or a raw
#                                   public model id (``senselab-s1-...``)
#   <alias>+notes=<frontier-model>  the same student with a cheap note-writer running the
#                                   reflection turn after each episode (cost charged to the run)
#   <frontier>@decision=<alias>     the control: the gateway renders the prompt for <alias>
#                                   (retrieval, avoid rows, priors — whatever that model's
#                                   serving options say) and <frontier> answers it in one shot.
#                                   ``CL_DECISION_RENDER_MODEL`` supplies <alias> when omitted.
#   <alias>~<variant>               the same model served narrower (SERVING_VARIANTS): the
#                                   gateway's per-request ``serving`` block switches avoid rows,
#                                   priors or context off for an ablation without another model.
# ---------------------------------------------------------------------------------------------

STUDENT_ID_PREFIX = "senselab-"
_DECISION = "@decision"
_NOTES = "+notes="
_VARIANT = "~"
SERVING_VARIANTS: dict[str, dict[str, Any]] = {
    "plain": {"avoid_rows": False, "priors": False},          # memory context only, no outcome record
    "noavoid": {"avoid_rows": False},
    "nopriors": {"priors": False},
    "k0": {"context_k": 0, "avoid_rows": False, "priors": False},  # the bare task: weights alone
    "k1": {"context_k": 1},
    "fallback": {},                                            # placeholder: fallback stays as configured
}
_NOTE_WRITER_DEFAULT = config.env("CL_NOTE_WRITER") or None


@dataclass(frozen=True)
class StudentSpec:
    alias: str
    public_model_id: str
    family: str                 # senselab-s1 | senselab-s1-pro
    price_key: str              # key into config.STUDENT_PRICES
    serve_base: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


def _registry() -> dict[str, Any]:
    try:
        return json.loads(config.STUDENT_REGISTRY.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def resolve_student(name: str) -> StudentSpec:
    """A registry alias, or a raw public model id (family read off its prefix)."""
    reg = _registry().get(name)
    if isinstance(reg, dict) and reg.get("public_model_id"):
        fam = str(reg.get("family") or "senselab-s1")
        return StudentSpec(alias=name, public_model_id=str(reg["public_model_id"]), family=fam,
                           price_key=str(reg.get("price") or fam), serve_base=bool(reg.get("serve_base")),
                           meta={k: v for k, v in reg.items() if k not in ("public_model_id", "family", "price")})
    if name.startswith(STUDENT_ID_PREFIX):
        fam = "senselab-s1-pro" if name.startswith("senselab-s1-pro-") else "senselab-s1"
        return StudentSpec(alias=name, public_model_id=name, family=fam, price_key=fam)
    raise ValueError(f"unknown model {name!r}: not a frontier id, not in {config.STUDENT_REGISTRY}")


def is_student_model(model: str) -> bool:
    base = model.split(_NOTES, 1)[0].split(_VARIANT, 1)[0]
    return _DECISION in base or base.startswith(STUDENT_ID_PREFIX) or base in _registry()


def _openai_shape(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Neutral history -> the OpenAI chat shape the gateway reads ``tried_from_messages`` off.
    System messages are dropped: the gateway's system instruction is the one the student was
    trained on, and the control must see the same bytes."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            calls = [{"id": tc["id"], "type": "function",
                      "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                     for tc in m.get("tool_calls") or []]
            msg: dict[str, Any] = {"role": "assistant", "content": m.get("content") or ""}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
    return out


def parse_action(text: str) -> tuple[str | None, dict[str, Any]]:
    """The gateway's own parser, duplicated so the harness does not import amfs-internal:
    ``{"tool": ..., "arguments": {...}}``, tolerant of code fences and a ``tool_name`` key."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else ""
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
    if not cleaned.startswith("{"):
        i, j = cleaned.find("{"), cleaned.rfind("}")
        if i >= 0 and j > i:
            cleaned = cleaned[i:j + 1]
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None, {}
    if not isinstance(parsed, dict):
        return None, {}
    tool = parsed.get("tool") or parsed.get("tool_name") or parsed.get("name")
    args = parsed.get("arguments") if "arguments" in parsed else parsed.get("input")
    if not isinstance(tool, str) or not isinstance(args, dict):
        return None, {}
    # A decision key in the tool slot — ``resolve:escalate_tier2`` — is the ``tool:action`` form
    # the rendered priors use. Read it as the tool plus its action rather than an unknown tool,
    # so a model is graded on the decision it made, not on where it wrote the colon. Applied
    # to every condition that answers through this parser (students, base-served, decision
    # controls that answer in JSON); an ``action`` already in the arguments wins.
    if ":" in tool and not tool.startswith("<"):
        head, _, action = tool.partition(":")
        if head and action and "action" not in args:
            tool, args = head, {"action": action, **args}
    return tool, args


def _action_turn(text: str, usage: Usage, *, amfs: dict[str, Any] | None, call_id: str) -> Turn:
    name, args = parse_action(text)
    calls = [ToolCall(id=call_id, name=name, arguments=args)] if name else []
    assistant = {"role": "assistant", "content": "" if name else text,
                 "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in calls]}
    return Turn(text=text, tool_calls=calls, usage=usage, assistant_message=assistant, amfs=amfs)


class _Gateway:
    """One HTTP client to the SenseLab inference gateway, paced by the same process-global
    limiter the senselab arm uses (one production API key serves both)."""

    def __init__(self) -> None:
        import httpx

        self.url = config.AMFS_HTTP_URL.rstrip("/") + "/api/v1/chat/completions"
        self._c = httpx.Client(timeout=httpx.Timeout(float(config.env("CL_STUDENT_TIMEOUT") or 120.0), connect=10.0),
                               headers={"Authorization": f"Bearer {config.env('AMFS_API_KEY', required=True)}",
                                        "Content-Type": "application/json"})
        try:
            from ..arms.senselab_arm import _LIMITER
            self._limiter = _LIMITER
        except Exception:  # noqa: BLE001
            self._limiter = None

    def post(self, body: dict[str, Any]) -> dict[str, Any]:
        import httpx

        last: Exception | None = None
        for attempt in range(8):
            if self._limiter is not None:
                self._limiter.acquire()
            try:
                r = self._c.post(self.url, json=body)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = e
                time.sleep(min(2.0 * (attempt + 1), 20.0))
                continue
            if r.status_code in (429, 502, 503, 504) and attempt < 7:
                wait = min(max(float(r.headers.get("Retry-After", 2.0)), 1.0) * (1.5 ** attempt), 30.0)
                time.sleep(wait)
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"gateway {r.status_code}: {r.text[:300]}")
            return r.json()
        raise RuntimeError(f"gateway unreachable: {last}")


_GATEWAY: _Gateway | None = None
_GATEWAY_LOCK = threading.Lock()


def _gateway() -> _Gateway:
    global _GATEWAY
    with _GATEWAY_LOCK:
        if _GATEWAY is None:
            _GATEWAY = _Gateway()
        return _GATEWAY


def _amfs_meta(amfs: dict[str, Any] | None, **extra: Any) -> dict[str, Any]:
    meta = {k: v for k, v in (amfs or {}).items() if v is not None}
    meta.update({k: v for k, v in extra.items() if v is not None})
    return {"amfs": meta}


def _split_variant(name: str) -> tuple[str, dict[str, Any] | None]:
    if _VARIANT not in name:
        return name, None
    alias, variant = name.split(_VARIANT, 1)
    if variant not in SERVING_VARIANTS:
        raise ValueError(f"unknown serving variant {variant!r}; known: {sorted(SERVING_VARIANTS)}")
    return alias, SERVING_VARIANTS[variant] or None


class StudentLLM(LLM):
    """A Managed Models student served by the gateway: one rendered prompt in, one action out.

    The gateway does the retrieval (scoped to ``amfs.entity_path`` — the cell's scope), renders
    the prompt the student was trained on, seals the decision as a trace and returns the
    ``amfs`` block the harness turns into memory hits. ``chat`` keeps the ``LLM`` signature
    plus ``amfs=`` (entity_path, agent_id, outcome_ref, candidate_actions) so the loop can pass
    per-episode scope without touching shared client state.
    """

    decision_mode = True

    def __init__(self, model: str, spec: StudentSpec, *, note_writer: str | None = None,
                 serving: dict[str, Any] | None = None) -> None:  # noqa: D107
        self.model = model
        self.spec = spec
        self.vendor = "senselab"
        self.note_writer = note_writer
        self.serving = serving
        self._gw = _gateway()

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, *,
             tool_choice: str | None = None, max_tokens: int = 700, amfs: dict[str, Any] | None = None) -> Turn:
        t0 = time.perf_counter()
        body = {"model": self.spec.public_model_id, "messages": _openai_shape(messages),
                "max_tokens": min(max_tokens, 400), "temperature": 0,
                "metadata": _amfs_meta(amfs, echo_context=True, serving=self.serving)}
        res = self._gw.post(body)
        text = ((res.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        u = res.get("usage") or {}
        usage = Usage(prompt_tokens=int(u.get("prompt_tokens") or 0), completion_tokens=int(u.get("completion_tokens") or 0))
        usage.latency_ms = (time.perf_counter() - t0) * 1000
        usage.calls = 1
        block = dict(res.get("amfs") or {})
        # A base-served twin or a fallback leg is priced as what actually answered.
        price = self.spec.price_key
        served = block.get("served_by")
        if served == "base" or self.spec.serve_base:
            price = {"senselab-s1": "gemini-2.5-flash-lite", "senselab-s1-pro": "gemini-2.5-flash"}.get(self.spec.family, price)
        usage.cost_usd = config.cost_usd(price, usage.prompt_tokens, usage.completion_tokens)
        block["price_key"] = price
        return _action_turn(text, usage, amfs=block, call_id=f"gw_{res.get('id', 'x')[-12:]}")


class DecisionLLM(LLM):
    """The frontier control for the students: the gateway renders the prompt exactly as it
    would for ``render_for`` (``render_only`` — nothing upstream, nothing sealed, nothing
    billed) and the frontier model answers those bytes in one shot. Same prompt, different
    model: the gap to the student is what the weights learned; the gap to the frontier full
    loop is what the tool loop and the agent's own reflection are worth."""

    decision_mode = True

    def __init__(self, model: str, base: LLM, render_for: StudentSpec, *, note_writer: str | None = None,
                 serving: dict[str, Any] | None = None) -> None:  # noqa: D107
        self.model = model
        self.base = base
        self.render_for = render_for
        self.vendor = base.vendor
        self.note_writer = note_writer
        self.serving = serving
        self._gw = _gateway()

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, *,
             tool_choice: str | None = None, max_tokens: int = 700, amfs: dict[str, Any] | None = None) -> Turn:
        t0 = time.perf_counter()
        body = {"model": self.render_for.public_model_id, "messages": _openai_shape(messages),
                "metadata": _amfs_meta(amfs, render_only=True, serving=self.serving)}
        res = self._gw.post(body)
        block = dict(res.get("amfs") or {})
        prompt = block.get("prompt") or {}
        render_ms = (time.perf_counter() - t0) * 1000
        turn = self.base.chat([{"role": "system", "content": prompt.get("system") or ""},
                               {"role": "user", "content": prompt.get("user") or ""}], None, max_tokens=max_tokens)
        block["render_ms"] = round(render_ms, 1)
        block["served_by"] = f"control:{self.base.model}"
        out = _action_turn(turn.text, turn.usage, amfs=block, call_id=f"dc_{res.get('id', 'x')[-12:]}")
        out.usage.latency_ms += render_ms
        return out

    def complete_text(self, prompt: str, *, max_tokens: int = 600) -> tuple[str, Usage]:
        return self.base.complete_text(prompt, max_tokens=max_tokens)


def _build(model: str) -> LLM:
    note_writer = _NOTE_WRITER_DEFAULT
    base_str = model
    if _NOTES in model:
        base_str, note_writer = model.split(_NOTES, 1)
        note_writer = note_writer or None
    if _DECISION in base_str:
        frontier, _, alias = base_str.partition(_DECISION)
        alias = alias.lstrip("=") or (config.env("CL_DECISION_RENDER_MODEL") or "")
        if not alias:
            raise ValueError(f"{model!r}: name the model to render for (…@decision=<alias>) or set CL_DECISION_RENDER_MODEL")
        alias, serving = _split_variant(alias)
        return DecisionLLM(model, get_llm(frontier), resolve_student(alias), note_writer=note_writer, serving=serving)
    alias, serving = _split_variant(base_str)
    if alias.startswith(STUDENT_ID_PREFIX) or alias in _registry():
        return StudentLLM(model, resolve_student(alias), note_writer=note_writer, serving=serving)
    return LLM(model)


_CACHE: dict[str, LLM] = {}


def get_llm(model: str) -> LLM:
    if model not in _CACHE:
        _CACHE[model] = _build(model)
    return _CACHE[model]
