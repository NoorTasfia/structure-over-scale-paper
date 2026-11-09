"""Prompted-JSON tool loop. The same protocol is used for every model/precision so tool-call
validity is comparable across conditions (it is an RQ3 measurement, not something to hide).

Gemma 4 often falls back to its native call syntax, `<|tool_call>call:name(arg=...)`,
despite the JSON instructions. parse_call accepts both: refusing the model's native
format would measure our protocol choice, not the model."""
import ast
import json
import re
import time
from collections import Counter

import openai

from .tools import TOOL_DOCS, TOOLSETS

TOOL_PARAM_ORDER = {
    "list_dir": ["path"], "read_file": ["path", "start", "end"],
    "grep": ["pattern", "path"], "outline": ["path"],
    "find_definition": ["name"], "find_references": ["name"],
    "callees": ["path", "function"], "submit": ["files", "functions"],
}

NATIVE_CALL = re.compile(r"<\|tool_call\|?>\s*(?:call:)?\s*(\w+\(.*)", re.S)

MAX_OBS_CHARS = 4000
KEEP_FULL_OBSERVATIONS = 3  # older observations are elided from the API view

SYSTEM = """You are a software engineer localizing a bug in a Python repository.
Find the files and functions that must be edited to resolve the issue. You cannot edit code.

Each turn, think briefly, then reply with exactly one JSON block:
```json
{{"tool": "<name>", "args": {{...}}}}
```
Tools:
{tools}
- submit(files, functions) - final answer, best first: up to 5 file paths and up to 5 functions as "path::Qual.name"
{repo_map}"""


def _parse_pycall(snippet):
    """Parse `name(arg, kw=...)` with literal arguments; positional args are mapped
    by the tool's parameter order."""
    try:
        node = ast.parse(snippet.strip(), mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return None
    name = node.func.id
    order = TOOL_PARAM_ORDER.get(name, [])
    args = {}
    try:
        for i, a in enumerate(node.args):
            if i >= len(order):
                return None
            args[order[i]] = ast.literal_eval(a)
        for kw in node.keywords:
            if kw.arg is None:
                return None
            args[kw.arg] = ast.literal_eval(kw.value)
    except (ValueError, SyntaxError):
        return None
    return name, args


def _json_call(raw):
    try:
        call = json.loads(raw)
    except json.JSONDecodeError:
        missing = raw.count("{") - raw.count("}")
        if not 0 < missing <= 3:
            return None
        try:
            call = json.loads(raw + "}" * missing)
        except json.JSONDecodeError:
            return None
    if not (isinstance(call, dict) and isinstance(call.get("tool"), str)):
        return None
    args = call.get("args")
    if not isinstance(args, dict):
        # Flattened form: {"tool": "submit", "files": [...]} — args at the top level.
        args = {k: v for k, v in call.items() if k not in ("tool", "args")}
    name = call["tool"]
    if "(" in name:
        # Fused form: {"tool": "grep(pattern=..., path=...)"} — a Python call in the tool field.
        pycall = _parse_pycall(name)
        if pycall is None:
            return None
        return pycall[0], {**pycall[1], **args}
    return name, args


def parse_call(text):
    # Gemma 4 native syntax, last occurrence wins. The call is cut at its line end;
    # ast then validates the full expression.
    for m in reversed(NATIVE_CALL.findall(text)):
        call = _parse_pycall(m.splitlines()[0])
        if call:
            return call
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*(?:```|$)", text, re.S)
    candidates = blocks or re.findall(r"\{.*\}", text, re.S)
    for raw in reversed(candidates):
        call = _json_call(raw)
        if call:
            return call
    return None


def api_view(messages):
    """Bound prompt growth: keep system + issue + all assistant turns, but elide
    every observation except the last KEEP_FULL_OBSERVATIONS. The stored
    trajectory keeps the full text; only what is sent to the model shrinks."""
    obs_idx = [i for i, m in enumerate(messages[2:], 2)
               if m["role"] == "user" and i > 2][: -KEEP_FULL_OBSERVATIONS or None]
    elide = set(obs_idx)
    return [
        {"role": m["role"], "content": "Observation: [elided to save context]"}
        if i in elide else m
        for i, m in enumerate(messages)
    ]


def run_episode(client, model, ws, issue, toolset, max_turns=20):
    names = TOOLSETS[toolset]
    repo_map = ""
    if toolset == "C":
        repo_map = "\nRepository map (files most related to the issue):\n" + ws.repo_map(issue)
    system = SYSTEM.format(tools="\n".join("- " + TOOL_DOCS[n] for n in names), repo_map=repo_map)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Issue:\n{issue}"},
    ]
    stats = Counter()
    started = time.time()
    result = {"files": [], "functions": [], "end": "max_turns"}

    for _ in range(max_turns):
        try:
            resp = client.chat.completions.create(model=model, messages=api_view(messages),
                                                  temperature=0, max_tokens=1024)
        except openai.BadRequestError as e:
            result["end"] = f"bad_request: {str(e)[:200]}"
            break
        reply = resp.choices[0].message.content or ""
        stats["turns"] += 1
        if resp.usage:
            stats["prompt_tokens"] += resp.usage.prompt_tokens
            stats["completion_tokens"] += resp.usage.completion_tokens
        messages.append({"role": "assistant", "content": reply})

        call = parse_call(reply)
        if call is None:
            stats["invalid_calls"] += 1
            obs = "Error: no valid JSON tool call found. Reply with exactly one ```json block."
        elif call[0] == "submit":
            args = call[1]
            result = {
                "files": [str(f).removeprefix("./") for f in args.get("files", [])][:5],
                "functions": [str(f) for f in args.get("functions", [])][:5],
                "end": "submit",
            }
            break
        elif call[0] not in names:
            stats["unknown_tool"] += 1
            obs = f"Error: unknown tool {call[0]!r}. Available: {', '.join(names)}, submit."
        else:
            stats[f"tool:{call[0]}"] += 1
            try:
                obs = getattr(ws, call[0])(**call[1])
            except Exception as e:
                stats["tool_errors"] += 1
                obs = f"Error: {type(e).__name__}: {e}"
        if len(obs) > MAX_OBS_CHARS:
            obs = obs[:MAX_OBS_CHARS] + "\n... [truncated]"
        if stats["turns"] == max_turns - 1:
            obs += "\n\nThis is your LAST turn: you MUST call submit now with your best guess."
        messages.append({"role": "user", "content": f"Observation:\n{obs}"})

    if result["end"] == "max_turns":
        # Force a final answer so a miss means "navigated wrong", not "never answered".
        messages.append({"role": "user", "content":
                         "Out of turns. Reply with ONLY the submit call, using your best guess so far."})
        try:
            resp = client.chat.completions.create(model=model, messages=api_view(messages),
                                                  temperature=0, max_tokens=512)
            reply = resp.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": reply})
            call = parse_call(reply)
            if call and call[0] == "submit":
                args = call[1]
                result = {
                    "files": [str(f).removeprefix("./") for f in args.get("files", [])][:5],
                    "functions": [str(f) for f in args.get("functions", [])][:5],
                    "end": "forced_submit",
                }
        except openai.OpenAIError:
            pass

    result["stats"] = dict(stats)
    result["seconds"] = round(time.time() - started, 1)
    result["messages"] = messages
    return result
