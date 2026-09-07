"""Minimal two-turn tool exchange: does the second call return?"""
import json, os, time, urllib.request

KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = os.environ.get("PROBE_MODEL", "deepseek/deepseek-v4-pro")
TOOLS = [{"type": "function", "function": {
    "name": "run", "description": "Run one allowed command: ls or cat <path>.",
    "parameters": {"type": "object",
                   "properties": {"command": {"type": "string"}},
                   "required": ["command"]}}}]


def call(messages, label, timeout=180):
    body = json.dumps({"model": MODEL, "messages": messages, "tools": TOOLS,
                       "temperature": 0.0, "stream": False}).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=body,
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.load(r)
    except Exception as exc:
        print("%-8s FAILED after %.1fs: %s" % (label, time.time() - start, str(exc)[:200]))
        return None
    msg = out["choices"][0]["message"]
    print("%-8s ok in %.1fs, tool_calls=%d, content=%r"
          % (label, time.time() - start,
             len(msg.get("tool_calls") or []), (msg.get("content") or "")[:60]))
    return msg


msgs = [{"role": "system", "content": "You inspect a directory. Use the run tool."},
        {"role": "user", "content": "List the directory, then say DONE."}]
first = call(msgs, "turn 1")
if first:
    msgs.append({k: v for k, v in first.items() if k in ("role", "content", "tool_calls")})
    for tc in (first.get("tool_calls") or []):
        msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": "TASK.md\nREFERENCE.md"})
    if not first.get("tool_calls"):
        msgs.append({"role": "user", "content": "TASK.md\nREFERENCE.md"})
    call(msgs, "turn 2")
