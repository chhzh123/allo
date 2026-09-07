#!/usr/bin/env python3
"""One trial: one model, one arm, one task, under a fixed budget.

    agent.py --arm spmw --model <slug> --trial <dir> [--tokens N] [--builds N]

The loop is identical for every arm; only the prompt's five slots, the starting
stub, the reference pack and the build command differ. The key is read from the
environment and never logged: transcripts record messages, tool calls and token
counts, and nothing else.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

TOOLS = [
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Replace one file in your working directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "file name, no directories"},
            "content": {"type": "string", "description": "the complete new contents"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run",
        "description": "Run one allowed command: build, route, cat <path>, or ls.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"]}}},
]


def redact(text, key):
    return text.replace(key, "<key>") if key and key in text else text


class Trial:
    def __init__(self, args, arm):
        self.args, self.arm = args, arm
        self.dir = os.path.abspath(args.trial)
        os.makedirs(self.dir, exist_ok=True)
        # Beside the trial directory, never inside it.
        self.record_path = self.dir.rstrip("/") + ".transcript.jsonl"
        self.log = open(self.record_path, "a", encoding="utf-8")
        self.tokens = 0
        self.builds = 0
        self.started = time.time()
        self.model_seconds = 0.0
        self.tool_seconds = 0.0
        self.submitted = False
        self.build_history = []
        # Where the effort went, segment by segment. A segment is the work
        # between two builds: the model thinking and writing, then the tools
        # running. Reported per build, so a trial reads as a sequence of
        # attempts rather than as one total.
        self.segments = []
        self.seg_tokens = 0
        self.seg_model_seconds = 0.0
        self.routes = 0
        self.route_history = []

    def record(self, kind, **fields):
        fields.update({"kind": kind, "t": round(time.time() - self.started, 2),
                       "tokens": self.tokens, "builds": self.builds})
        self.log.write(json.dumps(fields) + "\n")
        self.log.flush()

    # ---- the two tools ------------------------------------------------------
    def write_file(self, path, content):
        name = os.path.basename(path or "")
        if not name or name != path:
            return f"refused: write to `{path}`; use a bare file name"
        if name in ("tb_study.sv", "TASK.md", "REFERENCE.md"):
            return f"refused: `{name}` belongs to the harness"
        with open(os.path.join(self.dir, name), "w", encoding="utf-8") as handle:
            handle.write(content)
        return f"wrote {name}, {len(content.splitlines())} lines"

    def run(self, command):
        command = (command or "").strip()
        if command == "build":
            if self.builds >= self.args.builds:
                return "refused: no builds left in the budget"
            self.builds += 1
            cmd = self.arm["build"].replace("{trial}", self.dir)
            start = time.time()
            done = subprocess.run(["bash", "-c", cmd], capture_output=True,
                                  text=True, timeout=self.args.tool_timeout,
                                  stdin=subprocess.DEVNULL, check=False)
            tool_seconds = time.time() - start
            self.tool_seconds += tool_seconds
            out = (done.stdout + done.stderr).strip()
            stages = {}
            for line in out.splitlines():
                if line.startswith("STUDY STAGE "):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            stages[parts[2]] = round(float(parts[3]), 2)
                        except ValueError:
                            pass
            self.segments.append({
                "build": self.builds,
                "tokens": self.seg_tokens,
                "model_seconds": round(self.seg_model_seconds, 1),
                "tool_seconds": round(tool_seconds, 1),
                "stages": stages,
                "verdict": next((l.strip() for l in out.splitlines()
                                 if "STUDY RESULT" in l or "STUDY BUILD" in l), None),
            })
            self.record("segment", **self.segments[-1])
            self.seg_tokens = 0
            self.seg_model_seconds = 0.0
            verdict = [l for l in out.splitlines()
                       if ("STUDY " in l and "STUDY STAGE" not in l) or "MISMATCH" in l]
            body = "\n".join(verdict) if verdict else out[-4000:]
            self.build_history.append(body)
            self.record("build", command=cmd, output=body[:8000])
            left = (f"\n\nBudget left: {self.args.tokens - self.tokens} tokens, "
                    f"{self.args.builds - self.builds} builds.")
            return body[:8000] + left
        if command == "route":
            if not self.build_history:
                return "refused: build something that simulates before routing it"
            if self.routes >= self.args.routes:
                return "refused: no routing runs left in the budget"
            self.routes += 1
            rtl = os.path.join(self.dir, self.arm.get("rtl_dir", "sim"))
            out = os.path.join(self.dir, "route_%d" % self.routes)
            start = time.time()
            done_run = subprocess.run(
                ["bash", os.path.join(STUDY, "harness", "pnr_ooc.sh"), rtl, "dut_norm", out],
                capture_output=True, text=True, timeout=self.args.tool_timeout,
                stdin=subprocess.DEVNULL, check=False)
            seconds = time.time() - start
            self.tool_seconds += seconds
            text = (done_run.stdout + done_run.stderr).strip()
            lines = [l for l in text.splitlines() if l.startswith("PNR")]
            body = "\n".join(lines) if lines else text[-2000:]
            self.route_history.append(body)
            self.record("route", seconds=round(seconds, 1), output=body[:2000])
            return body + "\n\nRouting runs left: %d." % (self.args.routes - self.routes)
        if command == "ls":
            allowed = {"TASK.md", "REFERENCE.md", self.arm["FILE"]}
            return "\n".join(sorted(n for n in os.listdir(self.dir) if n in allowed))
        if command.startswith("cat "):
            name = os.path.basename(command[4:].strip())
            allowed = {"TASK.md", "REFERENCE.md", self.arm["FILE"]}
            if name not in allowed:
                return (f"refused: `{name}`; the files here are "
                        + ", ".join(sorted(allowed)))
            path = os.path.join(self.dir, name)
            if not os.path.isfile(path):
                return f"no such file: {name}"
            return open(path, encoding="utf-8", errors="replace").read()[:20000]
        return ("refused: the commands that run are `build`, `route`, "
                "`cat <path>` and `ls`")

    # ---- the model ----------------------------------------------------------
    def ask(self, messages, key):
        payload = {"model": self.args.model, "messages": messages, "tools": TOOLS,
                   "temperature": self.args.temperature, "stream": False}
        request = urllib.request.Request(
            ENDPOINT, data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                     "HTTP-Referer": "https://github.com/chhzh123/allo",
                     "X-Title": "SPMW agent study"})
        start = time.time()
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=self.args.reply_timeout) as reply:
                    body = json.loads(reply.read().decode())
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                detail = redact(str(exc), key)
                self.record("api_error", attempt=attempt, detail=detail[:400])
                if attempt == 3:
                    raise SystemExit(f"the model API kept failing: {detail[:200]}")
                time.sleep(5 * (attempt + 1))
        elapsed = time.time() - start
        self.model_seconds += elapsed
        self.seg_model_seconds += elapsed
        usage = body.get("usage") or {}
        spent = int(usage.get("total_tokens") or 0)
        self.tokens += spent
        self.seg_tokens += spent
        return body["choices"][0]["message"], usage


def fallback_calls(text):
    """A tool call written as a fenced block, for a model that emits no calls.

    Without this a model that formats its intent as text rather than as a tool
    call fails for a formatting reason rather than a design one, which is not
    what the study is measuring. Every use is recorded.
    """
    out = []
    for match in re.finditer(r"```(?:write_file|file)[ \t]+(\S+)\n(.*?)```", text, re.S):
        out.append(("write_file", {"path": match.group(1), "content": match.group(2)}))
    for match in re.finditer(r"```run\n(.*?)```", text, re.S):
        out.append(("run", {"command": match.group(1).strip()}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("spmw", "hls", "rtl"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--tokens", type=int, default=200000)
    ap.add_argument("--builds", type=int, default=40)
    ap.add_argument("--routes", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--tool-timeout", type=int, default=3600)
    ap.add_argument("--wall-seconds", type=int, default=14400)
    ap.add_argument("--reply-timeout", type=int, default=900)
    args = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        sys.exit("OPENROUTER_API_KEY is not set")

    with open(os.path.join(STUDY, "harness", "arms.json"), encoding="utf-8") as handle:
        arm = json.load(handle)[args.arm]
    trial = Trial(args, arm)

    # The working directory: the task, this arm's reference, and its stub.
    import shutil
    shutil.copy(os.path.join(STUDY, "task", "SPEC.md"), os.path.join(trial.dir, "TASK.md"))
    shutil.copy(os.path.join(STUDY, "docs", args.arm, "REFERENCE.md"),
                os.path.join(trial.dir, "REFERENCE.md"))
    stub = os.path.join(STUDY, "harness", "stubs", arm["stub"])
    if not os.path.exists(os.path.join(trial.dir, arm["stub"])):
        shutil.copy(stub, os.path.join(trial.dir, arm["stub"]))

    template = open(os.path.join(STUDY, "harness", "prompt_common.md"), encoding="utf-8").read()
    # A model that spends its whole budget reasoning in one turn never reaches
    # a build; the budget is reported so it can pace itself.
    system = (template.replace("{LANG}", arm["LANG"]).replace("{FILE}", arm["FILE"])
              .replace("{TOP}", arm["TOP"]).replace("{BUDGET}", f"{args.tokens:,}")
              .replace("{STEPS}", str(args.builds))
              .replace("{ROUTES}", str(args.routes)))
    trial.record("start", arm=args.arm, model=args.model, tokens_budget=args.tokens,
                 builds_budget=args.builds, routes_budget=args.routes, temperature=args.temperature, system=system)

    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "Read TASK.md and REFERENCE.md, then build the design."}]
    while (trial.tokens < args.tokens and trial.builds < args.builds
           and time.time() - trial.started < args.wall_seconds):
        message, usage = trial.ask(messages, key)
        text = message.get("content") or ""
        calls = [(c["function"]["name"], json.loads(c["function"]["arguments"] or "{}"))
                 for c in (message.get("tool_calls") or [])]
        used_fallback = False
        if not calls:
            calls = fallback_calls(text)
            used_fallback = bool(calls)
        trial.record("assistant", text=text[:8000],
                     calls=[{"name": n, "args": {k: (v[:300] if isinstance(v, str) else v)
                                                 for k, v in a.items()}} for n, a in calls],
                     fallback=used_fallback, usage=usage)
        messages.append({k: v for k, v in message.items() if k in ("role", "content", "tool_calls")})

        if re.search(r"^\s*SUBMIT\s*$", text, re.M):
            trial.submitted = True
            trial.record("submit")
            break
        if not calls:
            messages.append({"role": "user", "content":
                             "Use write_file or run. Write SUBMIT alone on a line when done."})
            continue
        for index, (name, arguments) in enumerate(calls):
            result = (trial.write_file(arguments.get("path"), arguments.get("content", ""))
                      if name == "write_file" else trial.run(arguments.get("command")))
            trial.record("tool_result", name=name,
                         args={k: (v[:200] if isinstance(v, str) else v)
                               for k, v in arguments.items()},
                         result=result[:1500])
            if message.get("tool_calls") and not used_fallback:
                messages.append({"role": "tool", "content": result[:12000],
                                 "tool_call_id": message["tool_calls"][index]["id"]})
            else:
                messages.append({"role": "user", "content": result[:12000]})

    trial.record("end", submitted=trial.submitted,
                 stopped_by=("submit" if trial.submitted else
                             "tokens" if trial.tokens >= args.tokens else
                             "builds" if trial.builds >= args.builds else "wall"), model_seconds=round(trial.model_seconds, 1),
                 tool_seconds=round(trial.tool_seconds, 1),
                 wall_seconds=round(time.time() - trial.started, 1),
                 last_build=(trial.build_history[-1][:2000] if trial.build_history else None))
    summary = {"arm": args.arm, "model": args.model, "trial": trial.dir,
               "segments": trial.segments,
               "stopped_by": ("submit" if trial.submitted else
                              "tokens" if trial.tokens >= args.tokens else
                              "builds" if trial.builds >= args.builds else "wall"),
               "tokens": trial.tokens, "builds": trial.builds, "routes": trial.routes, "submitted": trial.submitted,
               "model_seconds": round(trial.model_seconds, 1),
               "tool_seconds": round(trial.tool_seconds, 1),
               "wall_seconds": round(time.time() - trial.started, 1)}
    with open(trial.dir.rstrip("/") + ".summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
