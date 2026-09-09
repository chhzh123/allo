p = "/scratch/hc676/agentstudy/harness/agent.py"
s = open(p).read()

# 1. A login shell reads the profile and can block without a terminal; the
#    trial runner has already set the environment the build needs.
old = '''            done = subprocess.run(["bash", "-lc", cmd], capture_output=True,
                                  text=True, timeout=self.args.tool_timeout, check=False)'''
new = '''            done = subprocess.run(["bash", "-c", cmd], capture_output=True,
                                  text=True, timeout=self.args.tool_timeout,
                                  stdin=subprocess.DEVNULL, check=False)'''
assert s.count(old) == 1
s = s.replace(old, new)

# 2. Record what each call asked for and what it got back. Without this a
#    transcript says a tool ran but not what it did, which is most of what
#    the study wants to read afterwards.
old = '''        trial.record("assistant", text=text[:8000], calls=[c[0] for c in calls],
                     fallback=used_fallback, usage=usage)'''
new = '''        trial.record("assistant", text=text[:8000],
                     calls=[{"name": n, "args": {k: (v[:300] if isinstance(v, str) else v)
                                                 for k, v in a.items()}} for n, a in calls],
                     fallback=used_fallback, usage=usage)'''
assert s.count(old) == 1
s = s.replace(old, new)

old = '''            result = (trial.write_file(arguments.get("path"), arguments.get("content", ""))
                      if name == "write_file" else trial.run(arguments.get("command")))'''
new = '''            result = (trial.write_file(arguments.get("path"), arguments.get("content", ""))
                      if name == "write_file" else trial.run(arguments.get("command")))
            trial.record("tool_result", name=name,
                         args={k: (v[:200] if isinstance(v, str) else v)
                               for k, v in arguments.items()},
                         result=result[:1500])'''
assert s.count(old) == 1
s = s.replace(old, new)

# 3. A stalled request should recover in minutes, not in twenty of them.
s = s.replace('ap.add_argument("--reply-timeout", type=int, default=1200)',
              'ap.add_argument("--reply-timeout", type=int, default=300)')

# 4. Say what we want explicitly rather than relying on a default.
old = '''        payload = {"model": self.args.model, "messages": messages, "tools": TOOLS,
                   "temperature": self.args.temperature}'''
new = '''        payload = {"model": self.args.model, "messages": messages, "tools": TOOLS,
                   "temperature": self.args.temperature, "stream": False}'''
assert s.count(old) == 1
s = s.replace(old, new)
open(p, "w").write(s)
print("agent loop patched: no login shell, calls and results recorded, 5-minute reply timeout")
