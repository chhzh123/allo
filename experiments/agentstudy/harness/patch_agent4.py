p = "/scratch/hc676/agentstudy/harness/agent.py"
s = open(p).read()

# The working directory must hold only the task, the reference and the design.
# A log kept inside it is something the model can read back, and one did:
# 112,000 tokens in a single turn spent reading its own transcript.
old = '''        self.log = open(os.path.join(self.dir, "transcript.jsonl"), "a", encoding="utf-8")'''
new = '''        # Beside the trial directory, never inside it.
        self.record_path = self.dir.rstrip("/") + ".transcript.jsonl"
        self.log = open(self.record_path, "a", encoding="utf-8")'''
assert s.count(old) == 1
s = s.replace(old, new)

old = '''    with open(os.path.join(trial.dir, "summary.json"), "w", encoding="utf-8") as handle:'''
new = '''    with open(trial.dir.rstrip("/") + ".summary.json", "w", encoding="utf-8") as handle:'''
assert s.count(old) == 1
s = s.replace(old, new)

# And refuse to serve anything that is not the model's own working material,
# so a future log or report placed nearby cannot be read back either.
old = '''        if command.startswith("cat "):
            name = os.path.basename(command[4:].strip())
            path = os.path.join(self.dir, name)
            if not os.path.isfile(path):
                return f"no such file: {name}"
            return open(path, encoding="utf-8", errors="replace").read()[:20000]'''
new = '''        if command.startswith("cat "):
            name = os.path.basename(command[4:].strip())
            allowed = {"TASK.md", "REFERENCE.md", self.arm["FILE"]}
            if name not in allowed:
                return (f"refused: `{name}`; the files here are "
                        + ", ".join(sorted(allowed)))
            path = os.path.join(self.dir, name)
            if not os.path.isfile(path):
                return f"no such file: {name}"
            return open(path, encoding="utf-8", errors="replace").read()[:20000]'''
assert s.count(old) == 1
s = s.replace(old, new)

old = '''        if command == "ls":
            return "\\n".join(sorted(os.listdir(self.dir)))'''
new = '''        if command == "ls":
            allowed = {"TASK.md", "REFERENCE.md", self.arm["FILE"]}
            return "\\n".join(sorted(n for n in os.listdir(self.dir) if n in allowed))'''
assert s.count(old) == 1
s = s.replace(old, new)
open(p, "w").write(s)
print("transcript moved out of the working directory; ls and cat serve only the three files")

p = "/scratch/hc676/agentstudy/harness/show_trial.py"
s = open(p).read()
s = s.replace('path = os.path.join(T, "transcript.jsonl")', 'path = T.rstrip("/") + ".transcript.jsonl"')
open(p, "w").write(s)
p = "/scratch/hc676/agentstudy/harness/peek.py"
s = open(p).read()
s = s.replace('p = sys.argv[1] + "/transcript.jsonl"', 'p = sys.argv[1].rstrip("/") + ".transcript.jsonl"')
open(p, "w").write(s)
p = "/scratch/hc676/agentstudy/harness/status.py"
s = open(p).read()
s = s.replace('path = os.path.join(d, "transcript.jsonl")', 'path = d.rstrip("/") + ".transcript.jsonl"')
s = s.replace('for d in sorted(glob.glob(T + "/*__*")):\n    if not os.path.isdir(d):\n        continue',
              'for d in sorted(glob.glob(T + "/*__*")):\n    if not os.path.isdir(d):\n        continue')
open(p, "w").write(s)
print("readers follow the new location")
