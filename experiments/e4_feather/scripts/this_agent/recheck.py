#!/usr/bin/env python3
"""Re-run the checker of a finished RTL run (no re-simulation), keeping the run's timing keys."""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
for out in sys.argv[1:]:
    old = json.load(open(os.path.join(out, "check.json")))
    res = subprocess.run([sys.executable, os.path.join(HERE, "e4_feather_gen.py"), "check", "--out", out, "--log", os.path.join(out, "xsim.log")], capture_output=True, text=True, check=False)
    new = json.loads(res.stdout)
    for k in ("gen_s", "build_s", "sim_s", "xsim_rc", "build", "rtl"):
        if k in old:
            new[k] = old[k]
    json.dump(new, open(os.path.join(out, "check.json"), "w"), indent=1, default=int)
    print(os.path.basename(out), new["status"], "first", new.get("first_output_cycles"), "done", new.get("completion_cycles"), "offsets", new.get("offsets"), "host", new.get("host_check"))
