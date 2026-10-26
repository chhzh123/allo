# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remeasure the mini-TPU's simulation times on a quiet machine, interleaved.

    python3 -u e9_simtime_controlled.py <tree> <out> [phase...]

`<tree>` is a checkout of this branch (its `allo/_mlir` linked to a build of
the same C++), `<out>` where the records go. The phases, in the order given
(default all of them):

  sim64   SPMW's OpenMP simulator, FIFO depth floor 64, a thread a unit: three
          processes a case, each a build, a first run and five more runs
  sim0    the same at the depths the design declares: two processes, two more
          runs each
  cores   the simulator at floor 64 pinned to 12, 24 and 48 CPUs (one socket's
          cores, both sockets' cores, every hyperthread), LLaMA at each size
  cosim   E9's RTL cosimulation through Vivado, two runs a case
  csim    Vitis HLS's own C simulation, two runs a case

The cases are LLaMA and DeepSeek at 4x4, 8x8 and 16x16. The machine is shared,
so every run waits until the whole machine is using at most `GATE` cores, and
a run during which the rest of the machine used more than `ACCEPT` cores on
average is kept as rejected and taken again, up to `ATTEMPTS` times. Sizes and
programs are interleaved, rotating the order each repetition, so a slow drift
in the machine's state lands on every case alike.

What everyone else was doing is measured, not inferred from the load average,
which counts this measurement's own threads: `ext_load.py --monitor` logs the
machine's busy cores, this user's and everyone else's every two seconds, and
the simulator and csim records carry their own reading for each timed run.
"""

import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

GATE = 2.0  # cores the whole machine may be using when a run starts
ACCEPT = 2.5  # mean cores the rest of the machine may use during a run
ATTEMPTS = 3
WAIT_POLL_S = 60
DEADLINE_S = 30 * 3600
PROGRAMS = ("llama", "dsv4")
SIZES = (4, 8, 16)
ROTATIONS = ((4, 8, 16), (8, 16, 4), (16, 4, 8))
CPUSETS = {"12": "0-11", "24": "0-23", "48": "0-47"}
TICK = os.sysconf("SC_CLK_TCK")


class Session:
    def __init__(self, tree, out):
        self.tree, self.out = tree, out
        os.makedirs(os.path.join(out, "logs"), exist_ok=True)
        self.start = time.time()
        self.records = open(os.path.join(out, "records.ndjson"), "a", encoding="utf-8")
        self.log = open(os.path.join(out, "logs", "driver.log"), "a", encoding="utf-8")
        here = os.path.dirname(os.path.abspath(__file__))
        self.monitor_path = os.path.join(out, "monitor.ndjson")
        self.monitor = subprocess.Popen(
            [sys.executable, os.path.join(here, "ext_load.py"), "--monitor", "2"],
            stdout=open(self.monitor_path, "a", encoding="utf-8"),
            stderr=subprocess.DEVNULL,
        )
        self.say(f"START pid={os.getpid()} monitor={self.monitor.pid} tree={tree}")

    def say(self, text):
        self.log.write(f"{time.strftime('%F %T')} {text}\n")
        self.log.flush()

    def close(self):
        self.monitor.terminate()
        self.monitor.wait(timeout=30)
        self.say("DONE")

    # -- the machine ---------------------------------------------------------

    @staticmethod
    def busy_cores(window=10.0):
        def busy():
            with open("/proc/stat", encoding="utf-8") as handle:
                v = list(map(int, handle.readline().split()[1:9]))
            return (sum(v) - v[3] - v[4]) / TICK

        b0, t0 = busy(), time.time()
        time.sleep(window)
        return (busy() - b0) / (time.time() - t0)

    def gate(self):
        """Wait until the whole machine, this user included, is nearly idle."""
        waited = 0.0
        while True:
            cores = self.busy_cores()
            if cores <= GATE:
                return waited, round(cores, 2)
            if time.time() - self.start > DEADLINE_S:
                raise TimeoutError("the machine never went quiet before the deadline")
            if (
                waited == 0.0
                or int(waited) // 1800 != int(waited + WAIT_POLL_S + 10) // 1800
            ):
                self.say(f"  waiting: machine busy {cores:.1f} cores")
            time.sleep(WAIT_POLL_S)
            waited += WAIT_POLL_S + 10

    def monitored(self, t0, t1):
        """Everyone else's and this user's mean cores between two times."""
        others, mine = [], []
        with open(self.monitor_path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if t0 <= r["t"] <= t1 + 2:
                    others.append(r["others_cores"])
                    mine.append(r["mine_cores"])
        if not others:
            return None, None, None
        return (
            round(statistics.mean(others), 2),
            round(max(others), 2),
            round(statistics.mean(mine), 2),
        )

    # -- one case -------------------------------------------------------------

    def run(self, item, measure, judge):
        """Gate, measure, judge; take it again while the machine was busy."""
        best = None
        for attempt in range(1, ATTEMPTS + 1):
            waited, quiet = self.gate()
            t0 = time.time()
            rec = measure()
            t1 = time.time()
            rec.update(item)
            others, peak, mine = self.monitored(t0, t1)
            rec.update(
                attempt=attempt,
                gate_wait_s=round(waited),
                gate_busy_cores=quiet,
                t_start=round(t0, 1),
                t_end=round(t1, 1),
                monitor_others_mean=others,
                monitor_others_max=peak,
                monitor_mine_mean=mine,
            )
            noisy = judge(rec)
            rec["accepted"] = noisy is None
            rec["rejected_for"] = noisy
            self.records.write(json.dumps(rec) + "\n")
            self.records.flush()
            self.say(
                f"{item} attempt={attempt} waited={round(waited)}s "
                f"{'ok' if noisy is None else 'REJECTED: ' + noisy}"
            )
            if noisy is None:
                return rec
            best = rec
        return best

    # -- the phases ---------------------------------------------------------------

    def env(self, **extra):
        env = dict(os.environ)
        env.update({k: str(v) for k, v in extra.items()})
        return env

    def simtime(self, args, env, timeout=3600, prefix=()):
        script = os.path.join(self.tree, "experiments/e9_vta/scripts/e9_simtime.py")
        done = subprocess.run(
            [*prefix, sys.executable, script, *map(str, args)],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        lines = [ln for ln in done.stdout.splitlines() if ln.startswith("{")]
        if done.returncode or not lines:
            return {"rc": done.returncode, "error": done.stderr[-600:]}
        rec = json.loads(lines[-1])
        rec["rc"] = done.returncode
        return rec

    @staticmethod
    def judge_runs(rec):
        if rec.get("rc"):
            return f"rc={rec['rc']}"
        runs = list(rec.get("runs", []))
        if "build_run" in rec:
            runs.append(rec["build_run"])
        others = [r["others_cores"] for r in runs if r.get("others_cores") is not None]
        if not others:
            return "no reading"
        if statistics.mean(others) > ACCEPT or max(others) > 2 * ACCEPT:
            return f"others used {statistics.mean(others):.1f} cores (max {max(others):.1f})"
        if not all(r.get("pass") for r in rec.get("runs", [])):
            return "a result was wrong"
        return None

    def sim(self, floor, processes, repeat):
        for rep in range(1, processes + 1):
            programs = PROGRAMS if rep % 2 else PROGRAMS[::-1]
            for size in ROTATIONS[(rep - 1) % 3]:
                for program in programs:
                    item = {
                        "phase": f"sim{floor}",
                        "program": program,
                        "size": size,
                        "rep": rep,
                    }
                    env = self.env(ALLO_SIM_DEPTH=floor, ALLO_SIM_REPEAT=repeat)
                    self.run(
                        item,
                        lambda: self.simtime((program, size), env),
                        self.judge_runs,
                    )

    def cores(self, repeat=3):
        names = tuple(CPUSETS)
        for k in range(3):
            cpuset = names[k]
            for size in ROTATIONS[k]:
                item = {
                    "phase": "cores",
                    "program": "llama",
                    "size": size,
                    "cpus": int(cpuset),
                }
                env = self.env(ALLO_SIM_DEPTH=64, ALLO_SIM_REPEAT=repeat)
                prefix = ("taskset", "-c", CPUSETS[cpuset])
                self.run(
                    item,
                    lambda: self.simtime(("llama", size), env, prefix=prefix),
                    self.judge_runs,
                )

    def csim(self, processes=2):
        for rep in range(1, processes + 1):
            programs = PROGRAMS if rep % 2 else PROGRAMS[::-1]
            for size in ROTATIONS[(rep - 1) % 3]:
                for program in programs:
                    item = {
                        "phase": "csim",
                        "program": program,
                        "size": size,
                        "rep": rep,
                    }
                    where = os.path.join(
                        self.out, "runs", f"csim_S{size}_{program}_r{rep}"
                    )

                    def measure(program=program, size=size, where=where):
                        shutil.rmtree(where, ignore_errors=True)
                        rec = self.simtime(
                            (program, size, "csim", where), self.env(), 7200
                        )
                        shutil.rmtree(where, ignore_errors=True)
                        return rec

                    def judge(rec):
                        if (
                            rec.get("rc")
                            or not rec.get("pass")
                            or not rec.get("hot_pass")
                        ):
                            return "failed: " + str(rec.get("error", ""))[:80]
                        runs = [rec["vitis_run"], *rec.get("hot_runs", [])]
                        worst = max(r["others_cores"] for r in runs)
                        return (
                            None
                            if worst <= ACCEPT
                            else f"others used {worst:.1f} cores"
                        )

                    self.run(item, measure, judge)

    def cosim(self, processes=2):
        build = os.path.join(self.tree, "scripts/spmw_build_array.py")
        for rep in range(1, processes + 1):
            programs = PROGRAMS if rep % 2 else PROGRAMS[::-1]
            for size in ROTATIONS[(rep - 1) % 3]:
                for program in programs:
                    item = {
                        "phase": "cosim",
                        "program": program,
                        "size": size,
                        "rep": rep,
                    }
                    where = os.path.join(
                        self.out, "runs", f"cosim_S{size}_{program}_r{rep}"
                    )
                    log = os.path.join(
                        self.out, "logs", f"cosim_S{size}_{program}_r{rep}.log"
                    )

                    def measure(program=program, size=size, where=where, log=log):
                        shutil.rmtree(where, ignore_errors=True)
                        t0 = time.time()
                        with open(log, "w", encoding="utf-8") as handle:
                            rc = subprocess.run(
                                [
                                    sys.executable,
                                    build,
                                    "--design",
                                    f"ptpu-{program}",
                                    "--size",
                                    str(size),
                                    "--frequency",
                                    "300",
                                    "--jobs",
                                    "6",
                                    "--cosim",
                                    "--out",
                                    where,
                                ],
                                stdout=handle,
                                stderr=subprocess.STDOUT,
                                timeout=4 * 3600,
                                check=False,
                            ).returncode
                        t1 = time.time()
                        text = open(log, encoding="utf-8").read()
                        rec = {"rc": rc, "wall_s": round(t1 - t0, 1)}
                        rec["pass"] = "SPMW COSIM PASS" in text
                        for key, pattern in (
                            ("cycles", r"SPMW CYCLES total=(\d+)"),
                            ("hls_wall_s", r"HLS: ([\d.]+)s wall"),
                            ("elaborate_s", r"elaborated \d+ instances in ([\d.]+)s"),
                            ("staging_s", r"in ([\d.]+)s \(frontend"),
                        ):
                            found = re.search(pattern, text)
                            rec[key] = float(found.group(1)) if found else None
                        try:
                            cost = json.load(
                                open(os.path.join(where, "cost.json"), encoding="utf-8")
                            )
                            rec["cosim_steps_s"] = cost.get("cosim_seconds")
                        except (OSError, ValueError):
                            rec["cosim_steps_s"] = None
                        xsim = (rec["cosim_steps_s"] or {}).get("xsim")
                        if xsim:
                            others, peak, mine = self.monitored(t1 - xsim - 2, t1)
                            rec.update(
                                xsim_others_mean=others,
                                xsim_others_max=peak,
                                xsim_mine_mean=mine,
                            )
                        shutil.rmtree(where, ignore_errors=True)
                        return rec

                    def judge(rec):
                        if rec.get("rc") or not rec.get("pass"):
                            return f"failed rc={rec.get('rc')}"
                        whole = rec.get("monitor_others_mean")
                        during = rec.get("xsim_others_mean")
                        if whole is None or whole > ACCEPT:
                            return f"others used {whole} cores over the run"
                        if during is not None and during > ACCEPT:
                            return f"others used {during} cores during xsim"
                        return None

                    self.run(item, measure, judge)


def main():
    tree, out = sys.argv[1], sys.argv[2]
    phases = sys.argv[3:] or ["sim64", "sim0", "cores", "cosim", "csim"]
    session = Session(tree, out)
    try:
        for phase in phases:
            session.say(f"PHASE {phase}")
            if phase == "sim64":
                session.sim(64, processes=3, repeat=5)
            elif phase == "sim0":
                session.sim(0, processes=2, repeat=2)
            elif phase == "cores":
                session.cores()
            elif phase == "cosim":
                session.cosim()
            elif phase == "csim":
                session.csim()
            else:
                session.say(f"unknown phase {phase}")
    except Exception as exc:  # pylint: disable=broad-except
        session.say(f"STOPPED: {exc!r}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
