# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Who else is using the CPU, measured, not inferred from the load average.

    python3 ext_load.py [window_s]            # one reading: others' cores over the window
    python3 ext_load.py --monitor <period_s>  # a line every period, until killed
    python3 ext_load.py --top [window_s]      # the heaviest other users and commands

"Others" sums the CPU time of every process not owned by this user over the
window, from /proc/<pid>/stat; the load average cannot do this, since it
counts this user's own runnable threads too. A process that starts and ends
inside the window is missed, which errs towards less interference, not more.
"""
import collections
import json
import os
import sys
import time

ME = os.getuid()
TICK = os.sysconf("SC_CLK_TCK")


def name(uid):
    try:
        import pwd

        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return f"uid{uid}"


def snap():
    out = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/stat") as handle:
                text = handle.read()
            uid = os.stat(f"/proc/{pid}").st_uid
        except OSError:
            continue
        rest = text[text.rfind(")") + 2 :].split()
        comm = text[text.find("(") + 1 : text.rfind(")")]
        out[pid] = (uid, int(rest[11]) + int(rest[12]), comm)
    with open("/proc/stat") as handle:
        cpu = list(map(int, handle.readline().split()[1:9]))
    busy = sum(cpu) - cpu[3] - cpu[4]
    return out, busy


def reading(window):
    a, busy_a = snap()
    t0 = time.time()
    time.sleep(window)
    b, busy_b = snap()
    dt = time.time() - t0
    mine = others = 0.0
    users = collections.Counter()
    for pid, (uid, ticks, comm) in b.items():
        if pid not in a or a[pid][0] != uid:
            continue
        cores = (ticks - a[pid][1]) / TICK / dt
        if uid == ME:
            mine += cores
        else:
            others += cores
            users[(name(uid), comm)] += cores
    total = (busy_b - busy_a) / TICK / dt
    return {
        "t": round(time.time(), 1),
        "total_cores": round(total, 2),
        "mine_cores": round(mine, 2),
        "others_cores": round(others, 2),
        "top_others": [
            [u, c, round(v, 2)] for (u, c), v in users.most_common(3) if v > 0.05
        ],
        "load1": os.getloadavg()[0],
    }


def main():
    if sys.argv[1:2] == ["--monitor"]:
        period = float(sys.argv[2])
        while True:
            print(json.dumps(reading(period)), flush=True)
    elif sys.argv[1:2] == ["--top"]:
        r = reading(float(sys.argv[2]) if len(sys.argv) > 2 else 5)
        print(
            f"total busy {r['total_cores']} cores: mine {r['mine_cores']}, others {r['others_cores']}"
        )
        for u, c, v in r["top_others"]:
            print(f"  {u:10s} {c:24s} {v:6.2f} cores")
    else:
        print(json.dumps(reading(float(sys.argv[1]) if len(sys.argv) > 1 else 5)))


if __name__ == "__main__":
    main()
