#!/scratch/hc676/allo-agent/bin/python3
"""Per-transform events and RTL I/O data from a port-level VCD of the Vitis HLS cosim autotb (DUT ports only).
usage: cosim_events.py <vcd> <N> <UF> <NT> [--dump PREFIX]
Rising ap_clk edges are counted (cycle 1 = first edge). Handshakes are sampled with the values that were stable
before the edge's timestamp group (xsim records same-timestamp updates after the edge). With --dump, the data on
in_r_dout / out_r_din at each handshake is unpacked (hls::vector<complex<float>,2UF>: element u in bits [64u+63:64u],
real low word, imag high word) into PREFIX_inputs.txt / PREFIX_outputs.txt in 'rehex imhex' lines."""
import sys, json, re, statistics
a = sys.argv[1:]; dump = None
if "--dump" in a:
    i = a.index("--dump"); dump = a[i + 1]; del a[i:i + 2]
vcd, N, UF, NT = a[0], int(a[1]), int(a[2]), int(a[3])
B = N // (2 * UF); W = 128 * UF
names = {"clk": r"^ap_clk$", "start": r"^ap_start$", "done": r"^ap_done$", "ready": r"^ap_ready$", "idle": r"^ap_idle$",
         "in_read": r"^in(_r)?_read$", "in_empty_n": r"^in(_r)?_empty_n$", "in_dout": r"^in(_r)?_dout$",
         "out_write": r"^out(_r)?_write$", "out_full_n": r"^out(_r)?_full_n$", "out_din": r"^out(_r)?_din$",
         "out_we": r"^dataOut_we0$", "in_ce": r"^dataIn_ce0$"}
ids = {}; want = {}
val = {k: "0" for k in names}
cycle = 0; in_beats = []; out_beats = []; dones = []; readys = []; start_cycles = 0
in_data = []; out_data = []
def edge():
    global cycle, start_cycles
    cycle += 1
    if val["in_read"] == "1" and val["in_empty_n"] == "1":
        in_beats.append(cycle); in_data.append(val["in_dout"]) if dump else None
    if val["out_write"] == "1" and val["out_full_n"] == "1":
        out_beats.append(cycle); out_data.append(val["out_din"]) if dump else None
    if val["out_we"] == "1": out_beats.append(cycle)
    if val["in_ce"] == "1": in_beats.append(cycle)
    if val["done"] == "1": dones.append(cycle)
    if val["ready"] == "1": readys.append(cycle)
    if val["start"] == "1": start_cycles += 1
with open(vcd) as f:
    for line in f:
        line = line.strip()
        if line.startswith("$var"):
            t = line.split(); code, name = t[3], t[4]
            for k, pat in names.items():
                if k not in want and re.match(pat, name): want[k] = code; ids[code] = k
        elif line.startswith("$enddefinitions"): break
    pending = {}
    def flush():
        c = want.get("clk")
        if c in pending and pending[c] == "1" and val["clk"] == "0": edge()
        for code, v in pending.items(): val[ids[code]] = v
        pending.clear()
    for line in f:
        if not line: continue
        ch = line[0]
        if ch == "#": flush(); continue
        if ch in "01xzXZ":
            code = line[1:].strip()
            if code in ids: pending[code] = ch.lower()
        elif ch == "b" or ch == "B":
            v, code = line[1:].split()
            if code in ids: pending[code] = v.lower()
    flush()
missing = [k for k in names if k not in want]
array_if = "out_write" in missing
if array_if:
    comp = dones[:]
    if not in_beats: in_beats = [1]
else:
    comp = [out_beats[(j + 1) * B - 1] for j in range(NT) if (j + 1) * B - 1 < len(out_beats)]
res = {"array_interface": array_if, "signals_found": want, "signals_missing": missing, "cycles_simulated": cycle,
       "in_beats": len(in_beats), "out_beats": len(out_beats), "beats_per_transform": B, "transforms_completed": len(comp),
       "first_input_cycle": in_beats[0] if in_beats else None,
       "first_output_cycles": (out_beats[0] - in_beats[0]) if in_beats and out_beats else None,
       "first_transform_completion_cycles": (comp[0] - in_beats[0]) if comp and in_beats else None,
       "batch_completion_cycles": (comp[-1] - in_beats[0]) if comp and in_beats else None,
       "input_stall_cycles_transform0": (in_beats[B - 1] - in_beats[0] + 1 - B) if len(in_beats) >= B else None,
       "ap_done_pulses": len(dones), "ap_ready_pulses": len(readys), "ap_start_high_cycles": start_cycles,
       "definitions": {"cycle": "rising ap_clk edges", "first_output_cycles": "first out beat - first in beat",
                       "completion": "last out beat of the transform (ap_done for array interfaces) - first in beat",
                       "steady_interval": "median of consecutive completion differences over the last 3/4 of the transforms"}}
if len(comp) > 2:
    d = [comp[j + 1] - comp[j] for j in range(len(comp) - 1)]; tail = d[len(d) // 4:]
    inp_first = [in_beats[j * B] for j in range(NT) if j * B < len(in_beats)] if not array_if else []
    res.update({"completion_intervals_all": d, "steady_interval_cycles_median": statistics.median(tail), "steady_interval_min": min(tail), "steady_interval_max": max(tail),
                "input_start_intervals": [inp_first[j + 1] - inp_first[j] for j in range(len(inp_first) - 1)][:12]})
if dump and not array_if:
    def unpack(v, out):
        bits = v.replace("x", "0").replace("z", "0").rjust(W, "0")
        for u in range(2 * UF):
            word = bits[W - 64 * (u + 1): W - 64 * u]
            out.write("%08x %08x\n" % (int(word[32:], 2), int(word[:32], 2)))
    with open(dump + "_inputs.txt", "w") as fi:
        for v in in_data[: NT * B]: unpack(v, fi)
    with open(dump + "_outputs.txt", "w") as fo:
        for v in out_data[: NT * B]: unpack(v, fo)
    res["dumped_beats"] = {"in": min(len(in_data), NT * B), "out": min(len(out_data), NT * B)}
print(json.dumps(res))
