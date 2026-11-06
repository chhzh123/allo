import re, sys
name, data = None, {}
for line in open("/scratch/hc676/agentstudy/steady_all.log"):
    line = line.strip()
    if line.startswith("====="):
        name = line.split()[1]
        data[name] = {"iv": [], "res": "-", "lat": "-", "wrong": "-"}
    elif name:
        m = re.search(r"STUDY INTERVAL \d+ (\d+)", line)
        if m: data[name]["iv"].append(int(m.group(1)))
        m = re.search(r"STUDY PRODUCT 0 .*latency (\d+)", line)
        if m: data[name]["lat"] = m.group(1)
        m = re.search(r"STUDY RESULT (\S+)", line)
        if m: data[name]["res"] = m.group(1)
        m = re.search(r"STUDY VALUES \d+ checked, (\d+) wrong", line)
        if m: data[name]["wrong"] = m.group(1)
        if "STUDY BUILD" in line: data[name]["res"] = line[:26]
print("%-34s %-14s %-6s %-9s %-9s %s" % ("design", "result", "lat", "iv first", "iv steady", "meets bars"))
for n in sorted(data):
    d = data[n]; iv = d["iv"]
    steady = max(iv[-4:]) if len(iv) >= 4 else (max(iv) if iv else None)
    ok = (d["res"] == "CORRECT" and d["lat"] != "-" and steady is not None
          and 22 <= int(d["lat"]) <= 64 and steady <= 16)
    print("%-34s %-14s %-6s %-9s %-9s %s" % (n, d["res"][:14], d["lat"],
          iv[0] if iv else "-", steady if steady is not None else "-",
          "yes" if ok else "no"))
