#!/usr/bin/env python3
"""E4: the SPMW port against FEATHER's RTL, in three panels.

    plot_e4_comparison.py <results.csv> <out prefix>

Every value is read from `results.csv`; nothing is typed in here. The one
derived quantity is FEATHER's `fill_cycles`, which is its measured first output
less its `N^2` weight feed -- the panel draws the feed as a separate hatched
segment rather than hiding the subtraction, because that segment is the whole
reason the raw first-output figures are not a comparison.
"""

import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SIZES = (4, 8, 16)
RTL = "#1b4965"      # a deep slate for the hand-written baseline
SPMW = "#e08e45"     # a warm ochre for the port
FEED = "#9fb8c8"     # the feed, lighter: it is not part of the comparison


def load(path):
    rows = {r["run_id"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}
    out = {}
    for n in SIZES:
        r = rows[f"rtl_rowload_gemm128_N{n}_resident_general"]
        s = rows[f"spmw_gemm128_N{n}_resident_general_fabmul"]
        p = rows[f"pnr_rtl_rowload_N{n}"]
        q = rows[f"pnr_spmw_feather_fabmul_N{n}"]
        out[n] = {
            "rtl_fill": int(r["fill_cycles"]),
            "rtl_feed": int(r["weight_feed_cycles"]),
            "spmw_fill": int(s["fill_cycles"]),
            "rate_rtl": float(r["cycles_per_tile"]),
            "rate_spmw": float(s["cycles_per_tile"]),
            "rtl_lut": int(p["lut"]), "rtl_ff": int(p["ff"]),
            "spmw_lut": int(q["lut"]), "spmw_ff": int(q["ff"]),
        }
    return out


def main():
    src, prefix = sys.argv[1], sys.argv[2]
    d = load(src)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    })
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3))
    x = range(len(SIZES))
    w = 0.36

    # -- (a) latency ----------------------------------------------------------
    a = ax[0]
    rtl_fill = [d[n]["rtl_fill"] for n in SIZES]
    rtl_feed = [d[n]["rtl_feed"] for n in SIZES]
    sp_fill = [d[n]["spmw_fill"] for n in SIZES]
    a.bar([i - w / 2 for i in x], rtl_fill, w, color=RTL, label="FEATHER RTL: array fill")
    a.bar([i - w / 2 for i in x], rtl_feed, w, bottom=rtl_fill, color=FEED,
          hatch="//", edgecolor="white", linewidth=0.5,
          label="FEATHER RTL: $N^2$ weight feed")
    a.bar([i + w / 2 for i in x], sp_fill, w, color=SPMW, label="SPMW port: array fill")
    for i, n in enumerate(SIZES):
        a.text(i - w / 2, rtl_fill[i] + rtl_feed[i], f"{rtl_fill[i]+rtl_feed[i]}",
               ha="center", va="bottom", fontsize=7.5, color="#444")
        a.text(i + w / 2, sp_fill[i], f"{sp_fill[i]}\n({sp_fill[i]/rtl_fill[i]:.1f}x)",
               ha="center", va="bottom", fontsize=7.5, color="#444")
        a.text(i - w / 2, rtl_fill[i] / 2, f"{rtl_fill[i]}", ha="center", va="center",
               fontsize=7.5, color="white", fontweight="bold")
    a.set_title("(a)  latency to the first tile", fontsize=9.5, loc="left", pad=8)
    a.set_ylabel("cycles")
    a.set_ylim(0, max(rtl_fill[i] + rtl_feed[i] for i in range(3)) * 1.30)
    a.legend(fontsize=7, frameon=False, loc="upper left")
    a.annotate("compare the solid bars: the hatched segment is a\n"
               "weight feed the SPMW port does not perform",
               xy=(0.03, 0.46), xycoords="axes fraction", fontsize=7.5,
               color="#666", ha="left")

    # -- (b) throughput -------------------------------------------------------
    b = ax[1]
    b.plot(x, [d[n]["rate_rtl"] for n in SIZES], "o-", color=RTL, lw=2,
           ms=7, label="FEATHER RTL")
    b.plot(x, [d[n]["rate_spmw"] for n in SIZES], "s--", color=SPMW, lw=1.6,
           ms=5, label="SPMW port")
    b.plot(x, list(SIZES), ":", color="#999", lw=1, label="ideal ($N$)")
    for i, n in enumerate(SIZES):
        b.text(i, d[n]["rate_rtl"] + 0.7, f"{d[n]['rate_rtl']:.0f}", ha="center",
               fontsize=7.5, color="#444")
    b.set_title("(b)  steady throughput", fontsize=9.5, loc="left", pad=8)
    b.set_ylabel("cycles per tile   (lower is faster)")
    b.legend(fontsize=7, frameon=False, loc="upper left")
    b.annotate("the two curves coincide, and both sit on\nthe ideal: 100% array utilisation",
               xy=(0.97, 0.06), xycoords="axes fraction", fontsize=7.5,
               color="#666", ha="right")

    # -- (c) area -------------------------------------------------------------
    c = ax[2]
    c.bar([i - w / 2 for i in x], [d[n]["rtl_lut"] / 1000 for n in SIZES], w,
          color=RTL, label="FEATHER RTL")
    c.bar([i + w / 2 for i in x], [d[n]["spmw_lut"] / 1000 for n in SIZES], w,
          color=SPMW, label="SPMW port")
    for i, n in enumerate(SIZES):
        for off, key in ((-w / 2, "rtl_lut"), (w / 2, "spmw_lut")):
            c.text(i + off, d[n][key] / 1000, f"{d[n][key]/1000:.1f}", ha="center",
                   va="bottom", fontsize=7.5, color="#444")
    c.set_title("(c)  lookup tables, both DSP-free", fontsize=9.5, loc="left", pad=8)
    c.set_ylabel("CLB LUTs (thousands)")
    c.legend(fontsize=7, frameon=False, loc="upper left")
    c.set_ylim(0, max(d[n]["spmw_lut"] for n in SIZES) / 1000 * 1.32)
    c.annotate("the port's overhead is\nper element and the\n"
               "RTL's grows: they\ncross over by 16x16",
               xy=(0.04, 0.42), xycoords="axes fraction", fontsize=7.5,
               color="#666", ha="left", va="bottom")

    for axis in ax:
        axis.set_xticks(list(x))
        axis.set_xticklabels([f"{n}x{n}" for n in SIZES])
        axis.set_xlabel("array size")

    fig.suptitle(
        "FEATHER: the published RTL against the SPMW port  "
        "(int8 GEMM 128$^3$, weights resident, xcu280 at 3.333 ns)",
        fontsize=10, y=1.02, x=0.01, ha="left",
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{prefix}.{ext}", bbox_inches="tight", dpi=200)
    print(f"wrote {prefix}.pdf and {prefix}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
