"""Figures from results/."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from amseq.analysis import lorentzian_mag

ROOT = Path(__file__).parent
RES, FIG = ROOT / "results", ROOT / "results" / "figures"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def rows(name):
    """Per-temperature results, including truth columns added by run_all.py."""
    return json.loads((RES / "runs" / name / "summary.json").read_text())["rows"]


def resonances():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for r in rows("optimized"):
        tag = f"T{r['T_set']:.2f}K"
        d = np.loadtxt(RES / "runs" / "optimized" / f"sweep_{tag}.csv", delimiter=",", skiprows=1)
        line, = ax.plot(d[:, 0] / 1e3, d[:, 1], "o", ms=3)
        f = np.linspace(d[0, 0], d[-1, 0], 800)
        ax.plot(f / 1e3, lorentzian_mag(f, r["f0"], r["Q"], r["gain"]), "-",
                color=line.get_color(), lw=1,
                label=f"{r['T_set']:.1f} K: f0 {r['f0'] / 1e3:.2f} kHz, Q {r['Q']:.0f}")
    ax.set_xlabel("drive frequency (kHz)")
    ax.set_ylabel("|H| = CH1 / CH2 amplitude")
    ax.set_title("Resonator vs stage temperature (optimized sequence, simulated bench)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "resonance_vs_temperature.png", dpi=150)


def timing():
    b = json.loads((RES / "benchmark.json").read_text())
    names = ["baseline", "optimized"]
    tw = [b[n]["temp_wait_s"] for n in names]
    sw = [b[n]["sweep_s"] for n in names]
    other = [b[n]["wall_s"] - b[n]["temp_wait_s"] - b[n]["sweep_s"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(names, tw, color="#c9d6e8", label="temperature wait")
    ax.barh(names, sw, left=tw, color="#1f4e8c", label="frequency sweeps")
    ax.barh(names, other, left=np.add(tw, sw), color="#999999", label="health check, setup")
    for i, n in enumerate(names):
        ax.text(b[n]["wall_s"] + 2, i, f"{b[n]['wall_s']:.0f} s", va="center")
    ax.set_xlabel("wall time for 4 temperatures (s)")
    ax.set_title(f"Throughput: {b['comparison']['total_speedup']:.1f}x faster, same pass/fail limits")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / "throughput.png", dpi=150)


def accuracy():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, mk in [("baseline", "s"), ("optimized", "o"), ("short_wait", "x")]:
        r = rows(name)
        T = [x["T_set"] for x in r]
        axes[0].plot(T, [x["f0_err_ppm"] for x in r], mk + "-", label=name)
        axes[1].plot(T, [x["Q_err_pct"] for x in r], mk + "-", label=name)
    axes[0].set_ylabel("f0 error vs truth (ppm)")
    axes[1].set_ylabel("Q error vs truth (%)")
    for a in axes:
        a.axhline(0, color="black", lw=0.6)
        a.set_xlabel("set temperature (K)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Accuracy against simulator ground truth: a too-short temperature wait biases every point")
    fig.tight_layout()
    fig.savefig(FIG / "accuracy.png", dpi=150)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    resonances()
    timing()
    accuracy()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
