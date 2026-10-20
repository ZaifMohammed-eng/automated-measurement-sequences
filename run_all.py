"""Run every sequence against fresh simulated instruments and compare to truth.

    python run_all.py        # about 5 minutes

Runs
  baseline     fixed waits, *OPC? per step, ASCII, reference every point, 201-pt sweep
  optimized    stability wait, ring-down-sized settle, binary, cached setup, adaptive sweep
  short_wait   optimized sweep but a 2 s fixed temperature wait (too short)
  faults       optimized sequence with 3% of instrument replies dropped

Accuracy is measured against the simulator's ground truth at the set
temperature, which the measurement code never sees.
"""
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from amseq.drivers import Bench  # noqa: E402
from amseq.sequencer import Sequencer  # noqa: E402
from amseq.simulator import SimulatedBench  # noqa: E402

ROOT = Path(__file__).parent
RUNS = {
    "baseline": ("sequences/baseline.yaml", 0.0),
    "optimized": ("sequences/optimized.yaml", 0.0),
    "short_wait": ("sequences/short_wait.yaml", 0.0),
    "faults": ("sequences/optimized.yaml", 0.03),
}


def run_one(name, seq_file, drop_rate):
    out = ROOT / "results" / "runs" / name
    shutil.rmtree(out, ignore_errors=True)
    with SimulatedBench(drop_rate=drop_rate) as sim:
        bench = Bench.connect(sim.resources, timeout_ms=300, retries=3)
        s = Sequencer.from_yaml(ROOT / seq_file, bench, out).run()
        bench.close()
        for r in s["rows"]:
            tr = sim.truth(r["T_set"])
            r["f0_true"], r["Q_true"] = tr["f0"], tr["Q"]
            r["f0_err_ppm"] = (r["f0"] - tr["f0"]) / tr["f0"] * 1e6
            r["Q_err_pct"] = (r["Q"] - tr["Q"]) / tr["Q"] * 100
    (out / "summary.json").write_text(json.dumps(s, indent=2, default=str))
    return s


def main():
    logging.basicConfig(level=logging.WARNING, format="%(name)s %(message)s")
    res = {}
    for name, (seq, drop) in RUNS.items():
        s = run_one(name, seq, drop)
        rows = s["rows"]
        res[name] = {
            "status": s["status"],
            "wall_s": round(s["wall_s"], 2),
            "temp_wait_s": round(sum(r["temp_wait_s"] for r in rows), 2),
            "sweep_s": round(sum(r["sweep_s"] for r in rows), 2),
            "points_per_sweep": int(np.mean([r["n_points"] for r in rows])),
            "max_abs_f0_err_ppm": round(max(abs(r["f0_err_ppm"]) for r in rows), 2),
            "max_abs_Q_err_pct": round(max(abs(r["Q_err_pct"]) for r in rows), 2),
            "all_pass": all(r["pass"] for r in rows),
            "io": s["io"],
            "health": s["health"],
        }
        print(f"{name:11s} {res[name]['status']:4s} wall {res[name]['wall_s']:7.1f} s  "
              f"sweep {res[name]['sweep_s']:6.1f} s  f0 err <= {res[name]['max_abs_f0_err_ppm']:6.1f} ppm  "
              f"Q err <= {res[name]['max_abs_Q_err_pct']:5.1f}%  retries {s['io']['retries']}")
    b, o = res["baseline"], res["optimized"]
    res["comparison"] = {
        "total_speedup": round(b["wall_s"] / o["wall_s"], 2),
        "sweep_speedup": round(b["sweep_s"] / o["sweep_s"], 2),
        "temp_wait_speedup": round(b["temp_wait_s"] / o["temp_wait_s"], 2),
    }
    print(json.dumps(res["comparison"]))
    (ROOT / "results" / "benchmark.json").write_text(json.dumps(res, indent=2, default=str))

    import report
    report.main()


if __name__ == "__main__":
    main()
