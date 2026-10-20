"""YAML-driven measurement sequencer.

A sequence file lists steps; each step type is a function registered below.
The engine runs steps in order, retries a step that raises an instrument error,
logs every step to a JSONL run log, and writes results even if the run aborts.

    seq = Sequencer.from_yaml("sequences/optimized.yaml", bench, out_dir)
    summary = seq.run()
"""
from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

import numpy as np
import yaml

from .analysis import enob, fit_resonance, sine_fit_3, sine_fit_4
from .drivers import Bench, InstrumentError

log = logging.getLogger("amseq.seq")
STEPS = {}


def step(name):
    def deco(fn):
        STEPS[name] = fn
        return fn
    return deco


class LimitFailure(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Step implementations. Each receives the run context and the step's params.
# --------------------------------------------------------------------------
@step("health_check")
def health_check(ctx, offset_tol_V=0.001, enob_min=6.0, T_range=(1.5, 300.0)):
    """Instrument maintenance check: self-test, error queues, scope offset
    (re-zeroed if out of tolerance), digitizer ENOB per IEEE 1057, sensor sanity."""
    b: Bench = ctx["bench"]
    rec = {"time": time.time()}
    for d in b.all():
        rec[f"{d.name}_selftest"] = d.self_test()
        rec[f"{d.name}_errors"] = d.errors()
    v, _ = b.scope.waveform(3)
    rec["offset_before_V"] = float(np.mean(v))
    if abs(rec["offset_before_V"]) > offset_tol_V:
        b.scope.zero_cal()
        v, _ = b.scope.waveform(3)
        rec["zero_cal_run"] = True
    else:
        rec["zero_cal_run"] = False
    rec["offset_after_V"] = float(np.mean(v))
    b.fg.configure(1.0e6, ctx["cfg"]["fg_vpp"])
    v, dt = b.scope.waveform(2)
    fit = sine_fit_4(v, dt)
    rec["enob_bits"] = enob(fit["rms_resid"], 8 * ctx["cfg"]["scales"][1])
    rec["T_K"] = b.tc.temperature()
    rec["passed"] = bool(
        all(rec[f"{d.name}_selftest"] == 0 and not rec[f"{d.name}_errors"] for d in b.all())
        and abs(rec["offset_after_V"]) <= offset_tol_V
        and rec["enob_bits"] >= enob_min
        and T_range[0] <= rec["T_K"] <= T_range[1])
    with open(ctx["out"] / "health_log.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    ctx["health"] = rec
    if not rec["passed"]:
        raise LimitFailure(f"health check failed: {rec}")


@step("configure")
def configure(ctx):
    c, b = ctx["cfg"], ctx["bench"]
    b.fg.configure(c["f_start"], c["fg_vpp"])
    b.scope.configure(c["srat"], c["points"], c["scales"])
    for d in b.all():
        d.check_errors()


@step("set_temperature")
def set_temperature(ctx, wait="stable", wait_s=10.0, tol_K=0.005, window_s=1.0,
                    timeout_s=60.0):
    b, T = ctx["bench"], ctx["T_set"]
    t0 = time.monotonic()
    b.tc.set_point(T)
    if wait == "stable":
        r = b.tc.wait_stable(T, tol=tol_K, window=window_s, timeout=timeout_s)
        T_meas = r["T"]
    else:
        time.sleep(wait_s)
        T_meas = float(np.mean([b.tc.temperature() for _ in range(5)]))
    ctx["row"].update(T_meas=T_meas, temp_wait_s=time.monotonic() - t0)


def _amplitude(b: Bench, ch, f, binary, cache):
    v, dt = b.scope.waveform(ch, binary=binary, cache=cache)
    return sine_fit_3(v, dt, f)["amplitude"]


def _measure(ctx, freqs, settle_ms, binary, cache, reference, sync):
    b = ctx["bench"]
    ref = None
    out = []
    for f in freqs:
        b.fg.set_freq(f, sync=sync)
        time.sleep(settle_ms / 1e3)
        a1 = _amplitude(b, 1, f, binary, cache)
        if reference == "every_point" or ref is None:
            ref = _amplitude(b, 2, f, binary, cache)
        out.append((f, a1 / ref))
    return out


@step("sweep")
def sweep(ctx, mode="adaptive", f_start=None, f_stop=None, points=201,
          coarse_points=25, fine_points=31, fine_halfwidth_steps=2.0, settle_ms=2.0,
          binary=True, cache=True, reference="once", sync=False):
    c = ctx["cfg"]
    f_start = f_start or c["f_start"]
    f_stop = f_stop or c["f_stop"]
    t0 = time.monotonic()
    if mode == "uniform":
        data = _measure(ctx, np.linspace(f_start, f_stop, points), settle_ms,
                        binary, cache, reference, sync)
    elif mode == "adaptive":
        coarse = np.linspace(f_start, f_stop, coarse_points)
        data = _measure(ctx, coarse, settle_ms, binary, cache, reference, sync)
        step_hz = coarse[1] - coarse[0]
        f_pk = max(data, key=lambda x: x[1])[0]
        fine = np.linspace(f_pk - fine_halfwidth_steps * step_hz,
                           f_pk + fine_halfwidth_steps * step_hz, fine_points)
        data += _measure(ctx, fine, settle_ms, binary, cache, reference, sync)
    else:
        raise ValueError(f"unknown sweep mode {mode}")
    data.sort()
    ctx["sweep"] = np.array(data)
    ctx["row"].update(sweep_s=time.monotonic() - t0, n_points=len(data))
    tag = f"T{ctx['T_set']:.2f}K"
    np.savetxt(ctx["out"] / f"sweep_{tag}.csv", ctx["sweep"], delimiter=",",
               header="freq_Hz,mag", comments="")


@step("fit_resonance")
def fit_step(ctx):
    s = ctx["sweep"]
    ctx["row"].update(fit_resonance(s[:, 0], s[:, 1]))


@step("check_limits")
def check_limits(ctx, Q_min=None, f0_se_max_Hz=None):
    r = ctx["row"]
    ok = True
    if Q_min is not None and r["Q"] < Q_min:
        ok = False
    if f0_se_max_Hz is not None and r["f0_se"] > f0_se_max_Hz:
        ok = False
    r["pass"] = ok


@step("for_each_temperature")
def for_each_temperature(ctx, steps, temperatures_K=None):
    for T in temperatures_K or ctx["cfg"]["temperatures_K"]:
        ctx["T_set"] = T
        ctx["row"] = {"T_set": T}
        ctx["engine"].run_steps(steps)
        ctx["rows"].append(ctx["row"])
        log.info("T=%.2f K done: %s", T,
                 {k: round(v, 4) if isinstance(v, float) else v for k, v in ctx["row"].items()})


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
class Sequencer:
    def __init__(self, seq: dict, bench: Bench, out_dir: Path):
        self.seq, self.bench = seq, bench
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        cfg = dict(seq["config"])
        for k in ("fg_vpp", "srat", "f_start", "f_stop"):
            cfg[k] = float(cfg[k])           # YAML 1.1 reads "1e6" as a string
        cfg["points"] = int(cfg["points"])
        self.ctx = {"bench": bench, "cfg": cfg, "out": self.out,
                    "rows": [], "engine": self}
        self.step_retries = seq.get("step_retries", 2)
        self.log_fh = open(self.out / "run_log.jsonl", "w")

    @classmethod
    def from_yaml(cls, path, bench, out_dir):
        return cls(yaml.safe_load(Path(path).read_text()), bench, out_dir)

    def _log(self, **kw):
        self.log_fh.write(json.dumps({"t": time.time(), **kw}, default=str) + "\n")
        self.log_fh.flush()

    def run_steps(self, steps):
        for spec in steps:
            spec = dict(spec)
            kind = spec.pop("type")
            fn = STEPS[kind]
            for attempt in range(self.step_retries + 1):
                t0 = time.monotonic()
                try:
                    fn(self.ctx, **spec)
                    self._log(step=kind, status="ok", attempt=attempt,
                              dt=round(time.monotonic() - t0, 4))
                    break
                except InstrumentError as e:
                    self._log(step=kind, status="error", attempt=attempt, error=str(e))
                    if attempt == self.step_retries:
                        raise
                    for d in self.bench.all():
                        d.inst.clear()

    def run(self) -> dict:
        t0 = time.monotonic()
        status = "ok"
        try:
            self.run_steps(self.seq["steps"])
        except (InstrumentError, LimitFailure) as e:
            status = f"aborted: {e}"
            self._log(step="sequence", status="aborted", error=str(e))
        finally:
            self.log_fh.close()
        rows = self.ctx["rows"]
        if rows:
            keys = sorted({k for r in rows for k in r})
            with open(self.out / "results.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
        st = self.bench.stats
        summary = {"sequence": self.seq["name"], "status": status,
                   "wall_s": time.monotonic() - t0, "rows": rows,
                   "health": self.ctx.get("health"),
                   "io": {"commands": st.commands, "queries": st.queries,
                          "retries": st.retries, "timeouts": st.timeouts,
                          "bytes_in": st.bytes_in, "io_time_s": st.io_time}}
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return summary
