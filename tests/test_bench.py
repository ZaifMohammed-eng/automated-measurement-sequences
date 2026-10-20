"""Tests for drivers, analysis and the sequencer, against the simulators.

    pytest -q        # ~1 minute
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amseq.analysis import enob, fit_resonance, lorentzian_mag, sine_fit_3, sine_fit_4  # noqa: E402
from amseq.drivers import Bench, InstrumentError  # noqa: E402
from amseq.sequencer import Sequencer  # noqa: E402
from amseq.simulator import SimulatedBench  # noqa: E402


# ---------------------------------------------------------------- analysis
def test_sine_fit_3_recovers_amplitude():
    dt, f = 1 / 20e6, 1.0e6
    t = np.arange(2000) * dt
    v = 0.2 * np.sin(2 * np.pi * f * t + 0.7) + 0.01
    p = sine_fit_3(v, dt, f)
    assert abs(p["amplitude"] - 0.2) < 1e-9 and abs(p["offset"] - 0.01) < 1e-9


def test_sine_fit_4_recovers_frequency_and_ideal_enob():
    dt = 1 / 20e6
    t = np.arange(4000) * dt
    lsb = 0.8 / 256
    v = np.round(0.39 * np.sin(2 * np.pi * 1.0123e6 * t) / lsb) * lsb   # ideal 8-bit
    p = sine_fit_4(v, dt)
    assert abs(p["freq"] - 1.0123e6) < 1.0
    assert 7.7 < enob(p["rms_resid"], 0.8) < 8.3


def test_fit_resonance_on_noiseless_lorentzian():
    f = np.linspace(0.98e6, 1.02e6, 81)
    p = fit_resonance(f, lorentzian_mag(f, 1.001e6, 800, 0.9))
    assert abs(p["f0"] - 1.001e6) < 0.01 and abs(p["Q"] - 800) < 1e-3


# ---------------------------------------------------------------- drivers
@pytest.fixture
def bench():
    with SimulatedBench() as sim:
        b = Bench.connect(sim.resources)
        yield sim, b
        b.close()


def test_identify_and_error_queue(bench):
    sim, b = bench
    assert b.scope.idn().startswith("SIMLAB,DSO")
    b.fg.write("FREQ -5")                          # illegal value
    errs = b.fg.errors()
    assert errs and errs[0].startswith("-224")
    assert b.fg.errors() == []                     # queue drained


def test_binary_and_ascii_transfers_agree(bench):
    sim, b = bench
    b.fg.configure(1.0e6, 0.5)
    b.scope.configure(20e6, 2000)
    a_bin = sine_fit_3(*b.scope.waveform(2, binary=True), 1.0e6)["amplitude"]
    a_asc = sine_fit_3(*b.scope.waveform(2, binary=False), 1.0e6)["amplitude"]
    assert abs(a_bin - 0.25) < 0.003 and abs(a_asc - 0.25) < 0.003


def test_retry_recovers_dropped_replies():
    with SimulatedBench(drop_rate=0.3) as sim:
        b = Bench.connect(sim.resources, timeout_ms=200, retries=6)
        for _ in range(20):
            assert b.tc.query("*IDN?").startswith("SIMLAB")
        assert b.stats.retries > 0
        b.close()


def test_retry_gives_up_with_clear_error():
    with SimulatedBench(drop_rate=1.0) as sim:
        b = Bench.connect(sim.resources, timeout_ms=100, retries=1)
        with pytest.raises(InstrumentError, match="no reply"):
            b.tc.temperature()
        b.close()


# ---------------------------------------------------------------- sequencer
def test_health_check_rezeroes_drifted_offset(tmp_path, bench):
    sim, b = bench
    sim.bench.offset0 = 0.005                      # 5 mV, beyond the 1 mV limit
    seq = {"name": "hc", "config": {"fg_vpp": 0.5, "srat": 2e7, "points": 2000,
                                    "scales": [0.1, 0.1, 0.01], "f_start": 9.8e5,
                                    "f_stop": 1.004e6, "temperatures_K": [3.0]},
           "steps": [{"type": "health_check"}]}
    s = Sequencer(seq, b, tmp_path).run()
    assert s["status"] == "ok"
    assert s["health"]["zero_cal_run"] and abs(s["health"]["offset_after_V"]) < 0.001
    assert (tmp_path / "health_log.jsonl").exists()


def test_optimized_sequence_matches_truth(tmp_path, bench):
    sim, b = bench
    root = Path(__file__).resolve().parents[1]
    seq = Sequencer.from_yaml(root / "sequences" / "optimized.yaml", b, tmp_path)
    seq.ctx["cfg"]["temperatures_K"] = [3.0, 6.0]
    s = seq.run()
    assert s["status"] == "ok"
    for r in s["rows"]:
        tr = sim.truth(r["T_set"])
        assert abs(r["f0"] - tr["f0"]) / tr["f0"] < 20e-6
        assert abs(r["Q"] - tr["Q"]) / tr["Q"] < 0.02
        assert r["pass"]
