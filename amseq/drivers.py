"""PyVISA drivers with timeouts, retries, error-queue checks and an I/O trace.

Each driver talks plain SCPI over a VISA resource. The resource strings point at
the simulators in this repo, but nothing here is simulator-specific: a real LAN
instrument with the same command set is addressed the same way.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pyvisa

log = logging.getLogger("amseq.io")


class InstrumentError(RuntimeError):
    pass


@dataclass
class IOStats:
    commands: int = 0
    queries: int = 0
    retries: int = 0
    timeouts: int = 0
    bytes_in: int = 0
    io_time: float = 0.0
    trace: list = field(default_factory=list)


class SCPIDevice:
    def __init__(self, rm: pyvisa.ResourceManager, resource: str, name: str,
                 timeout_ms: int = 500, retries: int = 3, stats: IOStats | None = None):
        self.name, self.resource = name, resource
        self.retries = retries
        self.stats = stats or IOStats()
        self.inst = rm.open_resource(resource, read_termination="\n",
                                     write_termination="\n", timeout=timeout_ms)

    # ---------------------------------------------------------------- low level
    def _trace(self, kind, cmd, t0, extra=""):
        dt = time.perf_counter() - t0
        self.stats.io_time += dt
        self.stats.trace.append((self.name, kind, cmd, round(dt * 1e3, 3), extra))
        log.debug("%-5s %-6s %-22s %7.2f ms %s", self.name, kind, cmd, dt * 1e3, extra)

    def write(self, cmd: str):
        t0 = time.perf_counter()
        self.inst.write(cmd)
        self.stats.commands += 1
        self._trace("write", cmd, t0)

    def _with_retry(self, fn, cmd):
        for attempt in range(self.retries + 1):
            try:
                return fn()
            except pyvisa.errors.VisaIOError as e:
                if e.error_code != pyvisa.constants.StatusCode.error_timeout:
                    raise
                self.stats.timeouts += 1
                if attempt == self.retries:
                    raise InstrumentError(f"{self.name}: no reply to {cmd!r} "
                                          f"after {self.retries + 1} tries") from e
                self.stats.retries += 1
                log.warning("%s: timeout on %r, retry %d", self.name, cmd, attempt + 1)
                self.inst.clear()          # flush any late reply before retrying

    def query(self, cmd: str) -> str:
        def go():
            t0 = time.perf_counter()
            r = self.inst.query(cmd)
            self.stats.queries += 1
            self.stats.bytes_in += len(r)
            self._trace("query", cmd, t0, r[:40])
            return r
        return self._with_retry(go, cmd)

    def query_float(self, cmd: str) -> float:
        return float(self.query(cmd))

    # ---------------------------------------------------------------- common
    def idn(self) -> str:
        return self.query("*IDN?")

    def reset(self):
        self.write("*RST")
        self.write("*CLS")
        self.opc()

    def opc(self):
        if self.query("*OPC?").strip() != "1":
            raise InstrumentError(f"{self.name}: *OPC? did not return 1")

    def self_test(self) -> int:
        return int(self.query("*TST?"))

    def errors(self) -> list[str]:
        out = []
        for _ in range(20):
            e = self.query("SYST:ERR?")
            if e.startswith("0,"):
                break
            out.append(e)
        return out

    def check_errors(self):
        errs = self.errors()
        if errs:
            raise InstrumentError(f"{self.name}: {errs}")

    def close(self):
        self.inst.close()


class FunctionGenerator(SCPIDevice):
    def configure(self, freq: float, vpp: float):
        self.write(f"FREQ {freq:.3f}")
        self.write(f"VOLT {vpp:.4f}")
        self.write("OUTP ON")

    def set_freq(self, f: float, sync: bool = False):
        self.write(f"FREQ {f:.3f}")
        if sync:
            self.opc()

    def freq(self) -> float:
        return self.query_float("FREQ?")

    def vpp(self) -> float:
        return self.query_float("VOLT?")


class Oscilloscope(SCPIDevice):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._state: dict = {}          # last source/format/preamble sent

    def configure(self, srat: float, points: int, scales=(0.1, 0.1, 0.01)):
        self._state.clear()
        self.write(f"ACQ:SRAT {srat:.6e}")
        self.write(f"ACQ:POIN {points}")
        for ch, s in enumerate(scales, start=1):
            self.write(f"CHAN{ch}:SCAL {s}")

    def waveform(self, ch: int, binary: bool = True,
                 cache: bool = True) -> tuple[np.ndarray, float]:
        """Volts and sample interval for one acquisition on channel `ch`.

        With cache=True, source, format and preamble are only re-sent when they
        change, which removes three round trips per acquisition.
        """
        form = "BYTE" if binary else "ASC"
        if not cache or self._state.get("src") != ch:
            self.write(f"WAV:SOUR CHAN{ch}")
            self._state.pop("pre", None)
        if not cache or self._state.get("form") != form:
            self.write(f"WAV:FORM {form}")
        if not cache or "pre" not in self._state:
            self._state["pre"] = tuple(float(x) for x in self.query("WAV:PRE?").split(","))
        self._state.update(src=ch, form=form)
        lsb, dt, _ = self._state["pre"]

        def go():
            t0 = time.perf_counter()
            if binary:
                codes = self.inst.query_binary_values("WAV:DATA?", datatype="b",
                                                      container=np.array)
                v = codes.astype(float) * lsb
                nbytes = len(codes)
            else:
                raw = self.inst.query("WAV:DATA?")
                v = np.array([float(x) for x in raw.split(",")])
                nbytes = len(raw)
            self.stats.queries += 1
            self.stats.bytes_in += nbytes
            self._trace("query", f"WAV:DATA? CH{ch}", t0, f"{nbytes} B")
            return v
        return self._with_retry(go, "WAV:DATA?"), dt

    def zero_cal(self):
        self.write("CAL:ZERO")
        self.opc()
        self._state.clear()


class TempController(SCPIDevice):
    def set_point(self, kelvin: float):
        self.write(f"SETP 1,{kelvin:.4f}")

    def temperature(self) -> float:
        return self.query_float("KRDG? A")

    def heater(self) -> float:
        return self.query_float("HTR?")

    def wait_stable(self, target: float, tol: float = 0.005, window: float = 1.0,
                    poll: float = 0.05, timeout: float = 60.0) -> dict:
        """Wait until every reading in the last `window` seconds is within `tol`."""
        t_start = time.monotonic()
        hist: list[tuple[float, float]] = []
        while True:
            now = time.monotonic()
            T = self.temperature()
            hist.append((now, T))
            hist = [(t, x) for t, x in hist if now - t <= window]
            if (now - t_start >= window and
                    all(abs(x - target) <= tol for _, x in hist)):
                return {"waited_s": now - t_start, "T": float(np.mean([x for _, x in hist]))}
            if now - t_start > timeout:
                raise InstrumentError(f"temperature not stable at {target} K "
                                      f"after {timeout} s (last {T:.4f} K)")
            time.sleep(poll)


@dataclass
class Bench:
    fg: FunctionGenerator
    scope: Oscilloscope
    tc: TempController
    stats: IOStats

    @classmethod
    def connect(cls, resources: dict[str, str], timeout_ms: int = 500,
                retries: int = 3) -> "Bench":
        rm = pyvisa.ResourceManager("@py")
        st = IOStats()
        kw = dict(timeout_ms=timeout_ms, retries=retries, stats=st)
        return cls(FunctionGenerator(rm, resources["fg"], "fg", **kw),
                   Oscilloscope(rm, resources["scope"], "scope", **kw),
                   TempController(rm, resources["tc"], "tc", **kw), st)

    def all(self):
        return [self.fg, self.scope, self.tc]

    def close(self):
        for d in self.all():
            d.close()
