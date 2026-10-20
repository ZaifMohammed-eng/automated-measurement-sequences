"""SCPI instrument simulators served over TCP sockets.

Three instruments, each on its own port, speaking line-terminated SCPI the way
LAN instruments do (raw socket, port 5025 convention):

    FunctionGenerator   sine source: FREQ, VOLT, OUTP
    Oscilloscope        2-channel 8-bit digitizer + grounded CH3 for offset checks:
                        ACQ:SRAT, ACQ:POIN, CHANn:SCAL, WAV:SOUR, WAV:FORM, WAV:DATA?,
                        WAV:PRE?, CAL:ZERO
    TempController      cryostat stage controller (Lake Shore-style subset):
                        SETP 1,<K>, SETP? 1, KRDG? A, HTR?

The scope's CH1 sees the generator signal after a device under test (DUT); CH2
sees the generator directly. The DUT is a resonator whose centre frequency and
quality factor depend on the cryostat stage temperature, with a superconducting
(kinetic-inductance-like) temperature law:

    f0(T) = f00 * (1 - alpha/2 * (1/sqrt(1 - (T/Tc)^4) - 1))
    1/Q   = 1/Qc + 1/Qi(T),   Qi(T) = Qi0 / (1 + k (T/Tc)^4)
    H(f)  = g / (1 + 2jQ (f - f0)/f0)

Realism that the measurement code has to cope with:
  * DUT ring-down: after a frequency change the response settles with time
    constant Q/(pi f0); acquiring too early returns a blend of old and new.
  * Stage temperature follows the setpoint with a first-order lag plus noise.
  * 8-bit quantization, additive noise, and a scope offset that drifts with
    time until CAL:ZERO is run.
  * Per-command latency and finite link bandwidth, so transfer format matters.
  * Optional fault injection: a fraction of queries get no reply (timeouts).

Everything here is simulated. Swapping the VISA resource strings in the driver
layer for real instruments is the intended path to hardware, but this code has
not been run against physical instruments.
"""
from __future__ import annotations

import asyncio
import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------
# Shared physical state
# --------------------------------------------------------------------------
@dataclass
class DUT:
    f00: float = 1.0e6          # Hz, low-temperature centre frequency
    alpha: float = 0.10         # kinetic-inductance fraction
    Tc: float = 9.2             # K
    Qc: float = 2000.0
    Qi0: float = 5000.0
    k: float = 20.0
    gain: float = 0.9

    def f0(self, T: float) -> float:
        x = min((T / self.Tc) ** 4, 0.999)
        return self.f00 * (1 - 0.5 * self.alpha * (1 / math.sqrt(1 - x) - 1))

    def Q(self, T: float) -> float:
        qi = self.Qi0 / (1 + self.k * (T / self.Tc) ** 4)
        return 1 / (1 / self.Qc + 1 / qi)

    def H(self, f: float, T: float) -> complex:
        f0, Q = self.f0(T), self.Q(T)
        return self.gain / (1 + 2j * Q * (f - f0) / f0)


@dataclass
class Bench:
    """Physical state shared by the three simulated instruments."""
    dut: DUT = field(default_factory=DUT)
    # generator
    freq: float = 1.0e6
    volt: float = 0.5            # Vpp
    output: bool = False
    t_freq_change: float = 0.0
    prev_resp: complex = 0j
    # cryostat
    T0: float = 3.0
    T_set: float = 3.0
    t_set: float = 0.0
    tau_thermal: float = 1.0     # s
    T_noise: float = 0.002       # K rms on readings
    # scope
    srat: float = 20e6
    points: int = 2000
    scale: tuple = (0.1, 0.1, 0.01)   # V/div for CH1..CH3 (8 divisions full scale)
    noise: float = 0.002         # V rms
    offset0: float = 0.004       # V initial offset error
    offset_drift: float = 0.0005 # V per minute
    t_cal: float = 0.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(1))
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        now = time.monotonic()
        self.t_freq_change = self.t_set = self.t_cal = now

    # --- cryostat
    def stage_T(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        dt = now - self.t_set
        return self.T_set + (self.T0 - self.T_set) * math.exp(-dt / self.tau_thermal)

    def set_T(self, T: float):
        now = time.monotonic()
        self.T0 = self.stage_T(now)
        self.T_set = T
        self.t_set = now

    # --- generator
    def response(self, now: float) -> complex:
        """DUT complex response including ring-down after a frequency step."""
        T = self.stage_T(now)
        new = self.dut.H(self.freq, T)
        tau_r = self.dut.Q(T) / (math.pi * self.dut.f0(T))
        w = math.exp(-(now - self.t_freq_change) / tau_r)
        return new + (self.prev_resp - new) * w

    def set_freq(self, f: float):
        now = time.monotonic()
        self.prev_resp = self.response(now)
        self.freq = f
        self.t_freq_change = now

    # --- scope
    def offset(self, now: float) -> float:
        return self.offset0 + self.offset_drift * (now - self.t_cal) / 60.0

    def acquire(self, ch: int) -> tuple[np.ndarray, float, float]:
        """Return 8-bit codes, volts per code, seconds per sample."""
        now = time.monotonic()
        n, sr = self.points, self.srat
        t = np.arange(n) / sr
        amp = self.volt / 2 if self.output else 0.0
        phase0 = self.rng.uniform(0, 2 * np.pi)       # free-running trigger phase
        if ch == 1:
            r = self.response(now)
            v = amp * abs(r) * np.sin(2 * np.pi * self.freq * t + phase0 + np.angle(r))
        elif ch == 2:
            v = amp * np.sin(2 * np.pi * self.freq * t + phase0)
        else:                                          # CH3: grounded input
            v = np.zeros(n)
        v = v + self.offset(now) + self.rng.normal(0, self.noise, n)
        fs = 8 * self.scale[ch - 1]                    # full-scale range, V
        lsb = fs / 256
        codes = np.clip(np.round(v / lsb), -128, 127).astype(np.int8)
        return codes, lsb, 1 / sr


# --------------------------------------------------------------------------
# SCPI servers
# --------------------------------------------------------------------------
class Instrument:
    idn = "SIM,GENERIC,0,1.0"
    latency = 0.001                  # s per command
    bandwidth = 1.0e6                # bytes/s on the link

    def __init__(self, bench: Bench, drop_rate: float = 0.0, seed: int = 0):
        self.bench = bench
        self.errors: list[str] = []
        self.drop_rate = drop_rate
        self.rng = np.random.default_rng(seed)

    def common(self, cmd: str, arg: str):
        if cmd == "*IDN?":
            return self.idn
        if cmd in ("*RST", "*CLS"):
            self.errors.clear()
            if cmd == "*RST":
                self.reset()
            return None
        if cmd == "*OPC?":
            return "1"
        if cmd == "*TST?":
            return "0"
        if cmd in ("SYST:ERR?", ":SYST:ERR?"):
            return self.errors.pop(0) if self.errors else '0,"No error"'
        return NotImplemented

    def reset(self):
        pass

    def handle(self, cmd: str, arg: str):
        raise NotImplementedError

    def dispatch(self, line: str):
        parts = line.strip().split(None, 1)
        if not parts:
            return None
        cmd, arg = parts[0].upper(), (parts[1] if len(parts) > 1 else "")
        with self.bench.lock:
            r = self.common(cmd, arg)
            if r is NotImplemented:
                try:
                    r = self.handle(cmd, arg)
                except (ValueError, IndexError):
                    self.errors.append('-224,"Illegal parameter value"')
                    return None
                if r is NotImplemented:
                    self.errors.append(f'-113,"Undefined header;{cmd}"')
                    return None
        return r


class FunctionGenerator(Instrument):
    idn = "SIMLAB,FG-1000,SN0001,1.0"

    def reset(self):
        self.bench.set_freq(1.0e6)
        self.bench.volt, self.bench.output = 0.5, False

    def handle(self, cmd, arg):
        b = self.bench
        if cmd == "FREQ":
            f = float(arg)
            if not 1.0 <= f <= 50e6:
                raise ValueError
            b.set_freq(f)
            return None
        if cmd == "FREQ?":
            return f"{b.freq:.6f}"
        if cmd == "VOLT":
            v = float(arg)
            if not 0.001 <= v <= 10:
                raise ValueError
            b.volt = v
            return None
        if cmd == "VOLT?":
            return f"{b.volt:.6f}"
        if cmd == "OUTP":
            b.output = arg.strip().upper() in ("ON", "1")
            return None
        if cmd == "OUTP?":
            return "1" if b.output else "0"
        return NotImplemented


class Oscilloscope(Instrument):
    idn = "SIMLAB,DSO-2CH,SN0002,1.0"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.source = 1
        self.form = "ASC"
        self.last_pre = (1.0, 1.0)

    def handle(self, cmd, arg):
        b = self.bench
        if cmd == "ACQ:SRAT":
            b.srat = float(arg)
            return None
        if cmd == "ACQ:SRAT?":
            return f"{b.srat:.6e}"
        if cmd == "ACQ:POIN":
            n = int(arg)
            if not 100 <= n <= 100_000:
                raise ValueError
            b.points = n
            return None
        if cmd == "ACQ:POIN?":
            return str(b.points)
        if cmd.startswith("CHAN") and cmd.endswith(":SCAL"):
            ch = int(cmd[4])
            s = list(b.scale)
            s[ch - 1] = float(arg)
            b.scale = tuple(s)
            return None
        if cmd == "WAV:SOUR":
            self.source = int(arg.strip().upper().replace("CHAN", ""))
            if self.source not in (1, 2, 3):
                raise ValueError
            return None
        if cmd == "WAV:FORM":
            f = arg.strip().upper()
            if f not in ("ASC", "BYTE"):
                raise ValueError
            self.form = f
            return None
        if cmd == "WAV:PRE?":
            lsb = 8 * b.scale[self.source - 1] / 256
            return f"{lsb:.9e},{1 / b.srat:.9e},{b.points}"
        if cmd == "WAV:DATA?":
            # acquisition time + re-arm before data is available
            time.sleep(b.points / b.srat + 0.005)
            codes, lsb, _ = b.acquire(self.source)
            if self.form == "BYTE":
                payload = codes.tobytes()
                head = f"#{len(str(len(payload)))}{len(payload)}".encode()
                return head + payload
            return ",".join(f"{c * lsb:.6e}" for c in codes)
        if cmd == "CAL:ZERO":
            time.sleep(0.05)
            b.offset0, b.t_cal = float(self.rng.normal(0, 0.0003)), time.monotonic()
            return None
        return NotImplemented


class TempController(Instrument):
    idn = "SIMLAB,TC-336,SN0003,1.0"

    def handle(self, cmd, arg):
        b = self.bench
        if cmd == "SETP":
            loop, val = arg.split(",")
            T = float(val)
            if int(loop) != 1 or not 1.5 <= T <= 300:
                raise ValueError
            b.set_T(T)
            return None
        if cmd == "SETP?":
            return f"{b.T_set:.4f}"
        if cmd == "KRDG?":
            return f"{b.stage_T() + self.rng.normal(0, b.T_noise):.4f}"
        if cmd == "HTR?":
            err = b.T_set - b.stage_T()
            return f"{max(0.0, min(100.0, 50 + 40 * err)):.1f}"
        return NotImplemented


async def _serve(inst: Instrument, reader, writer):
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            text = line.decode(errors="replace")
            resp = await asyncio.to_thread(inst.dispatch, text)
            await asyncio.sleep(inst.latency)
            if resp is None:
                continue
            if "?" in text and inst.drop_rate and inst.rng.random() < inst.drop_rate:
                continue                                  # injected fault: no reply
            data = resp if isinstance(resp, bytes) else resp.encode()
            data += b"\n"
            await asyncio.sleep(len(data) / inst.bandwidth)
            writer.write(data)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        writer.close()


class SimulatedBench:
    """Run the three instrument servers on localhost in a background thread.

        with SimulatedBench() as sim:
            sim.resources  # VISA resource strings
    """

    def __init__(self, bench: Bench | None = None, drop_rate: float = 0.0,
                 host: str = "127.0.0.1"):
        self.bench = bench or Bench()
        self.host = host
        self.instruments = {
            "fg": FunctionGenerator(self.bench, drop_rate, seed=11),
            "scope": Oscilloscope(self.bench, drop_rate, seed=12),
            "tc": TempController(self.bench, drop_rate, seed=13),
        }
        self.ports: dict[str, int] = {}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)

    @property
    def resources(self) -> dict[str, str]:
        return {k: f"TCPIP0::{self.host}::{p}::SOCKET" for k, p in self.ports.items()}

    def truth(self, T: float) -> dict:
        d = self.bench.dut
        return {"f0": d.f0(T), "Q": d.Q(T)}

    def __enter__(self):
        self._thread.start()

        async def start():
            servers = []
            for name, inst in self.instruments.items():
                srv = await asyncio.start_server(
                    lambda r, w, i=inst: _serve(i, r, w), self.host, 0)
                self.ports[name] = srv.sockets[0].getsockname()[1]
                servers.append(srv)
            return servers

        self._servers = asyncio.run_coroutine_threadsafe(start(), self._loop).result()
        return self

    def __exit__(self, *exc):
        async def stop():
            for s in self._servers:
                s.close()
                if hasattr(s, "close_clients"):      # Python 3.13+
                    s.close_clients()
                try:
                    await asyncio.wait_for(s.wait_closed(), 1.0)
                except asyncio.TimeoutError:
                    pass
        asyncio.run_coroutine_threadsafe(stop(), self._loop).result()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2)
