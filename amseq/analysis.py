"""Signal analysis: IEEE Std 1057 sine fits, ENOB, and resonance fitting.

IEEE Std 1057 (digitizing waveform recorders) defines the three-parameter
sine fit (frequency known; solve for in-phase, quadrature and DC terms by
linear least squares) and the four-parameter fit (frequency also estimated,
iteratively). The four-parameter residual gives the noise used for effective
number of bits:

    ENOB = log2( FSR / (rms_residual * sqrt(12)) )

where FSR is the full-scale range. An ideal N-bit digitizer has residual
LSB/sqrt(12), so ENOB = N.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def sine_fit_3(v: np.ndarray, dt: float, f: float) -> dict:
    """IEEE 1057 three-parameter fit at known frequency f."""
    t = np.arange(len(v)) * dt
    D = np.column_stack([np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t),
                         np.ones_like(t)])
    (a, b, c), *_ = np.linalg.lstsq(D, v, rcond=None)
    resid = v - D @ np.array([a, b, c])
    return {"amplitude": float(np.hypot(a, b)), "phase": float(np.arctan2(-b, a)),
            "offset": float(c), "rms_resid": float(np.sqrt(np.mean(resid ** 2)))}


def sine_fit_4(v: np.ndarray, dt: float, f_guess: float | None = None,
               iters: int = 20) -> dict:
    """IEEE 1057 four-parameter fit (Gauss-Newton on frequency)."""
    n = len(v)
    t = np.arange(n) * dt
    if f_guess is None:                       # coarse estimate from the FFT peak
        spec = np.abs(np.fft.rfft(v - v.mean()))
        k = int(np.argmax(spec[1:]) + 1)
        f_guess = k / (n * dt)
    w = 2 * np.pi * f_guess
    p = sine_fit_3(v, dt, f_guess)
    a, b, c = p["amplitude"] * np.cos(p["phase"]), -p["amplitude"] * np.sin(p["phase"]), p["offset"]
    for _ in range(iters):
        cw, sw = np.cos(w * t), np.sin(w * t)
        D = np.column_stack([cw, sw, np.ones(n), -a * t * sw + b * t * cw])
        (a, b, c, dw), *_ = np.linalg.lstsq(D, v, rcond=None)
        w += dw
        if abs(dw) < 1e-12 * w:
            break
    model = a * np.cos(w * t) + b * np.sin(w * t) + c
    resid = v - model
    return {"amplitude": float(np.hypot(a, b)), "freq": float(w / (2 * np.pi)),
            "offset": float(c), "rms_resid": float(np.sqrt(np.mean(resid ** 2)))}


def enob(rms_resid: float, full_scale: float) -> float:
    return float(np.log2(full_scale / (rms_resid * np.sqrt(12))))


def lorentzian_mag(f, f0, Q, g):
    return g / np.sqrt(1 + (2 * Q * (f - f0) / f0) ** 2)


def fit_resonance(freqs: np.ndarray, mag: np.ndarray) -> dict:
    """Fit |H(f)| of a single-pole resonator; return f0, Q, peak gain and SEs."""
    i = int(np.argmax(mag))
    f0_0, g0 = freqs[i], mag[i]
    half = freqs[mag >= g0 / np.sqrt(2)]
    bw = max(half.max() - half.min(), (freqs[1] - freqs[0]) * 2) if len(half) > 1 else f0_0 / 1000
    p0 = [f0_0, f0_0 / bw, g0]
    p, cov = curve_fit(lorentzian_mag, freqs, mag, p0=p0, maxfev=20000)
    se = np.sqrt(np.diag(cov))
    return {"f0": float(p[0]), "Q": float(p[1]), "gain": float(p[2]),
            "f0_se": float(se[0]), "Q_se": float(se[1])}
