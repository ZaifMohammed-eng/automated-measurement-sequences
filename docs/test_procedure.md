# Test procedure: resonator characterization versus stage temperature

**Applies to:** `sequences/optimized.yaml` · **Bench:** function generator,
2-channel oscilloscope, cryostat temperature controller (simulated in this repo)

## 1. Purpose
Measure the centre frequency f0 and loaded quality factor Q of the device under
test (DUT) at a set of stage temperatures, with a pre-run instrument health check.

## 2. Setup
| Item | Setting |
|---|---|
| Generator | sine, 0.5 Vpp, output on |
| Scope CH1 | DUT output, 0.1 V/div |
| Scope CH2 | generator reference (tee before DUT), 0.1 V/div |
| Scope CH3 | 50 Ω terminated / grounded input, 0.01 V/div (offset check) |
| Acquisition | 20 MS/s, 2000 points (100 µs, about 100 cycles at 1 MHz) |
| Sweep | 980.0–1004.0 kHz; 25 coarse points, then 31 fine points over ±2 coarse steps around the peak |
| Temperatures | 3.0, 4.5, 6.0, 7.0 K |

## 3. Pre-run health check (every run)
| Check | Method | Limit | Action if out of limit |
|---|---|---|---|
| Self-test | `*TST?` on each instrument | returns 0 | stop; service instrument |
| Error queues | `SYST:ERR?` until empty | no errors | stop; inspect errors |
| Scope offset | mean of CH3 capture | \|offset\| ≤ 1.0 mV | run `CAL:ZERO`, re-measure; stop if still out |
| Digitizer ENOB | IEEE 1057 four-parameter sine fit, CH2, 1 MHz | ≥ 6.0 bits | stop; check scale and noise |
| Temperature sensor | `KRDG? A` | 1.5–300 K | stop; check sensor wiring |

Every result is appended to `health_log.jsonl` with a timestamp.

## 4. Measurement steps (per temperature)
1. `SETP 1,<T>`; wait until every reading over a 1.0 s window is within ±5 mK of
   the set point (timeout 30 s). Record the mean reading.
2. For each sweep frequency: set `FREQ`; wait 2 ms (>4 ring-down time constants,
   Q/πf0 ≤ 0.5 ms for Q ≤ 1500); capture CH1; take the amplitude by IEEE 1057
   three-parameter sine fit at the drive frequency. Capture CH2 once per sweep.
3. Compute |H| = A_CH1 / A_CH2; fit a single-pole Lorentzian magnitude for f0, Q, gain.

## 5. Acceptance limits
| Quantity | Limit |
|---|---|
| Q | ≥ 300 |
| Standard error of f0 | ≤ 20 Hz |

**Known limitation.** These limits check fit quality only. They passed a run
whose stage had not reached temperature (f0 off by up to 695 ppm). The
stability criterion in step 1 is the control for that failure; do not replace
it with a fixed wait.

## 6. Records
`results.csv` (one row per temperature), `sweep_T*.csv` (raw |H| vs f),
`run_log.jsonl` (each step, retries, errors), `summary.json` (results, health
check, I/O statistics), `health_log.jsonl` (maintenance history).
