"""Monte Carlo tolerance analysis of a Sallen-Key low-pass filter.

Shows: monte_carlo() with per-component tolerances (uniform or Gaussian),
statistics, production yield against a spec window and a histogram --
the same question a designer asks before ordering 1 % resistors vs 5 %
capacitors.
"""
import math

import numpy as np
from _common import out

import openpe as pe

FC, C2 = 1e3, 10e-9
R = 1 / (2 * math.pi * FC * C2 * math.sqrt(2))
F = np.logspace(2.5, 3.5, 300)


def filter_metrics(R1, R2, C1, C2):
    c = pe.Circuit("Sallen-Key")
    c.V("Vin", "in", "0", 0, ac=1)
    c.R("R1", "in", "a", R1)
    c.R("R2", "a", "b", R2)
    c.C("C1", "a", "out", C1)
    c.C("C2", "b", "0", C2)
    c.opamp("U1", "b", "out", "out", gain=2e5, gbw=10e6)
    ac = c.ac(F)
    return {"fc_Hz": ac.corners("V(out)")[1], "peak_dB": ac.peak("V(out)")[1]}


nominal = {"R1": R, "R2": R, "C1": 2 * C2, "C2": C2}
spec = (970.0, 1030.0)
for label, tol in (("1% R, 5% C ", {"R1": 0.01, "R2": 0.01, "C1": 0.05, "C2": 0.05}),
                   ("1% R, 1% C ", {"R1": 0.01, "R2": 0.01, "C1": 0.01, "C2": 0.01})):
    mc = pe.monte_carlo(filter_metrics, nominal, tol, n=300, seed=7, distribution="gauss")
    print(f"--- {label}(Gaussian, tolerance = 3 sigma)")
    print(mc.summary())
    print(f"Yield for fc in {spec[0]:.0f}..{spec[1]:.0f} Hz: {100 * mc.yield_('fc_Hz', *spec):.1f} %\n")
    tag = "5pct" if "5%" in label else "1pct"
    mc.histogram("fc_Hz", bins=30, limits=spec, save=out(f"12_monte_carlo_{tag}.png"),
                 title=f"Cut-off frequency spread, {label.strip()} ({len(mc.rows)} runs)")
