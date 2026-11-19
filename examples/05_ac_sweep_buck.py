"""AC sweep of a *switching* buck converter (control-to-output transfer
function) by small-signal injection, compared with the averaged model

    Gvd(s) = Vin / (L*C*s^2 + (L/R)*s + 1)

Shows: ac_sweep() - frequency response extracted from the time-domain
switching simulation, no averaged model needed - plus Bode plotting.
"""
import numpy as np
from _common import out

import openpe as pe

VIN, L, C, R, D0 = 24.0, 100e-6, 100e-6, 3.0, 0.5


def build():
    c = pe.Circuit("buck")
    c.V("Vin", "in", "0", VIN)
    c.S("Q1", "in", "sw", "g")
    c.D("D1", "0", "sw")
    c.L("L1", "sw", "out", L)
    c.C("C1", "out", "0", C)
    c.R("RL", "out", "0", R)
    c.add(
        pe.Constant("ac_pert", 0.0),          # replaced by the sweep sine
        pe.Constant("D0", D0),
        pe.Sum("d", ["D0", "ac_pert"]),
        pe.PWM("g", "d", freq=50e3),
        pe.VoltageSensor("vo", "out"),
    )
    return c


freqs = np.logspace(np.log10(200), np.log10(10e3), 9)
resp = pe.ac_sweep(build, freqs, measure="vo", amplitude=0.03, dt=0.1e-6, t_settle=6e-3, cycles=3)
model = pe.tf_response([VIN], [L * C, L / R, 1.0], np.logspace(2, 4.2, 200))
model_pts = pe.tf_response([VIN], [L * C, L / R, 1.0], freqs)
print(f"{'f (Hz)':>9} {'sim dB':>8} {'model dB':>9} {'sim deg':>8} {'model deg':>10}")
for k, f in enumerate(freqs):
    print(f"{f:9.0f} {resp['mag_db'][k]:8.2f} {model_pts['mag_db'][k]:9.2f} "
          f"{resp['phase_deg'][k]:8.1f} {model_pts['phase_deg'][k]:10.1f}")
pe.plot_bode(resp, title="Buck control-to-output: injection vs averaged model",
             analytic=model, save=out("05_ac_sweep_buck.png"))
