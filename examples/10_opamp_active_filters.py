"""Op-amp active filters: Sallen-Key Butterworth low-pass and a
multiple-feedback (MFB) band-pass, compared with their analytic transfer
functions, plus the effect of a finite op-amp gain-bandwidth product.

Shows: OpAmp element (finite gain, GBW pole, rails), AC analysis,
ACResult.corners / peak, tf_response for analytic comparison.
"""
import math

import matplotlib
import numpy as np
from _common import out

import openpe as pe

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

F = np.logspace(1, 5, 500)


def sallen_key(fc=1e3, c2=10e-9, gbw=10e6):
    r = 1 / (2 * math.pi * fc * c2 * math.sqrt(2))
    c = pe.Circuit("Sallen-Key Butterworth LPF")
    c.V("Vin", "in", "0", 0, ac=1)
    c.R("R1", "in", "a", r)
    c.R("R2", "a", "b", r)
    c.C("C1", "a", "out", 2 * c2)
    c.C("C2", "b", "0", c2)
    c.opamp("U1", "b", "out", "out", gain=2e5, gbw=gbw, vmax=15, vmin=-15)
    return c, r


def mfb_bandpass(f0=1e3, q=5.0, gain=1.0, cap=10e-9, gbw=10e6):
    r2 = q / (math.pi * f0 * cap)
    r1 = r2 / (2 * gain)
    r3 = r2 / (4 * q * q - 2 * gain)
    c = pe.Circuit("MFB band-pass")
    c.V("Vin", "in", "0", 0, ac=1)
    c.R("R1", "in", "a", r1)
    c.R("R3", "a", "0", r3)
    c.C("C1", "a", "out", cap)
    c.C("C2", "a", "n", cap)
    c.R("R2", "out", "n", r2)
    c.opamp("U1", "0", "n", "out", gain=2e5, gbw=gbw)
    return c, (r1, r2, r3, cap)


# ---- Sallen-Key
sk, r = sallen_key()
ac = sk.ac(F)
wc = 2 * math.pi * 1e3
model = pe.tf_response([wc ** 2], [1, math.sqrt(2) * wc, wc ** 2], F)
print(f"Sallen-Key: R = {r:.0f} ohm, simulated fc = {ac.corners('V(out)')[1]:.1f} Hz (design 1000 Hz), "
      f"max deviation from Butterworth {np.max(np.abs(ac.mag_db('V(out)') - model['mag_db'])):.3f} dB")

# ---- MFB band-pass with ideal-ish and slow op-amps
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6))
a1.semilogx(F, ac.mag_db("V(out)"), label="Sallen-Key (simulated)")
a1.semilogx(F, model["mag_db"], "--", label="Butterworth 2nd order (analytic)")
a1.set_ylim(-60, 5)
a1.set_ylabel("dB")
a1.legend(fontsize=8)
a1.grid(True, which="both", alpha=0.3)
a1.set_title("Low-pass")
for gbw in (10e6, 1e6, 200e3):
    bp, (r1, r2, r3, cap) = mfb_bandpass(gbw=gbw)
    acb = bp.ac(F)
    fine = bp.ac(np.logspace(2.5, 3.5, 4001))     # dense grid for f0 / Q
    f_pk, g_pk = fine.peak("V(out)")
    lo, hi = fine.corners("V(out)")
    print(f"MFB band-pass, GBW = {gbw / 1e6:5.1f} MHz: f0 = {f_pk:7.1f} Hz, gain {g_pk:5.2f} dB, "
          f"Q = {f_pk / (hi - lo):.2f}")
    a2.semilogx(F, acb.mag_db("V(out)"), label=f"op-amp GBW {gbw / 1e6:g} MHz")
bp_num = [-1 / (r1 * cap), 0]
bp_den = [1, 2 / (r2 * cap), (r1 + r3) / (r1 * r2 * r3 * cap * cap)]
a2.semilogx(F, pe.tf_response(bp_num, bp_den, F)["mag_db"], "k--", label="analytic (ideal op-amp)")
a2.set_ylim(-40, 5)
a2.set_xlabel("frequency (Hz)")
a2.set_ylabel("dB")
a2.legend(fontsize=8)
a2.grid(True, which="both", alpha=0.3)
a2.set_title("Band-pass (f0 = 1 kHz, Q = 5): finite GBW shifts f0 and raises Q")
fig.tight_layout()
fig.savefig(out("10_active_filters.png"), dpi=120)

# ---- large-signal: low-pass driven into the rails
sk2, _ = sallen_key()
sk2.element("Vin").wave = pe.Square(-20, 20, 200)
res = sk2.simulate(dt=2e-6, t_end=10e-3)
print(f"Square-wave drive +/-20 V: output clips at {res.min('V(out)'):.2f} / {res.max('V(out)'):.2f} V (rails +/-15 V)")
res.plot(["V(in)", "V(out)"], save=out("10_sallen_key_clipping.png"), title="Sallen-Key step response with rail clipping")
