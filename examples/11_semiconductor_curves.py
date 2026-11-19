"""Semiconductor characteristic curves by DC sweep (a 'curve tracer').

Shows: dc_sweep of independent sources, Shockley diode with Zener
breakdown, BJT output characteristics (with Early effect), MOSFET output
characteristics (channel-length modulation) and a CMOS inverter transfer
curve with its shoot-through current.
"""
import matplotlib
import numpy as np
from _common import out

import openpe as pe

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(2, 2, figsize=(11, 8))

# ---- diode / zener
c = pe.Circuit("diode")
c.V("V1", "a", "0", 0)
c.R("R1", "a", "d", 50)
c.Dpn("D1", "d", "0", is_=1e-14, n=1.0, bv=5.6, ibv=1e-3)
r = c.dc("V1", np.linspace(-8, 2, 401))
ax[0, 0].plot(r["V(d)"], 1e3 * r["I(D1)"])
ax[0, 0].set(xlabel="Vd (V)", ylabel="Id (mA)", title="Diode with 5.6 V Zener breakdown")
k = np.argmin(np.abs(r["I(D1)"] - 10e-3))
print(f"Diode: Vd = {r['V(d)'][k]:.3f} V at 10 mA;  Zener voltage at -10 mA: "
      f"{r['V(d)'][np.argmin(np.abs(r['I(D1)'] + 10e-3))]:.3f} V")

# ---- BJT output characteristics
for ib in (10e-6, 20e-6, 30e-6, 40e-6, 50e-6):
    c = pe.Circuit("bjt")
    c.I("IB", "0", "b", ib)
    c.V("VCE", "c", "0", 0)
    c.Q("Q1", "c", "b", "0", "npn", is_=1e-14, bf=150, vaf=60)
    r = c.dc("VCE", np.linspace(0, 10, 201))
    ax[0, 1].plot(r.t, 1e3 * r["I(Q1)"], label=f"Ib = {ib * 1e6:.0f} µA")
ax[0, 1].set(xlabel="Vce (V)", ylabel="Ic (mA)", title="NPN output characteristics (VA = 60 V)")
ax[0, 1].legend(fontsize=8)

# ---- NMOS output characteristics
for vgs in (1.5, 2.0, 2.5, 3.0, 3.5):
    c = pe.Circuit("nmos")
    c.V("VGS", "g", "0", vgs)
    c.V("VDS", "d", "0", 0)
    c.M("M1", "d", "g", "0", "nmos", vto=1.0, kp=2e-3, lam=0.02)
    r = c.dc("VDS", np.linspace(0, 5, 201))
    ax[1, 0].plot(r.t, 1e3 * r["I(M1)"], label=f"Vgs = {vgs} V")
ax[1, 0].plot(np.linspace(0, 2.5, 50), 1e3 * 0.5 * 2e-3 * np.linspace(0, 2.5, 50) ** 2, "k:", label="Vds = Vgs - Vt")
ax[1, 0].set(xlabel="Vds (V)", ylabel="Id (mA)", title="NMOS output characteristics (level 1)")
ax[1, 0].legend(fontsize=8)

# ---- CMOS inverter
c = pe.Circuit("cmos")
c.V("VDD", "vdd", "0", 5)
c.V("VIN", "in", "0", 0)
c.M("MP", "out", "in", "vdd", "pmos", vto=-1.0, kp=1e-3)
c.M("MN", "out", "in", "0", "nmos", vto=1.0, kp=1e-3)
vin = np.linspace(0, 5, 501)
r = c.dc("VIN", vin)
ax[1, 1].plot(vin, r["V(out)"], label="Vout")
ax2 = ax[1, 1].twinx()
ax2.plot(vin, -1e3 * r["I(VDD)"], "C1", label="supply current")
ax2.set_ylabel("Idd (mA)", color="C1")
ax[1, 1].set(xlabel="Vin (V)", ylabel="Vout (V)", title="CMOS inverter transfer curve")
vm = np.interp(2.5, r["V(out)"][::-1], vin[::-1])
gain = np.max(np.abs(np.gradient(r["V(out)"], vin)))
print(f"CMOS inverter: switching threshold {vm:.3f} V, peak shoot-through "
      f"{-1e3 * r.min('I(VDD)'):.3f} mA, max |gain| {gain:.0f}")
for a in ax.flat:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out("11_semiconductor_curves.png"), dpi=120)
