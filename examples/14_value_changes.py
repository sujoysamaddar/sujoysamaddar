"""Changing values: before and during a simulation.

1. Design time - size an RC low-pass for 1 kHz with ideal values, then replace
   them by standard parts (E24 / E96) and see the effect with an AC analysis.
2. Run time - a buck converter whose load resistor and voltage reference change
   while the simulation runs (".change" lines in the netlist, plus one more added
   from Python with Circuit.at), without any extra switches or sources.
3. Overrides - the same netlist with a different load, set from code the way the
   command line's  --set Rload.value=4  does.

Shows: openpe.eseries, Circuit.at / set_value, designparams.apply_overrides.
"""
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from _common import out

import openpe as pe
from openpe import eseries
from openpe.designparams import apply_overrides

# ---- 1. standard values ------------------------------------------------------
FC, C = 1e3, 100e-9
r_ideal = 1 / (2 * math.pi * FC * C)
choices = [("ideal", r_ideal)] + [(s, eseries.nearest(r_ideal, s)) for s in ("E12", "E24", "E96")]
freqs = np.logspace(0, 4, 800)               # from 1 Hz, where the gain is 0 dB
print(f"RC low-pass for {FC:.0f} Hz with C = 100 nF:  R = {r_ideal:.1f} ohm")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
for name, r in choices:
    c = pe.Circuit()
    c.V("V1", "in", "0", 0, ac=1)
    c.R("R1", "in", "out", r)
    c.C("C1", "out", "0", C)
    ac = c.ac(freqs)
    fc = ac.bandwidth("V(out)")
    print(f"  {name:>5}: R = {r:8.1f} ohm   fc = {fc:7.1f} Hz   ({100 * (fc - FC) / FC:+.2f} %)")
    ax1.semilogx(freqs, ac.mag_db("V(out)"), label=f"{name}: {r:.0f} ohm, fc {fc:.0f} Hz")
ax1.axvline(FC, color="gray", ls=":")
ax1.set(xlabel="frequency (Hz)", ylabel="gain (dB)", title="Ideal vs standard resistor values", ylim=(-12, 1), xlim=(100, 1e4))
ax1.legend(fontsize=8)
ax1.grid(True, which="both", alpha=0.3)

# ---- 2. value changes during the run ----------------------------------------------
net = pe.load_netlist(os.path.join(os.path.dirname(__file__), "..", "circuits", "load_step.net"))
net.circuit.at(8.5e-3, "Rload", 6.0)             # back to the light load, scheduled from Python
res = net.run(t_end=14e-3)
print("\nBuck converter with value changes (4 ms: load 6->3 ohm, 7 ms: reference 12->9 V, 8.5 ms: load 3->6 ohm)")
for label, a, b in (("start-up done", 3.5e-3, 4e-3), ("load doubled", 6.5e-3, 7e-3), ("reference lowered", 8e-3, 8.5e-3),
                    ("light load again", 13e-3, 14e-3)):
    w = res.window(a, b)
    print(f"  {label:<18} Vout {w.mean('V(out)'):6.3f} V   IL {w.mean('I(L1)'):5.2f} A")
print(f"  {res.meta['stats']['value_changes']} value changes applied")
w = res.window(0, 14e-3)
ax2.plot(w.t * 1e3, w["V(out)"], lw=0.8, label="V(out)")
ax2.plot(w.t * 1e3, w["vref"], "--", lw=1, label="reference")
ax2b = ax2.twinx()
ax2b.plot(w.t * 1e3, w["I(L1)"], color="C2", lw=0.6, alpha=0.6)
ax2b.set_ylabel("I(L1) (A)", color="C2")
for chg in res.meta["changes"]:
    ax2.axvline(chg["t"] * 1e3, color="gray", ls="--", lw=0.9)
    ax2.text(chg["t"] * 1e3, 1, f"{chg['target']}→{chg['value']:g}", rotation=90, va="bottom", ha="right", fontsize=7, color="gray")
ax2.set(xlabel="time (ms)", ylabel="V", title="Load and reference changes during the simulation", ylim=(0, 14))
ax2.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(out("14_value_changes.png"), dpi=120)
res.plot(["V(out)", "vref"], "I(L1)", save=out("14_value_changes_waveforms.png"), title="Buck with value changes")

# ---- 3. override a value without touching the file ------------------------------
text = open(os.path.join(os.path.dirname(__file__), "..", "circuits", "load_step.net"), encoding="utf-8").read()
text = apply_overrides(text, ["Rload.value=12", ".tran.stop time=4m"])      # same as: --set Rload.value=12 ...
res2 = pe.parse_netlist(text).run()
print(f"\nWith Rload overridden to 12 ohm: Vout = {res2.window(3.5e-3, 4e-3).mean('V(out)'):.3f} V, "
      f"IL = {res2.window(3.5e-3, 4e-3).mean('I(L1)'):.2f} A")
