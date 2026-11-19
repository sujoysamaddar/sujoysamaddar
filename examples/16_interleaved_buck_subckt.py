"""Sub-circuits: a two-phase interleaved buck converter from ONE .subckt.

circuits/interleaved_buck.net defines a converter phase once (".subckt phase")
and instantiates it twice ("X1 ... phase ph=0", "X2 ... phase ph=180") with
carriers 180 degrees apart.  At D = 0.5 the two inductor ripple currents
cancel at the output node, so the output ripple collapses.  The same netlist
with one phase is compared, and the loss report covers the devices inside
both instances (X1.S1, X2.S1, ...).

Shows: .subckt / X instances with parameters, PWM phase shift, .losses.
"""
import os

import matplotlib
from _common import out

import openpe as pe

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

path = os.path.join(os.path.dirname(__file__), "..", "circuits", "interleaved_buck.net")
text = open(path, encoding="utf-8").read()
one_text = text.replace("X2 in out d phase ph=180\n", "").replace("Rload out 0 2.4", "Rload out 0 4.8")

two = pe.parse_netlist(text).run()
one = pe.parse_netlist(one_text).run()
w2, w1 = two.window(3e-3, 4e-3), one.window(3e-3, 4e-3)
isum = w2["I(X1.L1)"] + w2["I(X2.L1)"]
print(f"{'':<22}{'1 phase':>12}{'2 phases':>12}")
print(f"{'Vout mean (V)':<22}{w1.mean('V(out)'):12.3f}{w2.mean('V(out)'):12.3f}")
print(f"{'Vout ripple p-p (mV)':<22}{1e3 * w1.ripple('V(out)'):12.1f}{1e3 * w2.ripple('V(out)'):12.1f}")
print(f"{'phase-1 ripple (A p-p)':<22}{w1.ripple('I(X1.L1)'):12.2f}{w2.ripple('I(X1.L1)'):12.2f}")
print(f"{'summed ripple (A p-p)':<22}{w1.ripple('I(X1.L1)'):12.2f}{isum.max() - isum.min():12.2f}   <- cancels at D = 0.5")
print()
print(w2.loss_report(output="Rload", input="Vin"))

fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
a1.plot(w2.t * 1e3, w2["I(X1.L1)"], label="phase 1 inductor")
a1.plot(w2.t * 1e3, w2["I(X2.L1)"], label="phase 2 inductor")
a1.plot(w2.t * 1e3, isum, "k", lw=1.5, label="sum (output current)")
a1.set(ylabel="A", title="Two interleaved phases: ripple currents cancel")
a1.legend(fontsize=8)
a2.plot(w1.t * 1e3, w1["V(out)"], label="1 phase")
a2.plot(w2.t * 1e3, w2["V(out)"], label="2 phases")
a2.set(xlabel="time (ms)", ylabel="V", title="Output voltage ripple")
a2.legend(fontsize=8)
for a in (a1, a2):
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out("16_interleaved_buck.png"), dpi=120)
