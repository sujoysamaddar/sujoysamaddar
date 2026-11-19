"""Space-vector PWM and a saturating inductor.

1. A three-phase inverter driven by SVPWM (min-max common-mode injection) reaches
   a phase voltage of Vdc/sqrt(3) = 0.577 Vdc.  Plain sine PWM stops at
   0.5 Vdc, so SVPWM gives 15 % more output from the same DC bus.
2. A buck converter whose inductor saturates when the load is stepped from
   6 ohm to 2 ohm at 2 ms (".change"-style value change with Circuit.at):
   the inductance collapses above Isat, the ripple current explodes and the
   peak current is far above the linear-inductor case.

Shows: SVPWM and PWM blocks, SaturableInductor, Circuit.at, harmonic / THD.
"""
import math

import matplotlib
import numpy as np
from _common import out

import openpe as pe

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---- 1. SVPWM three-phase inverter ---------------------------------------------------
VDC, F0, FSW, R, L = 400.0, 50.0, 10e3, 10.0, 5e-3
vref = 0.55 * VDC                                    # 220 V peak: beyond the 200 V limit of sine PWM
c = pe.Circuit("SVPWM inverter")
c.V("Vdc", "p", "0", VDC)
for k, ph in enumerate("abc"):
    c.S(f"Sh{ph}", "p", ph, f"gh{ph}", ron=10e-3, body_diode=True)
    c.S(f"Sl{ph}", ph, "0", f"gl{ph}", ron=10e-3, body_diode=True)
    c.R(f"R{ph}", ph, f"x{ph}", R)
    c.L(f"L{ph}", f"x{ph}", "n", L)
    c.add(pe.SineWave(f"v{ph}", vref, F0, -120.0 * k),
          pe.PWM(f"gh{ph}", f"d{ph}", FSW, out_comp=f"gl{ph}"))
c.add(pe.SVPWM("da", "db", "dc", "va", "vb", "vc", vdc=VDC))
res = c.simulate(dt=0.5e-6, t_end=80e-3, t_record_start=20e-3)
w = res.last(2 / F0)
v1, _ = w.harmonic("V(a,n)", F0, 1, cycles=2)
i1, _ = w.harmonic("I(La)", F0, 1, cycles=2)
print(f"SVPWM inverter, reference {vref:.0f} V peak (sine PWM limit {VDC / 2:.0f} V, SVPWM limit {VDC / math.sqrt(3):.0f} V)")
print(f"  load phase voltage fundamental {v1:.1f} V,  current {i1:.2f} A "
      f"(ideal {vref / math.hypot(R, 2 * math.pi * F0 * L):.2f} A),  current THD {100 * w.thd('I(La)', F0, cycles=2):.2f} %")
print(f"  DC-bus utilisation: {v1 / (VDC / 2):.3f} x Vdc/2   (sine PWM maximum 1.000, SVPWM maximum 1.155)")

# ---- 2. saturating inductor ----------------------------------------------------------------
def buck(sat):
    k = pe.Circuit("buck")
    k.V("Vin", "in", "0", 48)
    k.S("S1", "in", "sw", "g", ron=20e-3)
    k.D("D1", "0", "sw", vf=0.5)
    if sat:
        k.add(pe.SaturableInductor("L1", "sw", "out", l0=100e-6, isat=4.0, lsat=8e-6))
    else:
        k.L("L1", "sw", "out", 100e-6)
    k.C("C1", "out", "0", 100e-6)
    k.R("RL", "out", "0", 6.0)
    k.add(pe.Constant("d", 0.25), pe.PWM("g", "d", 50e3))
    k.at(2e-3, "RL", 2.0)                            # overload at 2 ms
    return k


lin = buck(False).simulate(0.1e-6, 3.5e-3)
sat = buck(True).simulate(0.1e-6, 3.5e-3)
for name, r in (("linear 100 uH", lin), ("saturating 100 uH / Isat 4 A", sat)):
    before, after = r.window(1.5e-3, 2e-3), r.window(3e-3, 3.5e-3)
    print(f"\n{name}\n  before overload: I ripple {before.ripple('I(L1)'):5.2f} A p-p, peak {before.max('I(L1)'):5.2f} A"
          f"\n  after overload : I ripple {after.ripple('I(L1)'):5.2f} A p-p, peak {after.max('I(L1)'):5.2f} A")
print(f"  inductance at the peak: {1e6 * sat.window(3e-3, 3.5e-3)['L(L1)'].min():.1f} uH")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].plot(w.t * 1e3, w["V(a,n)"], lw=0.5, label="V(a,n)")
tt = np.linspace(w.t[0], w.t[-1], 400)
ax[0].plot(tt * 1e3, vref * np.sin(2 * math.pi * F0 * tt), "k", lw=1.2, label="reference")
ax[0].set(xlabel="time (ms)", ylabel="V", title="SVPWM: phase voltage at 0.55 Vdc")
ax[0].legend(fontsize=8)
ax[1].plot(lin.t * 1e3, lin["I(L1)"], lw=0.6, label="linear")
ax[1].plot(sat.t * 1e3, sat["I(L1)"], lw=0.6, label="saturating")
ax[1].set(xlabel="time (ms)", ylabel="inductor current (A)", title="Overload at 2 ms")
ax[1].legend(fontsize=8)
ax[2].plot(sat.t * 1e3, 1e6 * sat["L(L1)"], lw=0.6)
ax[2].set(xlabel="time (ms)", ylabel="inductance (uH)", title="Inductance of the saturating inductor")
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out("17_svpwm_and_saturation.png"), dpi=120)
