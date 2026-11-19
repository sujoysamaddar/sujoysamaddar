"""Three-phase 6-pulse thyristor rectifier with firing-angle sweep.

Shows: 3-phase source, thyristors with gate pulses, source inductance
(commutation overlap), parameter sweep and comparison with theory
Vdc = 1.35 * VLL * cos(alpha) - 3*w*Ls*Id/pi.
"""
import math

from _common import out

import openpe as pe

VLL, F, LS = 400.0, 50.0, 0.2e-3


def build(alpha=30.0):
    c = pe.Circuit(f"6-pulse thyristor rectifier, alpha={alpha:g} deg")
    c.add(pe.ThreePhaseSource("G", "sa", "sb", "sc", "0", VLL, F))
    for ph in "abc":
        c.L(f"Ls{ph}", f"s{ph}", ph, LS)
    # thyristor k fires at alpha + 60*(k-1) degrees after the natural
    # commutation point (30 deg after the phase-a voltage zero crossing)
    legs = [("T1", "a", "p"), ("T3", "b", "p"), ("T5", "c", "p"),
            ("T4", "m", "a"), ("T6", "m", "b"), ("T2", "m", "c")]
    fire_order = {"T1": 0, "T2": 1, "T3": 2, "T4": 3, "T5": 4, "T6": 5}
    for name, an, ka in legs:
        k = fire_order[name]
        delay = ((30 + alpha + 60 * k) % 360) / 360 / F
        # 120-degree-wide gate pulse (keeps the device gated during overlap)
        c.add(pe.SquareWave(f"g_{name}", F, duty=1 / 3, delay=delay))
        c.SCR(name, an, ka, f"g_{name}")
    c.L("Ld", "p", "x", 20e-3)     # smoothing reactor
    c.R("Rd", "x", "m", 10)
    return c


def measure(res):
    w = res.last(2 / F)
    return {"Vdc": w.mean("V(p,m)"), "Id": w.mean("I(Rd)"), "Pdc": w.power("Rd")}


rows = pe.parameter_sweep(build, {"alpha": [0, 15, 30, 45, 60, 75]}, dt=5e-6, t_end=0.12, measure=measure)
for r in rows:
    r["Vdc_theory"] = 1.35 * VLL * math.cos(math.radians(r["alpha"])) - 3 * 2 * math.pi * F * LS * r["Id"] / math.pi
print(pe.format_table(rows))

res = build(30).simulate(5e-6, 0.12, t_record_start=0.08)
res.plot("V(p,m)", ["I(Lsa)", "I(Lsb)", "I(Lsc)"], "I(Rd)", save=out("03_rectifier_alpha30.png"))
print(f"Line current THD at alpha=30: {100 * res.thd('I(Lsa)', F, cycles=2):.1f} %")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot([r["alpha"] for r in rows], [r["Vdc"] for r in rows], "o-", label="simulated")
ax.plot([r["alpha"] for r in rows], [r["Vdc_theory"] for r in rows], "--", label="theory (with overlap)")
ax.set_xlabel("firing angle alpha (deg)")
ax.set_ylabel("Vdc (V)")
ax.grid(alpha=0.3)
ax.legend()
ax.set_title("Thyristor rectifier: firing-angle sweep")
fig.tight_layout()
fig.savefig(out("03_rectifier_sweep.png"), dpi=120)
