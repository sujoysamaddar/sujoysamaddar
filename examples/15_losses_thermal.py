"""Loss, efficiency and junction-temperature analysis of a buck converter.

The MOSFET and the Schottky diode carry data-sheet style loss parameters:
conduction loss from Ron / Vf, switching loss from turn-on, turn-off and
reverse-recovery energies given at a reference voltage and current (they
scale with the actual voltage and current at each commutation).

1. loss report at 100 kHz (per device: conduction + switching) and efficiency
2. efficiency against switching frequency - the classic trade-off
3. MOSFET junction temperature from a two-stage Foster thermal network

Shows: Switch(eon, eoff, vref, iref), Diode(erec), Results.loss_report,
Results.junction_temperature.
"""
import matplotlib
from _common import out

import openpe as pe

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

VIN, DUTY, RLOAD = 48.0, 0.25, 3.0            # 12 V at 4 A


def buck(fsw):
    c = pe.Circuit(f"Buck {fsw / 1e3:.0f} kHz")
    c.V("Vin", "in", "0", VIN)
    c.S("S1", "in", "sw", "g", ron=15e-3, eon=25e-6, eoff=35e-6, vref=48, iref=10)
    c.D("D1", "0", "sw", vf=0.45, ron=8e-3, erec=2e-6, vref=48, iref=10)
    c.L("L1", "sw", "out", 47e-6, ic=3.8)
    c.C("C1", "out", "0", 100e-6, ic=11.4)
    c.R("Rload", "out", "0", RLOAD)
    c.add(pe.Constant("d", DUTY), pe.PWM("g", "d", fsw))
    return c


def run(fsw, t_end=2.5e-3, steps=200, **kw):
    """200 steps per switching period: the power integrals carry a sampling error that halves with dt."""
    return buck(fsw).simulate(dt=1 / (fsw * steps), t_end=t_end, **kw)


# ---- 1. loss report at 100 kHz ----------------------------------------------
res = run(100e3)
rep = res.last(0.4e-3).loss_report(output="Rload", input="Vin")
print(rep)

# ---- 2. efficiency vs switching frequency ---------------------------------------
fsws = [25e3, 50e3, 100e3, 200e3, 300e3]
rows = []
print(f"\n{'fsw':>8} {'conduction':>11} {'switching':>10} {'efficiency':>11}")
for f in fsws:
    r = run(f).last(0.4e-3).loss_report(output="Rload", input="Vin")
    rows.append((f, r.conduction_loss, r.switching_loss, r.efficiency))
    print(f"{f / 1e3:6.0f} k {r.conduction_loss:10.3f} W {r.switching_loss:8.3f} W {100 * r.efficiency:9.2f} %")

# ---- 3. junction temperature of the MOSFET ----------------------------------------
long = run(100e3, t_end=40e-3, steps=100, record_every=10)
stages = [(0.8, 0.002), (1.2, 0.05)]          # junction-case and case-sink: (K/W, J/K)
tj = long.junction_temperature("S1", stages=stages, ta=40.0)
p_s1 = long.last(5e-3).loss_report().rows[0].total
print(f"\nMOSFET loss {p_s1:.2f} W -> steady-state Tj = {40 + sum(r for r, _ in stages) * p_s1:.1f} C "
      f"(after 40 ms the slow stage has reached {tj[-1]:.1f} C)")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
names = [r.name for r in rep.rows]
ax[0].bar(names, [r.conduction for r in rep.rows], label="conduction")
ax[0].bar(names, [r.switching for r in rep.rows], bottom=[r.conduction for r in rep.rows], label="switching")
ax[0].set(ylabel="loss (W)", title=f"Loss breakdown at 100 kHz  (efficiency {100 * rep.efficiency:.1f} %)")
ax[0].legend()
ax[1].plot([f / 1e3 for f, *_ in rows], [100 * e for *_, e in rows], "o-")
ax[1].set(xlabel="switching frequency (kHz)", ylabel="efficiency (%)", title="Efficiency vs switching frequency")
ax[1].grid(alpha=0.3)
ax[2].plot(long.t * 1e3, tj)
ax[2].set(xlabel="time (ms)", ylabel="junction temperature (C)", title="MOSFET Tj, two-stage Foster network")
ax[2].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out("15_losses_thermal.png"), dpi=120)
