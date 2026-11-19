"""Common-emitter BJT amplifier: the classic analog-electronics workflow.

Shows: nonlinear BJT (Ebers-Moll + Early effect, junction capacitances),
DC operating point (.op), AC small-signal analysis (.ac) with midband gain
and -3 dB corners, transient analysis started from the bias point, and
harmonic distortion versus input amplitude.
"""
import numpy as np
from _common import out

import openpe as pe


def build(vin_amp=10e-3):
    c = pe.Circuit("Common-emitter amplifier")
    c.V("Vcc", "vcc", "0", 12)
    c.V("Vin", "src", "0", pe.Sine(vin_amp, 1e3), ac=1, dc=0)
    c.R("Rs", "src", "in", 1e3)
    c.C("Cin", "in", "b", 10e-6)
    c.R("RB1", "vcc", "b", 47e3)
    c.R("RB2", "b", "0", 10e3)
    c.R("RC", "vcc", "c", 2.2e3)
    c.R("RE", "e", "0", 1e3)
    c.C("CE", "e", "0", 100e-6)
    c.C("Cout", "c", "out", 10e-6)
    c.R("RL", "out", "0", 10e3)
    c.Q("Q1", "c", "b", "e", "npn", is_=1e-14, bf=150, vaf=100, cje=20e-12, cjc=5e-12)
    return c


amp = build()
op = amp.op()
print(f"Bias: Vb={op['V(b)']:.3f} V  Ve={op['V(e)']:.3f} V  Vc={op['V(c)']:.3f} V")
print(f"      Ic={1e3 * op['I(Q1)']:.3f} mA  Vce={op['Vce(Q1)']:.2f} V  gm={1e3 * op['gm(Q1)']:.1f} mS  "
      f"P={1e3 * op['P(Q1)']:.1f} mW")

ac = amp.ac(np.logspace(0, 8, 400))
f_pk, g_pk = ac.peak("V(out)")
lo, hi = ac.corners("V(out)")
rin = 47e3 * 10e3 / 57e3 * (150 / op["gm(Q1)"]) / (47e3 * 10e3 / 57e3 + 150 / op["gm(Q1)"])
g_hand = op["gm(Q1)"] * (2.2e3 * 10e3 / 12.2e3) * rin / (rin + 1e3)
print(f"AC: midband gain {g_pk:.2f} dB (hand estimate {20 * np.log10(g_hand):.2f} dB), "
      f"bandwidth {lo:.1f} Hz .. {hi / 1e3:.0f} kHz")
ac.plot("V(out)", "V(b)", save=out("09_ce_amp_bode.png"), title="CE amplifier: AC analysis")

res = amp.simulate(dt=2e-6, t_end=6e-3, initial="op", t_record_start=4e-3)
print(f"Transient: output {res.ripple('V(out)'):.3f} V p-p for 20 mV p-p input, "
      f"THD {100 * res.thd('V(out)', 1e3, cycles=2):.2f} %")
res.plot(["V(src)"], ["V(out)"], ["I(Q1)"], save=out("09_ce_amp_transient.png"))

print("\nDistortion vs input amplitude:")
rows = pe.parameter_sweep(build, {"vin_amp": [2e-3, 5e-3, 10e-3, 20e-3, 40e-3]}, dt=2e-6, t_end=6e-3,
                          initial="op", t_record_start=4e-3,
                          measure=lambda r: {"Vout_pp": r.ripple("V(out)"),
                                             "THD_%": 100 * r.thd("V(out)", 1e3, cycles=2)})
print(pe.format_table(rows))
