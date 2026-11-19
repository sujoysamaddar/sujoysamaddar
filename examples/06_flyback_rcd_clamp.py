"""Flyback converter with a practical transformer (magnetising + leakage
inductance, winding resistance) and an RCD clamp.

Shows: Transformer element, the leakage-energy spike on the MOSFET drain
and how the clamp limits it, leakage/Coss ringing, switch-stress
measurement and loss/efficiency accounting.
"""
from _common import out

import openpe as pe

VIN, N, D, FSW = 48.0, 2.0, 0.4, 100e3


def build(clamp=True):
    c = pe.Circuit("Flyback 48 V -> 16 V, 100 kHz" + (" with RCD clamp" if clamp else ""))
    c.V("Vin", "in", "0", VIN)
    c.T("T1", "in", "drain", "0", "sa", n=N, lm=200e-6, llk=2e-6, rp=0.1, rs=0.02)
    c.S("Q1", "drain", "0", "g")
    c.C("Coss", "drain", "0", 200e-12)   # device output capacitance
    c.D("Do", "sa", "out")
    c.C("Co", "out", "0", 100e-6, ic=15)
    c.R("RL", "out", "0", 16)
    if clamp:
        c.D("Dc", "drain", "cl")
        c.C("Cc", "cl", "in", 100e-9, ic=40)
        c.R("Rc", "cl", "in", 4.7e3)
    c.add(pe.Constant("d", D), pe.PWM("g", "d", FSW, carrier="sawtooth"))
    return c


res = build(True).simulate(dt=20e-9, t_end=10e-3, t_record_start=9.9e-3)
ss = res.last(5 / FSW)
print(f"Vout         : {ss.mean('V(out)'):.2f} V (ideal {VIN * D / (1 - D) / N:.2f} V)")
print(f"Peak Vds     : {ss.max('V(drain)'):.1f} V  (Vin + n*Vo = {VIN + N * ss.mean('V(out)'):.1f} V)")
print(f"Clamp loss   : {ss.power('Rc'):.2f} W")
print(f"Efficiency   : {100 * ss.efficiency('RL', 'Vin'):.1f} %")
ss.plot("V(drain)", ["I(T1_Llk)", "I(T1_Lm)"], ["I(Do)"], "V(out)", save=out("06_flyback.png"))

noclamp = build(False).simulate(dt=20e-9, t_end=1e-3, t_record_start=0.95e-3)
print(f"Peak Vds without clamp: {noclamp.max('V(drain)'):.0f} V  <- why the clamp is needed")
