"""Complete linear power supply: mains transformer, bridge rectifier with
real (Shockley) diodes, reservoir capacitor and a Zener-referenced series-
pass BJT regulator.

Shows: mixing a power-electronics front end with nonlinear analog devices
in one transient simulation, ripple and ripple-rejection measurement,
diode peak currents, and line regulation by DC sweep.
"""
import math

import numpy as np
from _common import out

import openpe as pe


def build(vac_rms=230.0):
    c = pe.Circuit("Linear 9 V supply")
    c.V("Vac", "l", "n", pe.Sine(vac_rms * math.sqrt(2), 50))
    c.R("Rgnd", "n", "0", 1e6)
    c.T("T1", "l", "n", "s1", "s2", n=230 / 15, rs=0.5, lm=10.0)
    for name, a, k in (("D1", "s1", "p"), ("D2", "s2", "p"), ("D3", "0", "s1"), ("D4", "0", "s2")):
        c.Dpn(name, a, k, is_=1e-9, n=1.8, rs=0.02)
    c.C("Cres", "p", "0", 2200e-6)
    # regulator: Zener reference + emitter follower (series pass transistor)
    c.R("Rz", "p", "z", 330)
    c.Z("DZ", "0", "z", bv=9.7, ibv=5e-3, rs=5)
    c.Q("Q1", "p", "z", "out", "npn", is_=1e-13, bf=150, vaf=80)
    c.C("Cout", "out", "0", 100e-6)
    c.R("RL", "out", "0", 18)            # ~0.5 A load
    return c


# power-up from zero (a DC operating point of a mains transformer does not
# exist: its magnetising inductance shorts the source at DC)
res = build().simulate(dt=20e-6, t_end=0.3, t_record_start=0.2)
ss = res.last(0.04)
rip_in, rip_out = ss.ripple("V(p)"), ss.ripple("V(out)")
print(f"Raw DC  : {ss.mean('V(p)'):.2f} V, ripple {rip_in:.2f} V p-p")
print(f"Output  : {ss.mean('V(out)'):.3f} V, ripple {1e3 * rip_out:.1f} mV p-p "
      f"(ripple rejection {20 * math.log10(rip_in / rip_out):.1f} dB)")
print(f"Load    : {ss.mean('I(RL)'):.3f} A;  pass-transistor dissipation {ss.mean('P(Q1)'):.2f} W")
print(f"Diodes  : peak current {ss.max('I(D1)'):.2f} A (conduction angle short -> high peaks)")
print(f"Stats   : {res.meta['stats']}")
res.plot("V(p)", "V(out)", ["I(D1)", "I(D3)"], save=out("13_linear_supply.png"))

# line regulation: DC sweep of the (rectified) input as if the mains varied
reg = pe.Circuit("regulator only")
reg.V("Vraw", "p", "0", 14)
reg.R("Rz", "p", "z", 330)
reg.Z("DZ", "0", "z", bv=9.7, ibv=5e-3, rs=5)
reg.Q("Q1", "p", "z", "out", "npn", is_=1e-13, bf=150, vaf=80)
reg.R("RL", "out", "0", 18)
vin = np.linspace(10, 20, 101)
dc = reg.dc("Vraw", vin)
k1, k2 = np.searchsorted(vin, 13), np.searchsorted(vin, 18)
print(f"Line regulation (13 -> 18 V in): {1e3 * (dc['V(out)'][k2] - dc['V(out)'][k1]) / 5:.1f} mV/V")
dc.plot("V(out)", "P(Q1)", save=out("13_line_regulation.png"), title="Regulator: line regulation (DC sweep)")
