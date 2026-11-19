"""Single-phase boost PFC rectifier, average-current-mode control.

Shows: diode bridge + boost stage, multiplier-based current reference,
duty feed-forward, moving-average filter on the DC bus, power-factor and
line-current THD measurement.
"""
import math

from _common import out

import openpe as pe

VRMS, F, VO, FSW = 230.0, 50.0, 400.0, 25e3
VPK = VRMS * math.sqrt(2)

c = pe.Circuit("Boost PFC 230 V AC -> 400 V DC, 400 W")
c.V("Vac", "l", "n", pe.Sine(VPK, F))
c.R("Rgnd", "n", "0", 1e6)  # reference the floating mains
c.D("D1", "l", "rp")
c.D("D2", "n", "rp")
c.D("D3", "rn", "l")
c.D("D4", "rn", "n")
c.C("Cin", "rp", "rn", 1e-6)
c.L("Lb", "rp", "sw", 4e-3)
c.S("Q1", "sw", "rn", "g")
c.D("Db", "sw", "o")
c.C("Co", "o", "rn", 470e-6, ic=VPK)
c.R("RL", "o", "rn", VO ** 2 / 400)
c.add(
    pe.VoltageSensor("vin", "rp", "rn"),
    pe.VoltageSensor("vo_raw", "o", "rn"),
    pe.MovingAverage("vo", "vo_raw", 2 * F),         # removes 100 Hz ripple
    pe.Ramp("vref", slope=(VO - VPK) / 0.05, initial=VPK, final=VO),
    pe.Sum("verr", ["vref", "vo"], "+-"),
    pe.PI("ipk", "verr", kp=0.08, ki=1.5, lo=0.0, hi=8.0),
    pe.Function("iref", ["ipk", "vin"], lambda a, v: a * v / VPK),
    pe.CurrentSensor("il_raw", "Lb"),
    pe.SampleHold("il", "il_raw", FSW),             # synchronous = average current
    pe.Sum("ierr", ["iref", "il"], "+-"),
    pe.PI("dpi", "ierr", kp=0.15, ki=250.0, lo=-1.0, hi=1.0),
    pe.Function("dff", ["vin", "vo_raw"], lambda v, vo: 1 - v / max(vo, 1.0)),
    pe.Sum("dsum", ["dff", "dpi"]),
    pe.Limiter("d", "dsum", 0.0, 0.95),
    pe.PWM("g", "d", FSW),
)
res = c.simulate(dt=0.4e-6, t_end=0.3, record_every=5)
ss = res.last(2 / F)
print(f"DC bus       : {ss.mean('V(o,rn)'):.1f} V, 100 Hz ripple {ss.ripple('V(o,rn)'):.1f} V p-p")
print(f"Input power  : {-ss.power('Vac'):.0f} W")
print(f"Power factor : {ss.power_factor('V(l,n)', 'I(Vac)') * -1:.4f}")
print(f"Line THD     : {100 * ss.thd('I(Vac)', F, cycles=2):.2f} %")
print(res.meta["stats"])
res.plot("V(o,rn)", ["I(Lb)", "iref"], "ipk", save=out("08_pfc_startup.png"))
ss.plot(["V(l,n)"], ["I(Vac)"], save=out("08_pfc_line.png"), title="PFC: line voltage and current")
