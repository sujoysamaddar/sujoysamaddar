"""Closed-loop buck converter: average current-mode control, load step.

Shows: ideal switch + diode, PWM modulator, sensors, PI with anti-windup,
signal-controlled load change, waveform measurements.
"""
from _common import out

import openpe as pe

c = pe.Circuit("Buck converter, current-mode control (48 V -> 12 V)")
# ---- power circuit
c.V("Vin", "in", "0", 48)
c.S("Q1", "in", "sw", "gate")
c.D("D1", "0", "sw")
c.L("L1", "sw", "out", 100e-6)
c.C("C1", "out", "0", 220e-6)
c.R("Rload", "out", "0", 6.0)
c.R("Rstep", "out", "x", 6.0)            # second load switched in at 6 ms
c.S("Sload", "x", "0", "load_on")
# ---- control circuit
c.add(
    pe.VoltageSensor("vo", "out"),
    pe.CurrentSensor("il_raw", "L1"),
    # synchronous sampling at the carrier valley = average inductor current
    pe.SampleHold("il", "il_raw", fs=50e3),
    pe.Ramp("vref", slope=12 / 2e-3, final=12.0),   # soft start
    # outer voltage loop (~500 Hz crossover) -> inductor current reference
    pe.Sum("verr", ["vref", "vo"], "+-"),
    pe.PI("iref", "verr", kp=0.7, ki=700, lo=0.0, hi=10.0),
    # inner current loop (~5 kHz crossover) -> duty cycle
    pe.Sum("ierr", ["iref", "il"], "+-"),
    pe.PI("duty", "ierr", kp=0.065, ki=200, lo=0.0, hi=0.9),
    pe.PWM("gate", "duty", freq=50e3),
    pe.StepSignal("load_on", 6e-3),
)

res = c.simulate(dt=0.2e-6, t_end=14e-3)
st = res.meta["stats"]
print(f"{st['steps']} steps, {st['factorizations']} matrix factorisations, {st['cpu_time']:.2f}s")

before = res.window(5e-3, 6e-3)
after = res.window(13e-3, 14e-3)
print(f"Vout before step: {before.mean('V(out)'):.3f} V, ripple {1e3 * before.ripple('V(out)'):.1f} mV")
print(f"Vout after step : {after.mean('V(out)'):.3f} V, IL avg {after.mean('I(L1)'):.2f} A")
print(f"Efficiency (ideal switches): {after.efficiency(['Rload', 'Rstep'], 'Vin'):.4f}")
res.plot(["V(out)", "vref"], ["I(L1)", "iref"], "duty", save=out("01_buck_closed_loop.png"))
res.plot(["V(sw)"], ["I(L1)", "I(D1)"], xlim=(13.9e-3, 13.96e-3), save=out("01_buck_zoom.png"),
         title="Buck: switching detail")
