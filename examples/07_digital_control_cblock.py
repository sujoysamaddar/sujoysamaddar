"""Digitally controlled boost converter.

The whole controller is user code in a PythonBlock (the equivalent of a
PSIM "C block"): it runs once per sampling period, uses ADC values
captured by sample-and-hold blocks, applies a one-sample computation
delay, and implements cascaded discrete PI loops exactly as they would be
written for a microcontroller.
"""
from _common import out

import openpe as pe

FSW = 50e3
TS = 1 / FSW


def controller(t, dt, u, s):
    """C-block style code: u = [vo_adc, il_adc]; returns duty."""
    vo, il = u
    if t + 1e-12 < s["next"]:
        return s["duty"]                    # hold between interrupts
    s["next"] += TS
    # --- "ISR" -------------------------------------------------------
    vref = min(24.0, 12.0 + 12.0 * t / 3e-3)       # soft start
    ev = vref - vo
    s["iv"] = min(15.0, max(0.0, s["iv"] + 500.0 * TS * ev))
    iref = min(15.0, max(0.0, 0.55 * ev + s["iv"]))
    ei = iref - il
    s["ii"] = min(0.9, max(0.0, s["ii"] + 150.0 * TS * ei))
    d_new = min(0.9, max(0.0, 0.08 * ei + s["ii"]))
    s["duty"], s["pending"] = s["pending"], d_new  # one-sample computation delay
    return s["duty"]


c = pe.Circuit("Digitally controlled boost (12 V -> 24 V), PythonBlock controller")
c.V("Vin", "in", "0", 12)
c.L("L1", "in", "sw", 100e-6)
c.S("Q1", "sw", "0", "g")
c.D("D1", "sw", "out")
c.C("Co", "out", "0", 220e-6, ic=12)
c.R("RL", "out", "0", 24)
c.R("Rstep", "out", "x", 24)
c.S("Sstep", "x", "0", "step")
c.add(
    pe.VoltageSensor("vo", "out"),
    pe.CurrentSensor("il", "L1"),
    pe.SampleHold("vo_adc", "vo", FSW),
    pe.SampleHold("il_adc", "il", FSW),
    pe.PythonBlock("duty", ["vo_adc", "il_adc"], controller,
                   state={"next": 0.0, "iv": 0.0, "ii": 0.0, "duty": 0.0, "pending": 0.0}),
    pe.PWM("g", "duty", FSW),
    pe.StepSignal("step", 8e-3),
)
res = c.simulate(dt=0.25e-6, t_end=14e-3)
for t0, label in ((8e-3, "24 W"), (14e-3, "48 W")):
    w = res.window(t0 - 1e-3, t0)
    print(f"{label}: Vout {w.mean('V(out)'):.3f} V, ripple {1e3 * w.ripple('V(out)'):.0f} mV, "
          f"IL {w.mean('I(L1)'):.2f} A, duty {w.mean('duty'):.3f}")
res.plot(["V(out)", "vo_adc"], ["I(L1)", "il_adc"], "duty", save=out("07_digital_boost.png"))
