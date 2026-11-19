"""Four-quadrant DC motor drive: H-bridge chopper, cascaded speed and
current PI loops, load-torque step and speed reversal (regeneration).

Shows: motor-drive module (electro-mechanical DC machine), load torque
driven from the control circuit, speed sensor (ProbeSensor), rate
limiter, bipolar PWM, current limiting via PI output limits.
"""
from _common import out

import openpe as pe

VDC = 200.0
c = pe.Circuit("4-quadrant DC drive: speed reversal and load step")
c.V("Vdc", "p", "0", VDC)
c.C("Cdc", "p", "0", 2e-3, ic=VDC)
c.S("Q1", "p", "a", "g1", body_diode=True)
c.S("Q2", "a", "0", "g2", body_diode=True)
c.S("Q3", "p", "b", "g2", body_diode=True)
c.S("Q4", "b", "0", "g1", body_diode=True)
c.add(pe.DCMotor("M1", "a", "b", ra=0.5, la=2e-3, ke=0.5, j=0.01, bf=1e-3, t_load="tload"))

c.add(
    # references
    pe.Function("wref_cmd", [], lambda t: 250.0 if t < 0.5 else -150.0, with_time=True),
    pe.RateLimiter("wref", "wref_cmd", rising=1200.0, ic=0.0),
    pe.StepSignal("tload", 0.3, 0.0, 4.0),
    # speed loop -> current reference (limited to +/-20 A)
    pe.ProbeSensor("w", "speed(M1)"),
    pe.Sum("werr", ["wref", "w"], "+-"),
    pe.PI("iref", "werr", kp=6.0, ki=150.0, lo=-20.0, hi=20.0),
    # current loop -> bipolar duty cycle
    pe.CurrentSensor("ia", "M1"),
    pe.Sum("ierr", ["iref", "ia"], "+-"),
    pe.PI("u", "ierr", kp=0.03, ki=20.0, lo=-0.48, hi=0.48),
    pe.Function("d", ["u"], lambda x: 0.5 + x),
    pe.PWM("g1", "d", freq=10e3, out_comp="g2"),
    # DC-link current (positive = drawn from the supply), filtered
    pe.ProbeSensor("idc_raw", "I(Vdc)", gain=-1.0),
    pe.LowPass("idc", "idc_raw", fc=200.0),
)
res = c.simulate(dt=2e-6, t_end=0.9, record_every=10)
print(res.meta["stats"])
for t0, label in ((0.29, "no load"), (0.49, "4 N.m load"), (0.89, "reversed")):
    w = res.window(t0 - 0.02, t0)
    print(f"{label:<12} speed {w.mean('speed(M1)'):8.2f} rad/s  ({w.mean('rpm(M1)'):7.1f} rpm),"
          f" Ia {w.mean('I(M1)'):6.2f} A, Te {w.mean('Te(M1)'):5.2f} N.m")
res.plot(["speed(M1)", "wref"], ["I(M1)", "iref"], ["Te(M1)", "tload"], "idc",
         save=out("04_dc_motor_drive.png"))
