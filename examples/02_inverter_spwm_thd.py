"""Single-phase full-bridge inverter with unipolar SPWM, dead time and an
LC output filter.  Shows: MOSFETs with body diodes, DeadTime block, sine
reference + triangle carriers, FFT / THD of output voltage and of the
unfiltered bridge voltage.
"""
from _common import out

import openpe as pe

f0, fsw, m = 50.0, 10e3, 0.85
c = pe.Circuit("Full-bridge inverter, unipolar SPWM (400 V DC -> 230 V AC)")
# ---- power stage
c.V("Vdc", "p", "0", 400)
c.S("Q1", "p", "a", "g1", body_diode=True)
c.S("Q2", "a", "0", "g2", body_diode=True)
c.S("Q3", "p", "b", "g3", body_diode=True)
c.S("Q4", "b", "0", "g4", body_diode=True)
c.L("Lf", "a", "f", 3e-3)
c.C("Cf", "f", "b", 10e-6)
c.R("Rload", "f", "b", 30)
# ---- modulator
c.add(
    pe.SineWave("ref_a", m, f0),
    pe.Gain("ref_b", "ref_a", -1),
    pe.Triangle("carrier", fsw, -1, 1),
    pe.Comparator("cmd_a", "ref_a", "carrier"),
    pe.Comparator("cmd_b", "ref_b", "carrier"),
    pe.DeadTime("g1", "g2", "cmd_a", 1e-6),
    pe.DeadTime("g3", "g4", "cmd_b", 1e-6),
)
res = c.simulate(dt=0.1e-6, t_end=60e-3, t_record_start=20e-3)
ss = res.last(2 / f0)
print(f"Output RMS     : {ss.rms('V(f,b)'):.1f} V  (ideal {m * 400 / 2 ** 0.5:.1f} V before filter)")
print(f"Output THD     : {100 * ss.thd('V(f,b)', f0, cycles=2):.2f} %")
print(f"Bridge THD     : {100 * ss.thd('V(a,b)', f0, cycles=2, n_harmonics=600):.1f} %")
print(f"Load power     : {ss.power('Rload'):.0f} W")
print(f"Stats          : {res.meta['stats']}")
ss.plot(["V(a,b)", "V(f,b)"], "I(Lf)", save=out("02_inverter_waveforms.png"))
ss.plot_spectrum("V(f,b)", f0, cycles=2, fmax=25e3, save=out("02_inverter_spectrum.png"))
