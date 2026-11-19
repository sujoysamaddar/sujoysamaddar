"""Tests for the SPICE-style analog features: nonlinear devices, controlled
sources, op-amps, .op / .dc / .ac analyses and Monte Carlo."""
import math

import numpy as np
import pytest

import openpe as pe

VT = pe.thermal_voltage()


def _bisect(f, lo, hi):
    for _ in range(200):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if f(m) > 0 else (lo, m)
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- diodes
def test_shockley_diode_operating_point():
    c = pe.Circuit()
    c.V("V1", "a", "0", 5)
    c.R("R1", "a", "d", 1e3)
    c.Dpn("D1", "d", "0", is_=1e-14)
    op = c.op()
    vd = _bisect(lambda v: (5 - v) / 1e3 - 1e-14 * (math.exp(v / VT) - 1), 0, 1)
    assert op["V(d)"] == pytest.approx(vd, abs=1e-9)
    assert op["I(D1)"] == pytest.approx((5 - vd) / 1e3, rel=1e-6)


def test_diode_series_resistance_and_emission_coefficient():
    c = pe.Circuit()
    c.V("V1", "a", "0", 1.2)
    c.Dpn("D1", "a", "0", is_=1e-12, n=2, rs=10)
    op = c.op()
    i = op["I(D1)"]
    assert 1.2 == pytest.approx(10 * i + 2 * VT * math.log(i / 1e-12 + 1), abs=1e-6)


def test_zener_regulator():
    c = pe.Circuit()
    c.V("V1", "a", "0", 12)
    c.R("R1", "a", "z", 500)
    c.Z("DZ", "0", "z", bv=5.1, ibv=1e-3)
    op = c.op()
    iz = (12 - op["V(z)"]) / 500
    # breakdown law: iz = ibv * exp((vz - bv) / vt)
    assert op["V(z)"] == pytest.approx(5.1 + VT * math.log(iz / 1e-3), abs=1e-3)


def test_diode_reverse_leakage():
    c = pe.Circuit()
    c.V("V1", "a", "0", -5)
    c.Dpn("D1", "a", "0", is_=1e-14)
    assert c.op()["I(D1)"] == pytest.approx(-1e-14 - 5e-12, rel=1e-3)  # Is + gmin leakage


def test_halfwave_rectifier_with_real_diode():
    c = pe.Circuit()
    c.V("Vs", "a", "0", pe.Sine(10, 50))
    c.Dpn("D1", "a", "k")
    c.R("R1", "k", "0", 1e3)
    r = c.simulate(20e-6, 40e-3)
    ipk = r.max("I(D1)")
    vd = VT * math.log(ipk / 1e-14)
    assert r.max("V(k)") == pytest.approx(10 - vd, abs=5e-3)
    assert r.meta["stats"]["nonconverged"] == 0


# ---------------------------------------------------------------- BJT
def _ce_bias(polarity="npn"):
    s = 1 if polarity == "npn" else -1
    c = pe.Circuit()
    c.V("Vcc", "vcc", "0", 12 * s)
    c.R("RB1", "vcc", "b", 47e3)
    c.R("RB2", "b", "0", 10e3)
    c.R("RC", "vcc", "c", 2.2e3)
    c.R("RE", "e", "0", 1e3)
    c.Q("Q1", "c", "b", "e", polarity, bf=150, is_=1e-14)
    return c


@pytest.mark.parametrize("polarity", ["npn", "pnp"])
def test_bjt_bias_point(polarity):
    op = _ce_bias(polarity).op()
    s = 1 if polarity == "npn" else -1
    ic, ib = s * op["I(Q1)"], s * op["Ib(Q1)"]
    vbe = s * op["Vbe(Q1)"]
    assert ic / ib == pytest.approx(150, rel=1e-6)
    assert vbe == pytest.approx(VT * math.log(ic / 1e-14), abs=1e-6)
    # Thevenin base loop: Vth = Ib*Rth + Vbe + (Ic+Ib)*RE
    vth, rth = 12 * 10 / 57, 47e3 * 10e3 / 57e3
    assert vth == pytest.approx(ib * rth + vbe + (ic + ib) * 1e3, abs=1e-6)
    assert s * op["V(c)"] == pytest.approx(12 - 2.2e3 * ic, abs=1e-6)


def test_bjt_early_effect_output_conductance():
    c = pe.Circuit()
    c.V("Vb", "b", "0", 0.65)
    c.V("Vc", "c", "0", 5)
    c.Q("Q1", "c", "b", "0", is_=1e-14, bf=200, vaf=50)
    r = c.dc("Vc", np.linspace(2, 10, 9))
    ic = -r["I(Vc)"]
    slope = np.polyfit(np.linspace(2, 10, 9), ic, 1)[0]
    # gout ~ Ic/VA (relative to Ic at Vce -> 0 intercept)
    assert slope == pytest.approx(ic[0] / (50 + 2 - 0.65), rel=0.05)


def test_bjt_small_signal_gm_in_ac():
    c = _ce_bias()
    c.elements = [e for e in c.elements if e.name != "RE"]
    c.R("RE", "e", "0", 1e3)
    c.C("CE", "e", "0", 1.0)  # bypass
    c.V("Vac", "src", "0", 0, ac=1e-3)
    c.C("Cin", "src", "b", 1.0)
    op = c.op()
    ac = c.ac([1e3])
    gm = op["I(Q1)"] / VT
    gain = abs(ac["V(c)"][0]) / 1e-3
    assert gain == pytest.approx(gm * 2.2e3, rel=1e-3)


# ---------------------------------------------------------------- MOSFET
def test_nmos_saturation_and_triode():
    c = pe.Circuit()
    c.V("Vdd", "d", "0", 5)
    c.V("Vg", "g", "0", 2)
    c.M("M1", "d", "g", "0", vto=1, kp=1e-3)
    op = c.op()
    assert op["I(M1)"] == pytest.approx(0.5e-3, rel=1e-6)
    assert op["region(M1)"] == 2
    r = c.dc("Vdd", [0.5])
    assert r["I(M1)"][0] == pytest.approx(1e-3 * (1 * 0.5 - 0.125), rel=1e-6)
    assert r["region(M1)"][0] == 1


def test_pmos_and_lambda():
    c = pe.Circuit()
    c.V("Vdd", "s", "0", 5)
    c.V("Vg", "g", "0", 3)
    c.R("RD", "d", "0", 1e3)
    c.M("M1", "d", "g", "s", "pmos", vto=-1, kp=1e-3, lam=0.02)
    op = c.op()
    vsd = 5 - op["V(d)"]
    assert -op["I(M1)"] == pytest.approx(0.5e-3 * (1 + 0.02 * vsd), rel=1e-6)
    assert op["V(d)"] == pytest.approx(-op["I(M1)"] * 1e3, rel=1e-6)


def test_cmos_inverter_vtc_symmetry():
    c = pe.Circuit()
    c.V("Vdd", "vdd", "0", 5)
    c.V("Vin", "in", "0", 0)
    c.M("MP", "out", "in", "vdd", "pmos", vto=-1, kp=1e-3)
    c.M("MN", "out", "in", "0", "nmos", vto=1, kp=1e-3)
    vin = np.linspace(0, 5, 201)
    r = c.dc("Vin", vin)
    vout = r["V(out)"]
    assert vout[0] == pytest.approx(5, abs=1e-3)
    assert vout[-1] == pytest.approx(0, abs=1e-3)
    # matched devices: switching threshold at Vdd/2
    vm = np.interp(2.5, vout[::-1], vin[::-1])
    assert vm == pytest.approx(2.5, abs=0.03)


def test_mosfet_source_drain_symmetry():
    c = pe.Circuit()
    c.V("Vg", "g", "0", 3)
    c.V("Vx", "x", "0", -0.2)  # reversed drain-source voltage
    c.M("M1", "x", "g", "0", vto=1, kp=1e-3)
    op = c.op()
    # source and drain swap: vgs' = 3.2, vds' = 0.2 (triode)
    assert op["I(M1)"] == pytest.approx(-1e-3 * (2.2 * 0.2 - 0.02), rel=1e-6)


# ---------------------------------------------------------------- controlled sources / op-amp
def test_controlled_sources():
    c = pe.Circuit()
    c.V("V1", "a", "0", 2)
    c.R("R1", "a", "0", 1e3)
    c.E("E1", "e", "0", "a", "0", 3)
    c.R("Re", "e", "0", 1e3)
    c.G("G1", "0", "g", "a", "0", 1e-3)
    c.R("Rg", "g", "0", 1e3)
    c.F("F1", "0", "f", "V1", 2)
    c.R("Rf", "f", "0", 1e3)
    c.H("H1", "h", "0", "V1", 500)
    c.R("Rh", "h", "0", 1e3)
    op = c.op()
    i_v1 = op["I(V1)"]
    assert i_v1 == pytest.approx(-2e-3)
    assert op["V(e)"] == pytest.approx(6.0)
    assert op["V(g)"] == pytest.approx(2.0)
    assert op["V(f)"] == pytest.approx(2 * i_v1 * 1e3)
    assert op["V(h)"] == pytest.approx(500 * i_v1)


def test_inductor_is_dc_short_and_controls_cccs():
    c = pe.Circuit()
    c.V("V1", "a", "0", 10)
    c.R("R1", "a", "b", 100)
    c.L("L1", "b", "0", 1e-3)
    c.F("F1", "0", "y", "L1", 3)
    c.R("Ry", "y", "0", 10)
    op = c.op()
    assert op["V(b)"] == pytest.approx(0.0, abs=1e-9)
    assert op["V(y)"] == pytest.approx(3 * 0.1 * 10)


def test_opamp_inverting_amplifier_and_gbw():
    c = pe.Circuit()
    c.V("Vin", "i", "0", 0.5, ac=1)
    c.R("R1", "i", "n", 1e3)
    c.R("R2", "n", "o", 10e3)
    c.opamp("U1", "0", "n", "o", gain=1e5, gbw=1e6)
    a0 = 1e5
    assert c.op()["V(o)"] == pytest.approx(-0.5 * 10 / (1 + 11 / a0), rel=1e-6)
    ac = c.ac(np.logspace(0, 7, 701))
    assert ac.mag_db("V(o)")[0] == pytest.approx(20, abs=0.01)
    assert ac.bandwidth("V(o)") == pytest.approx(1e6 / 11, rel=0.01)


def test_opamp_saturation_transient():
    c = pe.Circuit()
    c.V("Vin", "i", "0", pe.Sine(2, 1e3))
    c.R("R1", "i", "n", 1e3)
    c.R("R2", "n", "o", 10e3)
    c.opamp("U1", "0", "n", "o", gain=1e5, vmax=12, vmin=-12)
    r = c.simulate(2e-6, 2e-3)
    assert r.max("V(o)") == pytest.approx(12, abs=0.05)
    assert r.min("V(o)") == pytest.approx(-12, abs=0.05)
    assert r.meta["stats"]["nonconverged"] == 0


# ---------------------------------------------------------------- AC analysis
def test_ac_rc_lowpass_exact():
    c = pe.Circuit()
    c.V("V1", "in", "0", 0, ac=1)
    c.R("R1", "in", "out", 1e3)
    c.C("C1", "out", "0", 1e-6)
    f = np.logspace(-1, 5, 601)
    ac = c.ac(f)
    h = 1 / (1 + 2j * np.pi * f * 1e-3)
    assert np.allclose(ac["V(out)"], h, rtol=1e-9, atol=1e-12)
    assert ac.bandwidth("V(out)") == pytest.approx(1 / (2 * np.pi * 1e-3), rel=1e-3)


def test_ac_series_rlc_resonance_and_current():
    L, C, R = 1e-3, 1e-6, 10.0
    c = pe.Circuit()
    c.V("V1", "in", "0", 0, ac=1)
    c.R("R1", "in", "a", R)
    c.L("L1", "a", "b", L)
    c.C("C1", "b", "0", C)
    f0 = 1 / (2 * np.pi * np.sqrt(L * C))
    ac = c.ac([f0])
    assert abs(ac["I(L1)"][0]) == pytest.approx(1 / R, rel=1e-6)
    assert abs(ac["V(b)"][0]) == pytest.approx(np.sqrt(L / C) / R, rel=1e-6)  # = Q


def test_ac_coupled_inductor_and_transformer():
    c = pe.Circuit()
    c.V("V1", "p", "0", 0, ac=1)
    c.T("T1", "p", "0", "s", "0", n=5)
    c.R("RL", "s", "0", 100)
    ac = c.ac([50.0, 5e3])
    assert np.allclose(np.abs(ac["V(s)"]), 0.2)


def test_ac_requires_excitation():
    c = pe.Circuit()
    c.V("V1", "a", "0", 1)
    c.R("R1", "a", "0", 1)
    with pytest.raises(ValueError, match="AC excitation"):
        c.ac([1e3])
    assert abs(c.ac([1e3], source="V1")["V(a)"][0]) == pytest.approx(1.0)


def test_ac_dc_motor_speed_response():
    c = pe.Circuit()
    c.V("Va", "a", "0", 100, ac=1)
    c.add(pe.DCMotor("M1", "a", "0", ra=1, la=5e-3, ke=0.5, j=0.01, bf=1e-3))
    sim = pe.Simulator(c, 1e-6, 0)
    w_branch = next(e for e in sim.elements if e.name == "M1").br[1]
    res = sim.ac_analysis([1e-3])
    # quasi-static speed response dw/dV = ke / (ke*kt + Ra*B)
    dwdv = abs(res._x[0, w_branch])
    assert dwdv == pytest.approx(0.5 / (0.25 + 1e-3), rel=1e-3)


# ---------------------------------------------------------------- DC sweep & initial OP
def test_dc_sweep_diode_iv_monotonic():
    c = pe.Circuit()
    c.V("V1", "a", "0", 0)
    c.R("R1", "a", "d", 100)
    c.Dpn("D1", "d", "0", bv=5.6)
    v = np.linspace(-10, 3, 131)
    r = c.dc("V1", v)
    i = r["I(D1)"]
    assert np.all(np.diff(i) > 0)
    assert r["V(d)"][0] == pytest.approx(-5.6 - VT * math.log(((10 - 5.6) / 100) / 1e-3), abs=5e-3)
    assert r.meta["xlabel"] == "V1 value"


def test_transient_from_operating_point():
    c = pe.Circuit()
    c.V("V1", "a", "0", 10)
    c.R("R1", "a", "b", 1e3)
    c.C("C1", "b", "0", 1e-6)
    c.R("R2", "b", "0", 1e3)
    r = c.simulate(1e-6, 1e-3, initial="op")
    assert np.allclose(r["V(b)"], 5.0, atol=1e-9)  # already in steady state
    r2 = c.simulate(1e-6, 1e-3)                   # default: from initial conditions
    assert r2["V(b)"][0] == pytest.approx(0.0, abs=1e-6)


def test_source_stepping_rescues_hard_op():
    # cascade of diodes from a large voltage: plain Newton from 0 is hard
    c = pe.Circuit()
    c.V("V1", "a", "0", 100)
    prev = "a"
    for k in range(6):
        c.Dpn(f"D{k}", prev, f"n{k}", is_=1e-16, n=1)
        prev = f"n{k}"
    c.R("R1", prev, "0", 10)
    op = c.op()
    i = op["I(R1)"]
    assert op["V(a)"] - op[f"V({prev})"] == pytest.approx(6 * VT * math.log(i / 1e-16), abs=1e-4)


# ---------------------------------------------------------------- Monte Carlo
def _rc_fc(R, C):
    c = pe.Circuit()
    c.V("V1", "in", "0", 0, ac=1)
    c.R("R1", "in", "out", R)
    c.C("C1", "out", "0", C)
    return {"fc": c.ac(np.logspace(1, 5, 200)).bandwidth("V(out)")}


def test_monte_carlo_reproducible_and_bounded():
    mc = pe.monte_carlo(_rc_fc, {"R": 1e3, "C": 1e-6}, {"R": 0.05, "C": 0.10}, n=30, seed=1)
    mc2 = pe.monte_carlo(_rc_fc, {"R": 1e3, "C": 1e-6}, {"R": 0.05, "C": 0.10}, n=30, seed=1)
    assert np.array_equal(mc.values("fc"), mc2.values("fc"))
    fc0 = 1 / (2 * np.pi * 1e-3)
    st = mc.stats("fc")
    assert fc0 / (1.05 * 1.10) * 0.99 <= st["min"] <= st["max"] <= fc0 / (0.95 * 0.90) * 1.01
    assert 0 < mc.yield_("fc", fc0 * 0.9, fc0 * 1.1) <= 1
    assert "fc" in mc.summary()
    for row in mc.rows:
        assert row["fc"] == pytest.approx(1 / (2 * np.pi * row["R"] * row["C"]), rel=0.01)


# ---------------------------------------------------------------- netlist
def test_netlist_analog_analyses():
    text = """
    * RC low-pass with SPICE source syntax
    V1 in 0 DC 1 AC 1 SIN(0 1 1k)
    R1 in out 1k
    C1 out 0 1u
    Q1 c b 0 NPN is=1e-14 bf=100
    VB b 0 0.7
    RCx vcc c 100
    VCC vcc 0 5
    M1 d g 0 NMOS vto=1 kp=1m lambda=0
    VG g 0 2
    RD vcc d 1k
    E1 e 0 out 0 2
    RE e 0 1k
    XU1 OPAMP out fb fb gain=1e5
    RFB fb 0 1meg
    .op
    .dc V1 0 2 0.5
    .ac dec 10 10 100k
    .end
    """
    nl = pe.parse_netlist("\n".join(line.strip() for line in text.splitlines()))
    kinds = [a[0] for a in nl.analyses]
    assert kinds == ["op", "dc", "ac"]
    op = nl.op()
    assert op["V(out)"] == pytest.approx(1.0)
    assert op["V(e)"] == pytest.approx(2.0)
    assert op["V(fb)"] == pytest.approx(1.0, rel=1e-4)
    assert op["I(M1)"] == pytest.approx(0.5e-3, rel=1e-6)
    assert op["I(Q1)"] == pytest.approx(1e-14 * math.exp(0.7 / VT), rel=1e-6)
    dc = nl.dc()
    assert list(dc.t) == [0.0, 0.5, 1.0, 1.5, 2.0]
    ac = nl.ac()
    assert len(ac.freq) == 41
    assert ac.bandwidth("V(out)") == pytest.approx(159.15, rel=0.01)
    src = nl.circuit.element("V1")
    assert src.wave(0.25e-3) == pytest.approx(1.0)  # SPICE order: offset amp freq


def test_netlist_shockley_and_zener_diode_selection():
    nl = pe.parse_netlist("V1 a 0 DC 1\nD1 a k is=1e-14\nD2 k 0\nD3 0 a bv=5\n.op\n")
    types = {e.name: type(e).__name__ for e in nl.circuit.elements}
    assert types == {"V1": "VoltageSource", "D1": "PNDiode", "D2": "Diode", "D3": "PNDiode"}


def test_dc_sweep_overrides_separate_dc_value():
    """Regression: sweeping a source that has both a DC value and a
    transient waveform must sweep the DC value."""
    c = pe.Circuit()
    c.V("V1", "a", "0", pe.Sine(1, 1e3), dc=1.0)
    c.R("R1", "a", "0", 1e3)
    r = c.dc("V1", [0.0, 0.5, 2.0])
    assert list(r["V(a)"]) == pytest.approx([0.0, 0.5, 2.0])
