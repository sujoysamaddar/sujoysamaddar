import math

import numpy as np
import pytest

import openpe as pe
from openpe.control import _tustin


# ---------------------------------------------------------------- basics
def test_parse_value():
    assert pe.parse_value("4.7u") == pytest.approx(4.7e-6)
    assert pe.parse_value("10kHz") == pytest.approx(1e4)
    assert pe.parse_value("2meg") == pytest.approx(2e6)
    assert pe.parse_value("100m") == pytest.approx(0.1)
    assert pe.parse_value("1e-3") == pytest.approx(1e-3)
    with pytest.raises(ValueError):
        pe.parse_value("abc")


@pytest.mark.parametrize("method", ["trap", "be"])
def test_rc_step(method):
    c = pe.Circuit()
    c.V("V1", "in", "0", 10)
    c.R("R1", "in", "out", 1e3)
    c.C("C1", "out", "0", 1e-6)
    r = c.simulate(1e-6, 3e-3, method=method)
    tau = 1e-3
    expected = 10 * (1 - np.exp(-r.t / tau))
    tol = 1e-3 if method == "trap" else 3e-2
    assert np.max(np.abs(r["V(out)"] - expected)) < tol


def test_rl_current_and_energy():
    c = pe.Circuit()
    c.V("V1", "a", "0", 5)
    c.R("R1", "a", "b", 2)
    c.L("L1", "b", "0", 10e-3)
    r = c.simulate(1e-6, 60e-3)
    assert r.final("I(L1)") == pytest.approx(2.5, rel=1e-3)
    tau = 5e-3
    k = np.searchsorted(r.t, tau)
    assert r["I(L1)"][k] == pytest.approx(2.5 * (1 - math.exp(-1)), rel=2e-3)


def test_lc_resonance_frequency():
    L, C = 1e-3, 1e-6
    c = pe.Circuit()
    c.L("L1", "a", "0", L, ic=1.0)
    c.C("C1", "a", "0", C)
    c.R("Rbig", "a", "0", 1e9)
    r = c.simulate(0.2e-6, 2e-3)
    i = r["I(L1)"]
    zc = np.where(np.diff(np.sign(i)) != 0)[0]
    period = 2 * np.mean(np.diff(r.t[zc]))
    assert period == pytest.approx(2 * math.pi * math.sqrt(L * C), rel=1e-3)
    # trapezoidal rule conserves energy of a lossless LC tank
    assert np.max(np.abs(i)) == pytest.approx(1.0, rel=1e-3)


def test_voltage_divider_and_power_sign():
    c = pe.Circuit()
    c.V("V1", "a", "0", 12)
    c.R("R1", "a", "b", 1)
    c.R("R2", "b", "0", 2)
    r = c.simulate(1e-5, 1e-4)
    assert r.final("V(b)") == pytest.approx(8.0)
    assert r.power("R2") == pytest.approx(32.0, rel=1e-6)
    assert r.power("V1") == pytest.approx(-48.0, rel=1e-6)  # source delivers
    assert r.final("V(a,b)") == pytest.approx(4.0)


def test_current_source_and_ground_aliases():
    c = pe.Circuit()
    c.I("I1", "gnd", "a", 2.0)  # 2 A injected into node a
    c.R("R1", "a", "0", 5)
    r = c.simulate(1e-5, 1e-4)
    assert r.final("V(a)") == pytest.approx(10.0)


# ---------------------------------------------------------------- switching
def test_half_wave_rectifier_peak():
    c = pe.Circuit()
    c.V("Vs", "a", "0", pe.Sine(100, 50))
    c.D("D1", "a", "k")
    c.R("R1", "k", "0", 100)
    r = c.simulate(10e-6, 40e-3)
    assert r.max("V(k)") == pytest.approx(100, rel=1e-3)
    assert r.min("V(k)") > -0.1
    assert r.window(20e-3).mean("V(k)") == pytest.approx(100 / math.pi, rel=1e-2)


def test_diode_forward_drop():
    c = pe.Circuit()
    c.V("Vs", "a", "0", 10)
    c.D("D1", "a", "k", vf=0.7)
    c.R("R1", "k", "0", 10)
    r = c.simulate(1e-5, 1e-4)
    assert r.final("V(k)") == pytest.approx(9.3, rel=1e-3)


@pytest.mark.parametrize("duty", [0.25, 0.5, 0.75])
def test_buck_ccm_conversion_ratio(duty):
    c = pe.Circuit()
    c.V("Vin", "in", "0", 48)
    c.S("S1", "in", "sw", "g")
    c.D("D1", "0", "sw")
    c.L("L1", "sw", "out", 200e-6)
    c.C("C1", "out", "0", 100e-6)
    c.R("RL", "out", "0", 4)
    c.add(pe.Constant("d", duty), pe.PWM("g", "d", 20e3))
    r = c.simulate(0.3e-6, 15e-3).last(1e-3)
    assert r.mean("V(out)") == pytest.approx(48 * duty, rel=1e-2)


def test_boost_ccm():
    c = pe.Circuit()
    c.V("Vin", "in", "0", 12)
    c.L("L1", "in", "sw", 200e-6)
    c.S("S1", "sw", "0", "g")
    c.D("D1", "sw", "out")
    c.C("C1", "out", "0", 220e-6)
    c.R("RL", "out", "0", 20)
    c.add(pe.Constant("d", 0.5), pe.PWM("g", "d", 50e3))
    r = c.simulate(0.2e-6, 30e-3).last(2e-3)
    assert r.mean("V(out)") == pytest.approx(24, rel=2e-2)


def test_matrix_cache_is_used():
    c = pe.Circuit()
    c.V("Vin", "in", "0", 48)
    c.S("S1", "in", "sw", "g")
    c.D("D1", "0", "sw")
    c.L("L1", "sw", "out", 100e-6)
    c.R("RL", "out", "0", 5)
    c.add(pe.Constant("d", 0.5), pe.PWM("g", "d", 10e3))
    sim = pe.Simulator(c, 1e-6, 5e-3)
    r = sim.run()
    assert r.meta["stats"]["factorizations"] <= 8
    assert r.meta["stats"]["steps"] == 5000


def test_body_diode_freewheels():
    # half bridge with only the top switch gated: bottom body diode freewheels
    c = pe.Circuit()
    c.V("Vdc", "p", "0", 100)
    c.S("Q1", "p", "mid", "g1", body_diode=True)
    c.S("Q2", "mid", "0", "g2", body_diode=True)
    c.L("L1", "mid", "o", 1e-3)
    c.R("R1", "o", "0", 10)
    c.add(pe.Constant("d", 0.4), pe.PWM("g1", "d", 10e3), pe.Constant("g2", 0))
    r = c.simulate(1e-6, 20e-3).last(1e-3)
    assert r.mean("V(o)") == pytest.approx(40, rel=2e-2)
    assert r.min("I(Q2_D)") >= -1e-3  # off-state leakage through roff
    assert r.max("I(Q2_D)") > 1.0


def test_thyristor_phase_control():
    alpha = 90.0
    f = 50.0
    c = pe.Circuit()
    c.V("Vs", "a", "0", pe.Sine(100, f))
    c.SCR("T1", "a", "k", "g")
    c.R("R1", "k", "0", 10)
    c.add(pe.SquareWave("g", f, duty=0.1, delay=alpha / 360 / f))
    r = c.simulate(5e-6, 60e-3).last(1 / f)
    expected = 100 / (2 * math.pi) * (1 + math.cos(math.radians(alpha)))
    assert r.mean("V(k)") == pytest.approx(expected, rel=2e-2)


def test_ideal_transformer_ratio():
    c = pe.Circuit()
    c.V("Vp", "p", "0", pe.Sine(100, 50))
    c.T("T1", "p", "0", "s", "0", n=4)
    c.R("RL", "s", "0", 10)
    r = c.simulate(10e-6, 40e-3)
    assert r.max("V(s)") == pytest.approx(25, rel=1e-3)
    # power balance
    assert r.window(20e-3).power("RL") == pytest.approx(-r.window(20e-3).power("Vp"), rel=1e-3)


def test_coupled_inductors_tight_coupling_ratio():
    c = pe.Circuit()
    c.V("Vp", "p", "0", pe.Sine(10, 1e3))
    c.add(pe.CoupledInductors("K1", "p", "0", "s", "0", 1e-3, 4e-3, k=0.9999))
    c.R("RL", "s", "0", 1e4)
    r = c.simulate(1e-6, 5e-3).last(1e-3)
    assert r.max("V(s)") == pytest.approx(20, rel=1e-2)


def test_three_phase_diode_rectifier():
    c = pe.Circuit()
    c.add(pe.ThreePhaseSource("G", "a", "b", "c", "0", 400, 50))
    for i, ph in enumerate("abc"):
        c.D(f"Dp{i}", ph, "p")
        c.D(f"Dn{i}", "m", ph)
    c.R("RL", "p", "m", 100)
    r = c.simulate(5e-6, 60e-3).last(20e-3)
    vpk = 400 * math.sqrt(2)
    assert r.mean("V(p,m)") == pytest.approx(3 / math.pi * vpk, rel=1e-2)


# ---------------------------------------------------------------- machines
def test_dc_motor_steady_state():
    V, Ra, ke, bf, tl = 100.0, 1.0, 0.5, 1e-3, 2.0
    c = pe.Circuit()
    c.V("Va", "a", "0", V)
    c.add(pe.DCMotor("M1", "a", "0", ra=Ra, la=5e-3, ke=ke, j=0.01, bf=bf, t_load=tl))
    r = c.simulate(20e-6, 3.0, record_every=50)
    # steady state: V = Ra i + ke w ; ke i = bf w + tl
    w = (ke * V - Ra * tl) / (ke ** 2 + Ra * bf)
    i = (bf * w + tl) / ke
    assert r.final("speed(M1)") == pytest.approx(w, rel=1e-3)
    assert r.final("I(M1)") == pytest.approx(i, rel=1e-3)
    assert r.final("Te(M1)") == pytest.approx(ke * i, rel=1e-3)


# ---------------------------------------------------------------- control
def test_tustin_first_order_dc_gain():
    b, a = _tustin([1.0], [1e-3, 1.0], 1e-5)
    assert sum(b) / sum(a) == pytest.approx(1.0)


def test_transfer_function_step_response():
    c = pe.Circuit()
    c.R("Rd", "x", "0", 1)
    c.V("Vd", "x", "0", 0)  # dummy power circuit
    c.add(pe.StepSignal("u", 0.0), pe.TransferFunction("y", "u", [1.0], [1e-3, 1.0]))
    r = c.simulate(1e-5, 5e-3)
    k = np.searchsorted(r.t, 1e-3)
    assert r["y"][k] == pytest.approx(1 - math.exp(-1), abs=5e-3)
    assert r.final("y") == pytest.approx(1.0, abs=1e-2)


def test_second_order_tf():
    wn, z = 2 * math.pi * 100, 0.2
    c = pe.Circuit()
    c.R("Rd", "x", "0", 1)
    c.add(pe.StepSignal("u", 0.0), pe.TransferFunction("y", "u", [wn ** 2], [1, 2 * z * wn, wn ** 2]))
    r = c.simulate(1e-5, 50e-3)
    overshoot = r.max("y") - 1
    assert overshoot == pytest.approx(math.exp(-z * math.pi / math.sqrt(1 - z * z)), rel=2e-2)


def test_pi_antiwindup_and_limits():
    c = pe.Circuit()
    c.R("Rd", "x", "0", 1)
    c.add(pe.Constant("e", 1.0), pe.PI("y", "e", kp=1, ki=1000, lo=-2, hi=2))
    r = c.simulate(1e-5, 10e-3)
    assert r.max("y") <= 2 + 1e-12


def test_algebraic_loop_detected():
    c = pe.Circuit()
    c.R("Rd", "x", "0", 1)
    c.add(pe.Gain("a", "b", 1), pe.Gain("b", "a", 1))
    with pytest.raises(ValueError, match="algebraic loop"):
        pe.Simulator(c, 1e-5, 1e-4)
    c2 = pe.Circuit()
    c2.R("Rd", "x", "0", 1)
    c2.add(pe.Gain("a", "b", 0.5), pe.UnitDelay("b", "a"))
    pe.Simulator(c2, 1e-5, 1e-4).run()


def test_missing_signal_error():
    c = pe.Circuit()
    c.V("V1", "a", "0", 1)
    c.S("S1", "a", "b", "nosuchgate")
    c.R("R1", "b", "0", 1)
    with pytest.raises(ValueError, match="nosuchgate"):
        pe.Simulator(c, 1e-6, 1e-5)


def test_deadtime():
    c = pe.Circuit()
    c.R("Rd", "x", "0", 1)
    c.add(pe.SquareWave("cmd", 10e3), pe.DeadTime("hi", "lo", "cmd", 2e-6))
    r = c.simulate(0.1e-6, 1e-3)
    assert np.all(r["hi"] * r["lo"] == 0)
    frac_off = np.mean((r["hi"] == 0) & (r["lo"] == 0))
    assert frac_off == pytest.approx(2 * 2e-6 * 10e3, rel=0.1)


def test_python_block_and_sample_hold():
    def counter(t, dt, u, st):
        st["n"] = st.get("n", 0) + 1
        return st["n"]

    c = pe.Circuit()
    c.R("Rd", "x", "0", 1)
    c.add(pe.PythonBlock("n", [], counter), pe.SampleHold("nh", "n", fs=1e3))
    r = c.simulate(1e-5, 10e-3)
    assert r.final("n") == 1001  # one call at t=0 plus 1000 steps
    assert len(np.unique(r["nh"])) == pytest.approx(10, abs=1)


def test_park_transform_roundtrip():
    c = pe.Circuit()
    c.R("Rd", "x", "0", 1)
    f = 50
    c.add(
        pe.SineWave("a", 1, f, 90), pe.SineWave("b", 1, f, -30), pe.SineWave("c", 1, f, 210),
        pe.Ramp("th", 2 * math.pi * f),
        pe.ABCtoDQ("d", "q", "a", "b", "c", "th"),
        pe.DQtoABC("a2", "b2", "c2", "d", "q", "th"),
    )
    r = c.simulate(1e-5, 20e-3)
    assert np.allclose(r["d"], 1.0, atol=1e-9)
    assert np.allclose(r["q"], 0.0, atol=1e-9)
    assert np.allclose(r["a2"], r["a"], atol=1e-9)


def test_pll_locks():
    c = pe.Circuit()
    c.R("Rd", "x", "0", 1)
    c.add(pe.SineWave("v", 1.0, 52.0, 30), pe.PLL("th", "f", "v", 50.0))
    r = c.simulate(20e-6, 0.5).last(0.1)
    assert r.mean("f") == pytest.approx(52.0, rel=5e-3)


# ---------------------------------------------------------------- post-processing
def test_fft_thd_square_wave():
    c = pe.Circuit()
    c.V("Vs", "a", "0", pe.Square(-1, 1, 50))
    c.R("R1", "a", "0", 1)
    r = c.simulate(2e-6, 40e-3)
    a1, _ = r.harmonic("V(a)", 50)
    assert a1 == pytest.approx(4 / math.pi, rel=1e-3)
    # THD of an ideal square wave (with infinite harmonics) = 48.3 %
    assert r.thd("V(a)", 50, cycles=1, n_harmonics=999) == pytest.approx(0.4834, rel=1e-2)


def test_rms_and_power_factor():
    c = pe.Circuit()
    c.V("Vs", "a", "0", pe.Sine(math.sqrt(2) * 230, 50))
    c.R("R1", "a", "b", 10)
    c.L("L1", "b", "0", 10 / (2 * math.pi * 50))  # X = R -> pf = 0.707
    r = c.simulate(10e-6, 200e-3).last(20e-3)
    assert r.rms("V(a)") == pytest.approx(230, rel=1e-3)
    assert r.power_factor("V(a)", "I(L1)") == pytest.approx(1 / math.sqrt(2), rel=1e-2)


def test_csv_export(tmp_path):
    c = pe.Circuit()
    c.V("V1", "a", "0", 1)
    c.R("R1", "a", "0", 1)
    r = c.simulate(1e-3, 1e-2)
    p = tmp_path / "out.csv"
    r.to_csv(p, ["V(a)", "I(R1)"])
    lines = p.read_text().splitlines()
    assert lines[0] == "time,V(a),I(R1)"
    assert len(lines) == 12  # header + t=0 + 10 steps


# ---------------------------------------------------------------- analyses
def _rc_build(R=1e3, C=1e-6):
    c = pe.Circuit()
    c.V("V1", "in", "0", 10)
    c.R("R1", "in", "out", R)
    c.C("C1", "out", "0", C)
    return c


def test_parameter_sweep():
    rows = pe.parameter_sweep(_rc_build, {"R": [1e3, 2e3]}, dt=1e-6, t_end=1e-3,
                              measure=lambda r: {"v": r.final("V(out)")})
    assert rows[0]["v"] == pytest.approx(10 * (1 - math.exp(-1)), rel=1e-3)
    assert rows[1]["v"] == pytest.approx(10 * (1 - math.exp(-0.5)), rel=1e-3)
    assert "R" in pe.format_table(rows)


def test_ac_sweep_rc_lowpass():
    def build():
        c = pe.Circuit()
        c.add(pe.Constant("ac_pert", 0.0))
        c.V("V1", "in", "0", signal="ac_pert")
        c.R("R1", "in", "out", 1e3)
        c.C("C1", "out", "0", 1e-6)
        c.add(pe.VoltageSensor("vo", "out"))
        return c

    fc = 1 / (2 * math.pi * 1e-3)
    resp = pe.ac_sweep(build, [fc / 10, fc, fc * 10], "vo", amplitude=1.0, dt=2e-6, t_settle=5e-3, cycles=2)
    assert resp["mag_db"][1] == pytest.approx(-3.01, abs=0.1)
    assert resp["phase_deg"][1] == pytest.approx(-45, abs=2)
    assert resp["mag_db"][2] == pytest.approx(-20.04, abs=0.3)


# ---------------------------------------------------------------- netlist
NETLIST = """
* closed loop buck
.param vin=48
Vin in 0 DC {vin}
S1 in sw g
D1 0 sw
L1 sw out 200u
C1 out 0 100u
R1 out 0 4
B1 VSENSOR vo out 0
B2 CONST vref value=12
B3 SUM err vref vo signs=+-
B4 PI d err kp=0.01 ki=50 lo=0 hi=0.95
B5 PWM g d freq=20k
.tran 0.5u 30m
.plot V(out) I(L1),I(D1)
.measure mean V(out)
.end
"""


def test_netlist_closed_loop_buck():
    nl = pe.parse_netlist(NETLIST)
    assert nl.tran["dt"] == pytest.approx(0.5e-6)
    r = nl.run()
    assert r.last(2e-3).mean("V(out)") == pytest.approx(12.0, rel=1e-2)
    assert nl.plots == [["V(out)"], ["I(L1)", "I(D1)"]]


def test_netlist_errors_have_line_numbers():
    with pytest.raises(ValueError, match="line 1"):
        pe.parse_netlist("R1 a 0 xyz")


def test_subcircuit():
    stage = pe.Circuit()
    stage.R("Ra", "in", "out", 1)
    stage.R("Rb", "out", "0", 1)
    c = pe.Circuit()
    c.V("V1", "src", "0", 8)
    c.subcircuit("X1", stage, {"in": "src", "out": "mid"})
    c.subcircuit("X2", stage, {"in": "mid", "out": "end"})
    r = c.simulate(1e-5, 1e-4)
    # second divider loads the first: Vmid = 8 * (2/3)/(1+2/3) = 3.2
    assert r.final("V(mid)") == pytest.approx(3.2)
    assert r.final("V(end)") == pytest.approx(1.6)


def test_initial_conditions_respected_at_t0():
    c = pe.Circuit()
    c.C("C1", "a", "0", 1e-6, ic=5.0)
    c.R("R1", "a", "0", 1e3)
    c.add(pe.VoltageSensor("va", "a"))
    r = c.simulate(1e-6, 1e-3)
    assert r.t[0] == 0.0
    assert r["V(a)"][0] == pytest.approx(5.0, rel=1e-4)
    assert r["va"][0] == pytest.approx(5.0, rel=1e-4)   # sensors see the IC at t=0
    k = np.searchsorted(r.t, 1e-3)
    assert r["V(a)"][k - 1] == pytest.approx(5 * math.exp(-1), rel=2e-3)


def test_plot_groups_keep_differential_names():
    nl = pe.parse_netlist("R1 a 0 1\nV1 a 0 DC 1\n.plot V(a,0),I(R1) I(V1)\n.tran 1u 10u\n.window 5u\n")
    assert nl.plots == [["V(a,0)", "I(R1)"], ["I(V1)"]]
    assert nl.window == pytest.approx(5e-6)
    r = nl.run()
    assert r.final("V(a,0)") == pytest.approx(1.0)


@pytest.mark.parametrize("name", ["rlc_step", "three_phase_rectifier", "buck_current_mode", "dc_motor"])
def test_bundled_netlists_parse(name):
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "circuits", f"{name}.net")
    nl = pe.load_netlist(path)
    assert nl.tran["dt"] > 0
    pe.Simulator(nl.circuit, nl.tran["dt"], nl.tran["t_end"])  # compiles


def test_cli_runs_netlist(tmp_path, capsys):
    import os
    from openpe.__main__ import main
    path = os.path.join(os.path.dirname(__file__), "..", "circuits", "rlc_step.net")
    assert main(["run", path, "--csv", str(tmp_path / "o.csv")]) == 0
    out = capsys.readouterr().out
    assert "max V(b)" in out
    # underdamped series RLC, zeta = 0.25 -> 44.4 % overshoot
    val = float(out.split("max V(b)")[1].split("=")[1].split()[0])
    assert val == pytest.approx(10 * (1 + math.exp(-math.pi * 0.25 / math.sqrt(1 - 0.0625))), rel=2e-3)
