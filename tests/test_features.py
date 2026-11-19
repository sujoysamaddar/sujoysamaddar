"""Loss and thermal analysis, step-response metrics, target solving,
sub-circuits, saturable inductor, lookup table and space-vector PWM."""
import math
import os

import numpy as np
import pytest

import openpe as pe
from openpe.__main__ import main
from openpe.measures import measure
from openpe.server import simulate_netlist
from openpe.tuning import solve_for

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================== losses
def _chopper(**sw):
    """Resistive load switched at 10 kHz: every cycle has exactly one turn-on and one turn-off."""
    c = pe.Circuit()
    c.V("Vin", "in", "0", 100)
    c.S("S1", "in", "out", "g", ron=0.1, **sw)
    c.R("RL", "out", "0", 9.9)                      # on-state current = 100 / 10 = 10 A
    c.add(pe.Constant("d", 0.4), pe.PWM("g", "d", 10e3))
    return c


def test_conduction_loss_includes_on_state_drop():
    c = pe.Circuit()
    c.V("V1", "a", "0", 20)
    c.S("S1", "a", "b", "g", ron=0.05, vf=2.0)
    c.R("R1", "b", "0", 3.95)
    c.add(pe.Constant("g", 1))
    r = c.simulate(1e-5, 1e-3)
    i = (20 - 2.0) / (3.95 + 0.05)
    row = r.last(0.5e-3).loss_report().rows[0]
    assert row.name == "S1" and row.kind == "Switch"
    assert row.conduction == pytest.approx(2.0 * i + 0.05 * i * i, rel=1e-6)
    assert row.switching == 0.0


def test_switching_energy_matches_per_event_formula():
    eon, eoff, vref, iref = 120e-6, 200e-6, 50.0, 5.0
    r = _chopper(eon=eon, eoff=eoff, vref=vref, iref=iref).simulate(0.2e-6, 5e-3)
    w = r.window(1e-3, 5e-3)                        # 40 whole cycles
    rep = w.loss_report()
    v, i = 100.0, 100.0 / 10.0
    per_cycle = (eon + eoff) * (v / vref) * (i / iref)
    assert rep.rows[0].switching == pytest.approx(per_cycle * 10e3, rel=2e-3)
    assert "Esw(S1)" in r.data and np.all(np.diff(r["Esw(S1)"]) >= 0)       # cumulative energy only grows


def test_diode_reverse_recovery_and_body_diode_options():
    c = pe.Circuit()
    c.V("Vin", "in", "0", 48)
    c.S("S1", "in", "sw", "g", ron=20e-3, body_diode=True, erec=40e-6, vref=48, iref=2)
    c.add(pe.Constant("d", 0.5), pe.PWM("g", "d", 20e3))
    c.L("L1", "sw", "out", 400e-6)
    c.R("RL", "out", "0", 12)
    c.C("C1", "out", "0", 50e-6)
    c.S("S2", "sw", "0", "g2", ron=20e-3, body_diode=True, erec=40e-6, vref=48, iref=2)   # synchronous leg, never gated
    c.add(pe.Constant("g2", 0))
    r = c.simulate(0.25e-6, 12e-3)
    rep = r.last(2e-3).loss_report()
    names = {row.name: row for row in rep.rows}
    assert {"S1", "S2", "S2_D"} <= set(names)                  # the body diode is reported as its own device
    assert names["S2_D"].switching > 0                         # reverse recovery -> switching loss
    assert names["S1"].switching == 0                          # S1 itself has no eon/eoff


def test_efficiency_accounts_for_switching_loss_that_is_not_in_the_circuit():
    c = pe.Circuit("buck")
    c.V("Vin", "in", "0", 48)
    c.S("S1", "in", "sw", "g", ron=20e-3, eon=50e-6, eoff=80e-6, vref=48, iref=2.0)
    c.D("D1", "0", "sw", vf=0.6, ron=10e-3, erec=10e-6, vref=48, iref=2.0)
    c.L("L1", "sw", "out", 500e-6)
    c.C("C1", "out", "0", 100e-6)
    c.R("RL", "out", "0", 6)
    c.add(pe.Constant("d", 0.25), pe.PWM("g", "d", 20e3))
    rep = c.simulate(0.25e-6, 40e-3).last(10e-3).loss_report(output="RL", input="Vin")
    assert rep.switching_loss > 2.0 and rep.conduction_loss > 0.5
    assert rep.efficiency == pytest.approx(rep.p_out / (rep.p_in - rep.p_storage + rep.switching_loss))
    assert abs(rep.other_loss) < 0.02 * rep.p_in                # circuit power balance closes
    d = rep.to_dict()
    assert d["efficiency"] == pytest.approx(rep.efficiency) and len(d["rows"]) == 2
    assert "efficiency" in str(rep) and "S1" in str(rep)


def test_stored_energy_drift_does_not_count_as_loss():
    """A window in which a capacitor is still charging: the energy going into C must be
    separated from the real losses, otherwise 'other losses' would be huge."""
    c = pe.Circuit()
    c.V("V1", "a", "0", 10)
    c.S("S1", "a", "b", "g", ron=0.5)
    c.R("R1", "b", "c", 100)
    c.C("C1", "c", "0", 100e-6)
    c.add(pe.Constant("g", 1))
    r = c.simulate(1e-5, 20e-3)                                    # tau = 10 ms: charging the whole time
    rep = r.window(2e-3, 10e-3).loss_report(output="R1", input="V1")
    assert rep.p_storage > 0.1 * rep.p_in                           # a large part of the input charges the capacitor
    assert abs(rep.other_loss) < 0.01 * rep.p_in                    # ...and is not booked as a loss
    assert rep.efficiency == pytest.approx(rep.p_out / (rep.p_in - rep.p_storage), rel=1e-9)
    assert rep.to_dict()["p_storage"] == pytest.approx(rep.p_storage)
    # inductors too
    L = pe.Circuit()
    L.V("V1", "a", "0", 10)
    L.R("R1", "a", "b", 5)
    L.L("L1", "b", "0", 20e-3)                                      # tau = 4 ms
    rl = L.simulate(1e-5, 12e-3).window(1e-3, 6e-3).loss_report(output="R1", input="V1")
    assert rl.p_storage > 0 and abs(rl.other_loss) < 0.01 * rl.p_in


def test_loss_parameters_validated():
    with pytest.raises(ValueError, match="vref and iref"):
        pe.Switch("S1", "a", "b", "g", eon=1e-6, vref=0)


def test_junction_temperature_foster_network():
    t = np.arange(0, 0.2, 1e-4)
    res = pe.Results(t, {"V(a)": np.zeros_like(t)}, {"dt": 1e-4})
    p = np.full_like(t, 10.0)                                     # 10 W step
    tj = pe.junction_temperature(res, p, rth=2.0, cth=0.01, ta=40)    # tau = 20 ms
    assert tj[0] == pytest.approx(40.0, abs=0.3)
    assert tj[np.searchsorted(t, 0.02)] == pytest.approx(40 + 20 * (1 - math.exp(-1)), abs=0.1)
    assert tj[-1] == pytest.approx(40 + 20 * (1 - math.exp(-10)), abs=0.05)
    two = pe.junction_temperature(res, p, stages=[(1.0, 0.005), (1.0, 0.05)], ta=0)
    assert two[-1] == pytest.approx(20 * (1 - 0.5 * math.exp(-0.2 / 0.005) - 0.5 * math.exp(-0.2 / 0.05)), abs=0.05)
    assert pe.junction_temperature(res, p, rth=2.0, ta=40)[5] == pytest.approx(60.0)   # no capacitance: instantaneous
    with pytest.raises(ValueError, match="rth"):
        pe.junction_temperature(res, p)
    with pytest.raises(ValueError, match="one sample"):
        pe.junction_temperature(res, p[:5], rth=1.0)


def test_junction_temperature_of_a_device_uses_conduction_and_switching_power():
    r = _chopper(eon=100e-6, eoff=100e-6, vref=100, iref=10).simulate(0.5e-6, 20e-3)
    tj = r.junction_temperature("S1", rth=1.5, cth=0.2, ta=30)       # tau = 0.3 s: still heating up
    p_avg = r.loss_report().rows[0].total
    assert tj[0] == pytest.approx(30, abs=0.1)
    cycle = int(round(1 / 10e3 / 0.5e-6))                              # samples per switching period
    assert tj[-cycle:].mean() > tj[:cycle].mean()                      # heats up overall...
    assert np.diff(tj).min() < 0                                       # ...but ripples: it cools while the switch is off
    assert tj[-1] - 30 == pytest.approx(p_avg * 1.5 * (1 - math.exp(-20e-3 / 0.3)), rel=0.05)


def test_losses_directive_server_and_cli(capsys):
    text = """V1 in 0 DC 100
S1 in out g ron=0.1 eon=100u eoff=100u vref=100 iref=10
R1 out 0 9.9
B1 CONST d value=0.4
B2 PWM g d freq=10k
.tran 0.5u 4m
.window 2m
.losses in=V1 out=R1
.end
"""
    nl = pe.parse_netlist(text)
    assert nl.losses == {"devices": [], "input": ["V1"], "output": ["R1"]}
    res = simulate_netlist(text)
    L = res["losses"]
    assert L["rows"][0]["name"] == "S1" and L["rows"][0]["switching"] == pytest.approx(200e-6 * 10e3, rel=0.01)
    assert 0.9 < L["efficiency"] < 1.0
    bad = simulate_netlist(text.replace("out=R1", "out=Rnope"))
    assert bad["ok"] and "losses" not in bad and any(".losses" in w for w in bad["warnings"])
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".net", delete=False, encoding="utf-8") as fh:
        fh.write(text)
    try:
        assert main(["run", fh.name]) == 0
    finally:
        os.unlink(fh.name)
    out = capsys.readouterr().out
    assert "Loss report" in out and "efficiency" in out and "S1" in out


# ============================================================== step response metrics
def test_step_info_second_order_matches_theory():
    r = pe.load_netlist(os.path.join(ROOT, "circuits", "rlc_step.net")).run()
    info = r.step_info("V(b)")
    zeta = 5 / 2 * math.sqrt(10e-6 / 1e-3)
    wd = math.sqrt(1 / (1e-3 * 10e-6)) * math.sqrt(1 - zeta ** 2)
    assert info["overshoot_pct"] == pytest.approx(100 * math.exp(-math.pi * zeta / math.sqrt(1 - zeta ** 2)), abs=0.01)
    assert info["peak_time"] == pytest.approx(math.pi / wd, rel=2e-3)
    assert info["peak"] == pytest.approx(info["final"] * (1 + info["overshoot_pct"] / 100), rel=1e-9)
    assert info["initial"] == pytest.approx(0, abs=1e-9) and info["final"] == pytest.approx(10, rel=1e-4)
    assert 0 < info["delay_time"] < info["peak_time"] and info["rise_time"] < info["peak_time"]
    assert 0 < info["settling_time"] < 3e-3


def test_step_info_first_order_rise_and_settling_and_falling_step():
    c = pe.Circuit()
    c.V("V1", "in", "0", pe.Step(1e-3, 5.0, 1.0))                   # 5 V -> 1 V at 1 ms
    c.R("R1", "in", "o", 1e3)
    c.C("C1", "o", "0", 1e-6)
    r = c.simulate(1e-5, 12e-3)
    info = r.step_info("V(o)", t_step=1e-3)
    assert info["initial"] == pytest.approx(5 * (1 - math.exp(-1)), abs=0.03)     # value just before the step (within one 10 us sample)
    assert info["final"] == pytest.approx(1.0, abs=1e-3)
    assert info["rise_time"] == pytest.approx(math.log(9) * 1e-3, rel=5e-3)      # 10-90 %: tau * ln 9
    assert info["settling_time"] == pytest.approx(math.log(50) * 1e-3, rel=5e-3)  # 2 %: tau * ln 50
    assert info["overshoot_pct"] < 0.01                                          # first order: no overshoot (integration noise only)
    with pytest.raises(ValueError, match="does not change"):
        r.step_info("V(in)", t_step=2e-3)
    with pytest.raises(ValueError, match="end of the record"):
        r.step_info("V(o)", t_step=12e-3)


def test_step_info_reports_nan_when_it_never_settles():
    t = np.linspace(0, 1, 2001)
    y = 1 - np.exp(-3 * t) + 0.3 * np.sin(2 * np.pi * 40 * t) * (t > 0.2)    # never within 2 % at the end
    res = pe.Results(t, {"y": y}, {"dt": t[1] - t[0]})
    assert math.isnan(res.step_info("y", final=1.0)["settling_time"])


def test_measure_kinds_and_errors():
    r = pe.load_netlist(os.path.join(ROOT, "circuits", "rlc_step.net")).run()
    assert measure(r, "overshoot V(b)") == pytest.approx(44.43, abs=0.02)
    assert measure(r, ["max", "V(b)"]) == pytest.approx(14.443, abs=0.01)
    assert measure(r, "final V(b)") == pytest.approx(10, rel=1e-3)
    assert measure(r, "peaktime V(b) 0") == pytest.approx(3.245e-4, rel=5e-3)
    assert measure(r, "power R1") > 0
    assert 0 < measure(r, "rms V(b)") < 15
    with pytest.raises(ValueError, match="unknown measurement"):
        measure(r, "median V(b)")
    with pytest.raises(ValueError, match="needs a signal"):
        measure(r, "max")
    with pytest.raises(ValueError, match="fundamental"):
        measure(r, "thd V(b)")
    with pytest.raises(ValueError, match="eff needs"):
        measure(r, "eff R1")


# ============================================================== target solving
def _rlc_text():
    with open(os.path.join(ROOT, "circuits", "rlc_step.net"), encoding="utf-8") as fh:
        return fh.read()


def test_solve_for_overshoot_matches_the_analytic_damping_resistor():
    sol = solve_for(_rlc_text(), "R1.value", "overshoot V(b)", 20, 2, 50)
    zeta = math.sqrt(math.log(5) ** 2 / (math.pi ** 2 + math.log(5) ** 2))      # 20 % overshoot
    assert sol.converged and sol.value == pytest.approx(zeta * 2 / math.sqrt(10e-6 / 1e-3), rel=5e-3)
    assert sol.measure == pytest.approx(20, rel=1e-3) and sol.iterations <= 14
    assert "R1 in a 9.1" in sol.text                                          # the tuned netlist
    assert pe.parse_netlist(sol.text).run().step_info("V(b)")["overshoot_pct"] == pytest.approx(20, rel=2e-3)
    assert "R1.value" in str(sol) and "runs" in str(sol)


def test_solve_for_log_range_and_errors():
    sol = solve_for(_rlc_text(), "R1.value", "max V(b)", 12, 0.5, 500, log=True)
    assert sol.converged and sol.measure == pytest.approx(12, rel=1e-3)
    with pytest.raises(ValueError, match="not between the measurements"):
        solve_for(_rlc_text(), "R1.value", "max V(b)", 12, 2, 3)
    with pytest.raises(ValueError, match="hi must be greater"):
        solve_for(_rlc_text(), "R1.value", "max V(b)", 12, 5, 2)
    with pytest.raises(ValueError, match="positive limits"):
        solve_for(_rlc_text(), "R1.value", "max V(b)", 12, 0, 5, log=True)
    short = solve_for(_rlc_text(), "R1.value", "overshoot V(b)", 20, 2, 50, max_iter=1)
    assert not short.converged and "NOT converged" in str(short)
    with pytest.raises(ValueError, match="did you mean"):
        solve_for(_rlc_text(), "R1.valu", "max V(b)", 12, 2, 50)


def test_cli_solve(capsys, tmp_path):
    net = os.path.join(ROOT, "circuits", "rlc_step.net")
    out_file = tmp_path / "tuned.net"
    assert main(["run", net, "--solve", "R1.value=2:50", "--target", "overshoot V(b)=20", "--save", str(out_file)]) == 0
    out = capsys.readouterr().out
    assert "R1.value = 9.1" in out and "run  1" in out and "tuned netlist written" in out
    assert "R1 in a 9.1" in out_file.read_text()
    assert main(["run", net, "--solve", "R1.value=2:50"]) == 2                      # --target missing
    assert main(["run", net, "--solve", "R1.value=2:3", "--target", "overshoot V(b)=20"]) == 2
    assert "not between" in capsys.readouterr().err


# ============================================================== sub-circuits
def test_subckt_expands_like_the_flat_circuit():
    flat = """V1 in 0 DC 10
R1 in mid 1k
R2 mid out 2k
C1 out 0 1u
R3 out 0 4k
.tran 10u 5m
"""
    sub = """.subckt rc in out gnd r=1k c=1u
Ra in out {r}
Cb out gnd {c}
.ends
V1 in 0 DC 10
R1 in mid 1k
X1 mid out 0 rc r=2k c=1u
R3 out 0 4k
.tran 10u 5m
"""
    a, b = pe.parse_netlist(flat).run(), pe.parse_netlist(sub).run()
    assert [e.name for e in pe.parse_netlist(sub).circuit.elements] == ["V1", "R1", "X1.Ra", "X1.Cb", "R3"]
    assert np.allclose(a["V(out)"], b["V(out)"], atol=1e-9)


def test_subckt_parameters_independent_instances_and_nesting():
    text = """.subckt half a b r=1k
R1 a b {r}
.ends
.subckt pair in out r=1k
X1 in mid half r={r}
X2 mid out half r={r}
.ends
V1 in 0 DC 12
Xp in out pair r=3k
Xq in out2 pair
Rl out 0 6k
Rm out2 0 1k
.op
"""
    nl = pe.parse_netlist(text)
    names = [e.name for e in nl.circuit.elements]
    assert "Xp.X1.R1" in names and "Xq.X2.R1" in names                       # nested prefixes
    assert nl.circuit.element("Xp.X1.R1").r == 3000 and nl.circuit.element("Xq.X1.R1").r == 1000
    op = nl.op()
    assert op["V(out)"] == pytest.approx(12 * 6 / 12)                         # 6k + 6k divider
    assert op["V(out2)"] == pytest.approx(12 * 1 / 3)                         # 2k series, 1k load


def test_subckt_with_control_blocks_and_global_params():
    text = """.param fsw=10k
.subckt leg vin out duty
S1 vin out g ron=10m
D1 0 out
B1 PWM g duty freq={fsw}
.ends
V1 in 0 DC 100
B1 CONST d value=0.3
X1 in sw d leg
L1 sw o 1m
R1 o 0 5
C1 o 0 100u
.tran 0.5u 8m
"""
    r = pe.parse_netlist(text).run().last(1e-3)
    assert r.mean("V(o)") == pytest.approx(30, rel=0.03)


def test_subckt_errors():
    with pytest.raises(ValueError, match="no matching .ends"):
        pe.parse_netlist(".subckt s a\nR1 a 0 1\n")
    with pytest.raises(ValueError, match="1 port.*2 node"):
        pe.parse_netlist(".subckt s a\nR1 a 0 1\n.ends\nX1 n1 n2 s\n")
    with pytest.raises(ValueError, match="no parameter 'bogus'"):
        pe.parse_netlist(".subckt s a k=1\nR1 a 0 {k}\n.ends\nX1 n s bogus=3\n")
    with pytest.raises(ValueError, match="sub-circuit"):
        pe.parse_netlist("X1 a b c nosuch\n")
    with pytest.raises(ValueError, match="needs a name"):
        pe.parse_netlist(".subckt\n.ends\n")


def test_interleaved_buck_ripple_cancels_and_reports_losses():
    path = os.path.join(ROOT, "circuits", "interleaved_buck.net")
    text = open(path, encoding="utf-8").read()
    two = pe.parse_netlist(text).run()
    one_text = text.replace("X2 in out d phase ph=180\n", "").replace("Rload out 0 2.4", "Rload out 0 4.8")
    one = pe.parse_netlist(one_text).run()
    w2, w1 = two.window(3e-3, 4e-3), one.window(3e-3, 4e-3)
    assert w2.mean("V(out)") == pytest.approx(12, rel=0.04)
    sum_ripple = (w2["I(X1.L1)"] + w2["I(X2.L1)"]).max() - (w2["I(X1.L1)"] + w2["I(X2.L1)"]).min()
    assert sum_ripple < 0.1 * w1.ripple("I(X1.L1)")                           # ripple cancels at D = 0.5
    assert w2.ripple("V(out)") < 0.2 * w1.ripple("V(out)")
    rep = w2.loss_report(output="Rload", input="Vin")
    assert {r.name for r in rep.rows} >= {"X1.S1", "X2.S1", "X1.D1", "X2.D1"}
    assert rep.rows[0].switching > 0 and 0.85 < rep.efficiency < 1.0


# ============================================================== saturable inductor
def _sat_circuit(**kw):
    c = pe.Circuit()
    c.V("V1", "a", "0", 10)
    c.R("Rs", "a", "b", 1e-6)
    c.add(pe.SaturableInductor("L1", "b", "0", l0=1e-3, isat=2.0, **kw))
    return c


def test_saturable_inductor_conserves_flux_and_saturates():
    r = _sat_circuit(lsat=0.1e-3).simulate(1e-6, 1e-3)
    t, i, lam = r.t, r["I(L1)"], r["flux(L1)"]
    assert np.max(np.abs(lam - 10 * t)[5:] / (10 * t[5:])) < 1e-4                 # lambda = integral of v dt
    k_lo, k_hi = np.searchsorted(t, 0.05e-3), np.searchsorted(t, 0.9e-3)
    assert r["L(L1)"][k_lo] > 0.9e-3 and r["L(L1)"][k_hi] == pytest.approx(0.1e-3, rel=1e-3)
    lin = 10 * t[k_hi] / 1e-3
    assert i[k_hi] > 5 * lin                                                      # current runs away after saturation
    # the slope di/dt equals V / L(i) at each moment
    k = np.searchsorted(t, 0.3e-3)
    assert (i[k + 1] - i[k - 1]) / (t[k + 1] - t[k - 1]) == pytest.approx(10 / r["L(L1)"][k], rel=1e-3)


def test_saturable_inductor_small_signal_ac_dc_and_initial_current():
    L0, LS, IS, IB = 1e-3, 0.1e-3, 2.0, 3.0
    c = pe.Circuit()
    c.V("Vb", "a", "0", 0.0, ac=1.0)
    c.I("Ib", "0", "b", IB)                                           # 3 A bias through the inductor
    c.add(pe.SaturableInductor("L1", "a", "b", l0=L0, isat=IS, lsat=LS))
    c.R("Rl", "b", "0", 1e-3)
    ac = c.ac([1e3])
    l_ac = abs(ac["V(a)"][0] / ac["I(L1)"][0]) / (2 * math.pi * 1e3)
    assert l_ac == pytest.approx(LS + (L0 - LS) / math.cosh(IB / IS) ** 2, rel=1e-4)
    d = pe.Circuit()
    d.V("V1", "a", "0", 5)
    d.R("R1", "a", "b", 5)
    d.add(pe.SaturableInductor("L1", "b", "0", l0=L0, isat=IS))
    assert d.op()["I(L1)"] == pytest.approx(1.0, rel=1e-6)           # DC: a short circuit
    e = pe.Circuit()
    e.R("R1", "a", "0", 1e6)
    e.add(pe.SaturableInductor("L1", "a", "0", l0=L0, isat=IS, ic=1.5))
    assert e.simulate(1e-6, 1e-5)["I(L1)"][0] == pytest.approx(1.5, rel=1e-3)


def test_saturable_inductor_netlist_and_validation():
    nl = pe.parse_netlist("V1 a 0 DC 1\nL1 a b 1m isat=2 lsat=100u ic=0.5\nR1 b 0 1\n.op\n")
    el = nl.circuit.element("L1")
    assert type(el).__name__ == "SaturableInductor"
    assert (el.l0, el.isat, el.ic) == (1e-3, 2.0, 0.5) and el.lsat == pytest.approx(1e-4)
    with pytest.raises(ValueError, match="lsat must satisfy"):
        pe.SaturableInductor("L1", "a", "b", l0=1e-3, isat=1, lsat=2e-3)
    with pytest.raises(ValueError, match="must be > 0"):
        pe.SaturableInductor("L1", "a", "b", l0=0, isat=1)


# ============================================================== lookup table, SVPWM
def test_lookup_table_block_and_netlist():
    lt = pe.LookupTable("y", "x", [0, 1, 3], [0, 10, 0])
    assert [lt.step([x], None) for x in (-1, 0.5, 2, 5)] == [0.0, 5.0, 5.0, 0.0]            # clamped outside
    ex = pe.LookupTable("y", "x", [0, 1], [0, 2], extrapolate=True)
    assert ex.step([3], None) == 6.0 and ex.step([-1], None) == -2.0
    with pytest.raises(ValueError, match="strictly increasing"):
        pe.LookupTable("y", "x", [0, 0], [1, 2])
    with pytest.raises(ValueError, match="same length"):
        pe.LookupTable("y", "x", [0, 1], [1])
    nl = pe.parse_netlist("""V1 a 0 DC 1
R1 a 0 1
B1 RAMP x slope=1000
B2 LOOKUP y x xs=0,2,4 ys=0,5,1
.tran 10u 4m
.plot y
""")
    r = nl.run()
    assert r["y"][np.searchsorted(r.t, 1e-3)] == pytest.approx(2.5, abs=0.05)
    assert r.final("y") == pytest.approx(1.0, abs=0.01)


def test_svpwm_extends_the_linear_range_and_reproduces_line_voltages():
    blk = pe.SVPWM("da", "db", "dc", "va", "vb", "vc", vdc=400)
    amp = 400 / math.sqrt(3) * 0.999                                 # sine PWM would clip above vdc/2 = 200 V
    for a in np.linspace(0, 2 * math.pi, 241):
        u = [amp * math.cos(a), amp * math.cos(a - 2 * math.pi / 3), amp * math.cos(a + 2 * math.pi / 3)]
        d = blk.step(u, None)
        assert 0 <= min(d) and max(d) <= 1
        assert 400 * (d[0] - d[1]) == pytest.approx(u[0] - u[1], abs=1e-9)
        assert 400 * (d[1] - d[2]) == pytest.approx(u[1] - u[2], abs=1e-9)
    over = blk.step([300.0, -150.0, -150.0], None)                    # beyond vdc/sqrt(3): duties saturate at 0..1
    assert max(over) == 1.0 and min(over) == 0.0
    with pytest.raises(ValueError, match="vdc"):
        pe.SVPWM("a", "b", "c", "x", "y", "z", vdc=0)
    nl = pe.parse_netlist("V1 a 0 DC 1\nR1 a 0 1\nB1 CONST va value=100\nB2 CONST vb value=-50\nB3 CONST vc value=-50\n"
                          "B4 SVPWM da db dc va vb vc vdc=400\n.tran 10u 100u\n.plot da db dc\n")
    r = nl.run()
    # common-mode term (max+min)/2 = 25 V: duty = 1/2 + (v - 25) / 400
    assert r.final("da") == pytest.approx(0.6875) and r.final("db") == pytest.approx(0.3125)


def test_three_phase_inverter_with_svpwm_reaches_above_the_sine_pwm_limit():
    VDC, F0, FSW, R, L = 400.0, 50.0, 10e3, 10.0, 5e-3
    vref = 0.55 * VDC                                                  # 220 V peak: above the 200 V sine-PWM limit
    c = pe.Circuit("SVPWM inverter")
    c.V("Vdc", "p", "0", VDC)
    for ph, nm in enumerate("abc"):
        c.S(f"Sh{nm}", "p", nm, f"gh{nm}", ron=10e-3, body_diode=True)
        c.S(f"Sl{nm}", nm, "0", f"gl{nm}", ron=10e-3, body_diode=True)
        c.R(f"R{nm}", nm, f"x{nm}", R)
        c.L(f"L{nm}", f"x{nm}", "n", L)
        c.add(pe.SineWave(f"v{nm}", vref, F0, -120.0 * ph),
              pe.PWM(f"gh{nm}", f"d{nm}", FSW, vmin=0.0, vmax=1.0, out_comp=f"gl{nm}"))
    c.add(pe.SVPWM("da", "db", "dc", "va", "vb", "vc", vdc=VDC))
    r = c.simulate(0.5e-6, 80e-3, t_record_start=20e-3)
    w = r.last(2 / F0)
    # phase-to-neutral load voltage: fundamental equals the reference (the common-mode injection cancels)
    amp, _ = w.harmonic("V(a,n)", F0, 1, cycles=2)
    assert amp == pytest.approx(vref, rel=0.03)
    assert w.thd("I(La)", F0, cycles=2) < 0.03
    i_amp, _ = w.harmonic("I(La)", F0, 1, cycles=2)
    assert i_amp == pytest.approx(vref / math.hypot(R, 2 * math.pi * F0 * L), rel=0.03)
