"""Value changing: run-time changes (.change / Circuit.at), E-series standard
values, command-line overrides and sweeps."""
import os

import pytest

import openpe as pe
from openpe import eseries
from openpe.__main__ import main
from openpe.designparams import apply_overrides, extract_params, snap_to_series
from openpe.server import simulate_netlist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- E-series
def test_eseries_tables():
    assert len(eseries.E6) == 6 and len(eseries.E12) == 12 and len(eseries.E24) == 24
    assert len(eseries.E48) == 48 and len(eseries.E96) == 96
    # spot-checks against the published tables
    assert eseries.E96[:8] == [1.00, 1.02, 1.05, 1.07, 1.10, 1.13, 1.15, 1.18]
    assert eseries.E96[-3:] == [9.31, 9.53, 9.76]
    assert eseries.E48[23] == 3.01 or eseries.E48[24] == 3.16
    assert eseries.E24[13] == 3.6 and eseries.E12[7] == 3.9


def test_nearest_and_decades():
    assert eseries.nearest(4.6e3, "E24") == 4.7e3
    assert eseries.nearest(4.6e3, "E12") == 4.7e3
    assert eseries.nearest(1.04e-6, "E24") == pytest.approx(1.0e-6)
    assert eseries.nearest(9.8e3, "E24") == pytest.approx(10e3)          # rounds up into the next decade
    assert eseries.nearest(95e3, "E96") == pytest.approx(95.3e3)
    assert eseries.nearest(1e-9, "E24") == pytest.approx(1e-9)
    assert eseries.error_percent(4.6e3, "E24") == pytest.approx(100 * 100 / 4600, rel=1e-6)
    assert eseries.is_standard(4.7e3, "E24") and not eseries.is_standard(4.6e3, "E24")
    for bad in (0, -5, float("inf")):
        with pytest.raises(ValueError):
            eseries.nearest(bad)
    with pytest.raises(ValueError, match="unknown series"):
        eseries.nearest(1.0, "E7")


def test_step_through_the_series():
    assert eseries.step(4.7e3, +1, "E24") == 5.1e3
    assert eseries.step(4.7e3, -1, "E24") == 4.3e3
    assert eseries.step(4.6e3, +1, "E24") == 4.7e3                     # non-standard: nearest in that direction
    assert eseries.step(4.6e3, -1, "E24") == 4.3e3
    assert eseries.step(9.1e3, +1, "E24") == 10e3                       # crosses a decade
    assert eseries.step(1.0e3, -1, "E24") == 910.0
    assert eseries.step(1e-6, -1, "E12") == pytest.approx(0.82e-6)
    v = 1e3
    for _ in range(12):                                                 # a full decade of E12 returns to 10x
        v = eseries.step(v, +1, "E12")
    assert v == pytest.approx(1e4)


# ---------------------------------------------------------------- run-time changes
def _divider():
    c = pe.Circuit()
    c.V("V1", "in", "0", 10)
    c.R("R1", "in", "out", 1e3)
    c.R("R2", "out", "0", 1e3)
    return c


def test_resistor_change_takes_effect_exactly_at_the_scheduled_time():
    c = _divider()
    c.at(5e-3, "R2", 3e3)
    r = c.simulate(1e-4, 10e-3)
    v = r["V(out)"]
    assert v[50] == pytest.approx(5.0, abs=1e-6)          # t = 5.0 ms: still the old value
    assert v[51] == pytest.approx(7.5, abs=1e-6)          # first step starting at 5 ms uses the new value
    assert v[-1] == pytest.approx(7.5, abs=1e-6)
    assert r.meta["stats"]["value_changes"] == 1
    assert r.meta["changes"] == [{"t": 0.005, "target": "R2", "attr": None, "value": 3000.0}]


def test_changes_do_not_alter_the_original_circuit():
    c = _divider()
    c.at(1e-3, "R2", 9e3)
    c.simulate(1e-4, 3e-3)
    assert c.get_value("R2") == 1000.0


def test_source_change_and_waveform_attributes():
    c = pe.Circuit()
    c.V("V1", "a", "0", 1.0)
    c.R("R1", "a", "0", 1e3)
    c.at(2e-3, "V1", 3.0)
    r = c.simulate(1e-4, 4e-3)
    assert r["V(a)"][5] == pytest.approx(1.0) and r["V(a)"][-1] == pytest.approx(3.0)
    s = pe.Circuit()
    s.V("V1", "a", "0", pe.Sine(1.0, 1e3))
    s.R("R1", "a", "0", 1e3)
    s.at(2e-3, "V1", freq=2e3)                              # change a parameter of the waveform
    r = s.simulate(2e-6, 4e-3)
    late = r.window(3e-3, 4e-3)
    assert late.harmonic("V(a)", 2e3, h=1, cycles=2)[0] == pytest.approx(1.0, rel=1e-2)


def test_capacitor_inductor_changes():
    c = pe.Circuit()
    c.V("V1", "in", "0", pe.Square(0, 1, 100))
    c.R("R1", "in", "out", 1e3)
    c.C("C1", "out", "0", 1e-6)
    c.at(0.05, "C1", 100e-6)                                # 100x slower filter
    r = c.simulate(1e-5, 0.1)
    early, late = r.window(0.02, 0.05), r.window(0.07, 0.1)
    assert early.ripple("V(out)") > 5 * late.ripple("V(out)")      # 0.99 V -> 0.11 V (late window still charging)
    L = pe.Circuit()
    L.V("V1", "a", "0", 5)
    L.R("R1", "a", "b", 5)
    L.L("L1", "b", "0", 1e-3)
    L.at(5e-3, "L1", 2e-3)
    assert L.simulate(1e-5, 20e-3).final("I(L1)") == pytest.approx(1.0, rel=1e-3)


def test_control_block_change_moves_the_operating_point():
    c = pe.Circuit()
    c.V("Vin", "in", "0", 48)
    c.S("S1", "in", "sw", "g")
    c.D("D1", "0", "sw")
    c.L("L1", "sw", "out", 200e-6)
    c.C("C1", "out", "0", 100e-6)
    c.R("RL", "out", "0", 4)
    c.add(pe.Constant("d", 0.25), pe.PWM("g", "d", 20e3))
    c.at(10e-3, "d", 0.5)                                   # control block addressed by its output signal
    r = c.simulate(0.3e-6, 22e-3)
    assert r.window(9e-3, 10e-3).mean("V(out)") == pytest.approx(12, rel=0.02)
    assert r.window(21e-3, 22e-3).mean("V(out)") == pytest.approx(24, rel=0.02)


def test_dc_motor_load_torque_change():
    V, Ra, ke, bf = 100.0, 1.0, 0.5, 1e-3
    c = pe.Circuit()
    c.V("Va", "a", "0", V)
    c.add(pe.DCMotor("M1", "a", "0", ra=Ra, la=5e-3, ke=ke, j=0.01, bf=bf, t_load=2.0))
    c.at(0.3, "M1", t_load=4.0)
    r = c.simulate(50e-6, 0.6, record_every=20)
    w = lambda tl: (ke * V - Ra * tl) / (ke ** 2 + Ra * bf)   # noqa: E731
    assert r.window(0.25, 0.3).mean("speed(M1)") == pytest.approx(w(2.0), rel=2e-3)
    assert r.final("speed(M1)") == pytest.approx(w(4.0), rel=2e-3)


def test_change_at_time_zero_acts_on_the_initial_state():
    c = _divider()
    c.at(0.0, "R2", 3e3)
    r = c.simulate(1e-4, 1e-3)
    assert r["V(out)"][0] == pytest.approx(7.5, abs=1e-6) and r.final("V(out)") == pytest.approx(7.5, abs=1e-6)


def test_multiple_changes_run_in_time_order():
    c = _divider()
    c.at(6e-3, "R2", 1e3)
    c.at(2e-3, "R2", 3e3)
    r = c.simulate(1e-4, 8e-3)
    assert r["V(out)"][30] == pytest.approx(7.5, abs=1e-6) and r.final("V(out)") == pytest.approx(5.0, abs=1e-6)
    assert r.meta["stats"]["value_changes"] == 2


def test_change_errors_are_reported_clearly():
    c = _divider()
    with pytest.raises(ValueError, match="unknown target 'R9'"):
        c.at(1e-3, "R9", 5)
    with pytest.raises(ValueError, match="give a value"):
        c.at(1e-3, "R2")
    c.at(1e-3, "R2", -5)
    with pytest.raises(ValueError, match="must be > 0"):
        c.simulate(1e-4, 2e-3)
    c2 = _divider()
    c2.at(1e-3, "R2", nonsense=5)
    with pytest.raises(ValueError, match="no parameter 'nonsense'"):
        c2.simulate(1e-4, 2e-3)
    c3 = _divider()
    c3.add(pe.Constant("g", 1), pe.Switch("S1", "in", "out", "g"))
    c3.at(1e-3, "S1", 5)                                      # a switch has no default value
    with pytest.raises(ValueError, match="give the parameter to change.*ron"):
        c3.simulate(1e-4, 2e-3)


def test_set_value_and_get_value():
    c = _divider()
    c.set_value("R2", 3e3)
    assert c.get_value("R2") == 3000.0
    assert c.op()["V(out)"] == pytest.approx(7.5)
    c.add(pe.Constant("duty", 0.2), pe.PI("u", "duty", kp=1, ki=2))
    c.set_value("u", kp=5.0, ki=7.0)
    assert (c.get_value("u", "kp"), c.get_value("u", "ki")) == (5.0, 7.0)
    assert c.get_value("V1") == 10.0


# ---------------------------------------------------------------- netlist
NET = """* load step
V1 in 0 DC 10
R1 in out 1k
R2 out 0 1k
.change 5m R2 3k
.change 8m V1 20
.tran 100u 10m
.plot V(out)
.end
"""


def test_netlist_change_directive_runs():
    nl = pe.parse_netlist(NET)
    assert [(c.t, c.target, c.value) for c in nl.circuit.changes] == [(0.005, "R2", 3000.0), (0.008, "V1", 20.0)]
    r = nl.run()
    assert r.window(4e-3, 5e-3).mean("V(out)") == pytest.approx(5.0, rel=1e-3)
    assert r.window(6e-3, 8e-3).mean("V(out)") == pytest.approx(7.5, rel=1e-3)
    assert r.final("V(out)") == pytest.approx(15.0, rel=1e-3)


def test_netlist_change_by_block_name_and_key_value():
    nl = pe.parse_netlist("""V1 a 0 DC 1
R1 a 0 1k
B1 CONST k value=0.2
B2 GAIN y k k=10
.change 1m B1 value=0.5
.change 1m B2 k=100
.tran 100u 2m
.plot y
""")
    assert nl.run().final("y") == pytest.approx(50.0)               # 0.5 * 100


def test_netlist_change_errors_and_extraction():
    with pytest.raises(ValueError, match="line 1.*expected"):
        pe.parse_netlist(".change 1m R1\n")
    text = NET
    labels = {p.label: p.value for p in extract_params(text)}
    assert labels[".change1 R2.time"] == "5m" and labels[".change1 R2.value"] == "3k"
    assert labels[".change2 V1.time"] == "8m"
    moved = apply_overrides(text, [".change1 R2.time=7m"])
    assert ".change 7m R2 3k" in moved
    r = pe.parse_netlist(moved).run()
    assert r.window(5.5e-3, 6.5e-3).mean("V(out)") == pytest.approx(5.0, rel=1e-3)   # change moved later


def test_server_reports_changes():
    res = simulate_netlist(NET)
    assert res["ok"] and [c["target"] for c in res["changes"]] == ["R2", "V1"]


# ---------------------------------------------------------------- overrides / snapping / CLI
def test_apply_overrides_variants_and_errors():
    t = "R1 a 0 1k\nR2 a b 2k\nV1 b 0 DC 5\n.tran 1u 1m\n.param gain=3\n"
    out = apply_overrides(t, ["r1.value=2.2k", "gain=4", "V1=9", ".tran.stop time=5m"])
    assert "R1 a 0 2.2k" in out and ".param gain=4" in out and "V1 b 0 DC 9" in out and ".tran 1u 5m" in out
    with pytest.raises(ValueError, match="did you mean: R1.value"):
        apply_overrides(t, ["R1.valu=3"])
    with pytest.raises(ValueError, match="LABEL=VALUE"):
        apply_overrides(t, ["R1.value"])
    with pytest.raises(ValueError):
        apply_overrides(t, ["R1.value=abc"])


def test_snap_to_series():
    text = "R1 a 0 4.6k\nC1 a 0 12.3n\nL1 a b 5.2u\nR2 a b 1.0k\nV1 b 0 DC 5\n"
    new, ch = snap_to_series(text, "E24")
    assert "R1 a 0 4.7k" in new and "C1 a 0 12n" in new
    assert "L1 a b 5.2u" in new                                   # inductors only with kinds containing 'L'
    assert [c[0] for c in ch] == ["R1", "C1"]                     # R2 was already standard
    assert ch[0][3] == pytest.approx(100 * 100 / 4600, rel=1e-6)
    new2, ch2 = snap_to_series(text, "E12", kinds="RCL")
    assert "L1 a b 5.6u" in new2 and len(ch2) == 3
    assert snap_to_series(new, "E24")[1] == []                    # idempotent


def test_cli_set_list_and_sweep(capsys, tmp_path):
    net = os.path.join(ROOT, "circuits", "rlc_step.net")
    assert main(["run", net, "--list-params"]) == 0
    out = capsys.readouterr().out
    assert "R1.value" in out and "5" in out
    assert main(["run", net, "--set", "R1.value=40"]) == 0
    out = capsys.readouterr().out
    assert "override: R1.value=40" in out
    peak = float(out.split("max V(b)")[1].split("=")[1].split()[0])
    assert peak < 10.5                                            # well damped: almost no overshoot
    assert main(["run", net, "--set", "R1.valu=40"]) == 2
    assert "did you mean" in capsys.readouterr().err
    csv_path = tmp_path / "s.csv"
    assert main(["run", net, "--sweep", "R1.value=2:50:4:log", "--csv", str(csv_path)]) == 0
    out = capsys.readouterr().out
    rows = [ln.split() for ln in out.splitlines() if ln.strip() and ln.split()[0][0].isdigit()]
    assert [round(float(r[0]), 3) for r in rows] == pytest.approx([2, 5.848, 17.1, 50], rel=1e-3)
    peaks = [float(r[1]) for r in rows]
    assert peaks == sorted(peaks, reverse=True)
    assert csv_path.read_text().splitlines()[0] == "value,max V(b)"
    assert main(["run", net, "--sweep", "R1.value=bad"]) == 2
    assert main(["run", os.path.join(ROOT, "circuits", "diode_iv.net"), "--sweep", "R1.value=1:2:2"]) == 2


def test_bundled_load_step_example():
    r = pe.load_netlist(os.path.join(ROOT, "circuits", "load_step.net")).run()
    before, after, last = (r.window(a, b).mean("I(L1)") for a, b in ((3.5e-3, 4e-3), (6.5e-3, 7e-3), (9.5e-3, 10e-3)))
    assert after == pytest.approx(2 * before, rel=0.05)           # load doubled -> inductor current doubled
    assert r.window(9.5e-3, 10e-3).mean("V(out)") == pytest.approx(9.0, abs=0.1)
    assert last == pytest.approx(9.0 / 3, rel=0.05)
    assert r.meta["stats"]["value_changes"] == 2
