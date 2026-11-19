"""Design-time parameters and the fixes found during the robustness audit."""
import glob
import os
import subprocess
import sys

import numpy as np
import pytest

import openpe as pe
from openpe.designparams import extract_params, find, format_value, set_param, sweep_values
from openpe.server import _decimate, simulate_netlist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NET = """* tunable buck
.param vin=48
Vin in 0 DC {vin}
V2 a 0 DC 1 AC 0.5 SIN(0 2 50 30)
S1 in sw g ron=10m
D1 0 sw vf=0.7
L1 sw out 100u ic=0.5
C1 out 0 220u
R1 out 0 6 ; load
T1 p 0 s 0 2 lm=10m
B1 CONST d value=0.25
B2 PWM g d freq=50k
.tran 0.2u 2m start=1m
.ac dec 10 10 100k
.end
"""


def test_extract_finds_every_value():
    labels = {p.label: p.value for p in extract_params(NET)}
    assert labels["vin"] == "48"
    assert "Vin.dc" not in labels                        # {vin} refers to the .param
    assert labels["V2.dc"] == "1" and labels["V2.ac"] == "0.5"
    assert labels["V2.SIN amp"] == "2" and labels["V2.SIN freq"] == "50" and labels["V2.SIN phase"] == "30"
    assert labels["S1.ron"] == "10m" and labels["D1.vf"] == "0.7"
    assert labels["L1.value"] == "100u" and labels["L1.ic"] == "0.5"
    assert labels["R1.value"] == "6"                     # comment after ';' ignored
    assert labels["T1.turns ratio"] == "2" and labels["T1.lm"] == "10m"
    assert labels["B1.value"] == "0.25" and labels["B2.freq"] == "50k"
    assert labels[".tran.time step"] == "0.2u" and labels[".tran.stop time"] == "2m"
    assert labels[".tran.start"] == "1m"
    assert labels[".ac.stop freq"] == "100k"


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(ROOT, "circuits", "*.net"))))
def test_every_bundled_netlist_round_trips(path):
    text = open(path, encoding="utf-8").read()
    params = extract_params(text)
    assert params, "no adjustable values found"
    for p in params:                                      # rewriting any value keeps the netlist valid
        new = set_param(text, p, p.value)
        assert new == text
    pe.parse_netlist(set_param(text, params[-1], format_value(params[-1].number * 1.5 or 1)))


def test_set_param_changes_the_circuit():
    p = find(extract_params(NET), "R1.value")
    text = set_param(NET, p, "4.7k")
    assert "R1 out 0 4.7k ; load" in text
    assert pe.parse_netlist(text).circuit.element("R1").r == pytest.approx(4700)
    text = set_param(text, find(extract_params(text), "vin"), 24)
    assert ".param vin=24" in text
    text = set_param(text, find(extract_params(text), "V2.SIN freq"), "1k")
    assert pe.parse_netlist(text).circuit.element("V2").wave.freq == pytest.approx(1000)
    with pytest.raises(ValueError):
        set_param(text, find(extract_params(text), "R1.value"), "abc")
    stale = find(extract_params(NET), "R1.value")
    with pytest.raises(ValueError, match="changed"):
        set_param(set_param(NET, stale, "1"), stale, "2")


def test_format_and_sweep_values():
    assert format_value(2200) == "2.2k" and format_value(1e6) == "1meg" and format_value(4.7e-6) == "4.7u"
    assert format_value(0) == "0" and format_value(-0.25) == "-250m"
    assert pe.parse_value(format_value(123.456e-9)) == pytest.approx(123.5e-9, rel=1e-3)
    assert sweep_values("1k", "3k", 3) == pytest.approx([1000, 2000, 3000])
    assert sweep_values(1, 100, 3, log=True) == pytest.approx([1, 10, 100])
    with pytest.raises(ValueError):
        sweep_values(0, 10, 3, log=True)


# ------------------------------------------------------------- audit fixes
def test_saving_a_plot_does_not_switch_the_matplotlib_backend(tmp_path):
    code = (
        "import matplotlib; matplotlib.use('svg')\n"
        "import openpe as pe, numpy as np\n"
        "c = pe.Circuit(); c.V('V1','a','0',1); c.R('R1','a','0',1)\n"
        "r = c.simulate(1e-3, 1e-2)\n"
        f"r.plot('V(a)', save=r'{tmp_path / 'a.png'}')\n"
        f"r.plot_spectrum('V(a)', 100, save=r'{tmp_path / 'b.png'}')\n"
        f"pe.plot_bode(pe.tf_response([1], [1e-3, 1], np.logspace(1, 4, 5)), save=r'{tmp_path / 'c.png'}')\n"
        "print(matplotlib.get_backend())\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "svg"                   # was forced to 'Agg' before the fix
    assert all((tmp_path / f).stat().st_size > 1000 for f in ("a.png", "b.png", "c.png"))


def test_failing_measure_keeps_the_results():
    res = simulate_netlist("V1 a 0 SIN(0 1 50)\nR1 a 0 1\n.tran 1m 10m\n"
                           ".measure thd V(a) 50 2\n.measure max V(a)\n.end\n")
    assert res["ok"] and "tran" in res
    assert res["measures"]["thd V(a) 50 2"] is None
    assert res["measures"]["max V(a)"] == pytest.approx(1.0, rel=1e-3)
    assert any("thd" in w for w in res["warnings"])


def test_unknown_plot_signal_is_reported():
    res = simulate_netlist("V1 a 0 DC 1\nR1 a 0 1\n.tran 1m 10m\n.plot V(aa) V(a)\n.end\n")
    assert list(res["tran"]["signals"]) == ["V(a)"]
    assert any("V(aa)" in w and "Available" in w for w in res["warnings"])


def test_spice_milli_trap_warns():
    assert pe.parse_netlist("R1 a 0 1M\n").warnings
    assert not pe.parse_netlist("R1 a 0 1meg\nR2 a 0 1Meg\nC1 a 0 10m\n").warnings


def test_decimation_preserves_peaks():
    t = np.arange(100000) * 1e-6
    y = np.sin(2 * np.pi * 50 * t) + 0.3 * np.sign(np.sin(2 * np.pi * 7777 * t))   # fast ripple
    td, d = _decimate(t, {"y": y}, 500)
    assert len(td) == len(d["y"]) <= 500
    assert d["y"].max() == pytest.approx(y.max()) and d["y"].min() == pytest.approx(y.min())
    assert np.all(np.diff(td) >= 0)
