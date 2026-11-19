"""Drive OpenPE Studio (the Tkinter desktop GUI) for real.  Skipped when
Tkinter or a display is unavailable (CI runs it under xvfb)."""
import os
import time

import pytest

tk = pytest.importorskip("tkinter")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture()
def app():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    from openpe.gui import Studio
    a = Studio(root, os.path.join(ROOT, "circuits", "rlc_step.net"))
    root.update()
    yield a
    a.dirty = False
    a.quit()


def _pump(app, cond, timeout=60):
    t0 = time.time()
    while not cond():
        app.root.update()
        time.sleep(0.02)
        assert time.time() - t0 < timeout, "GUI did not finish in time"
    app.root.update()


def test_transient_run_fills_waveform_tab(app):
    app.run_netlist()
    _pump(app, lambda: app.result is not None and not app.busy)
    assert "tran" in app.result
    assert app.sig_list.size() == 2                      # V(b), I(L1) from .plot
    assert len(app.fig_w.axes) == 2                      # stacked subplots
    app.overlay.set(True)
    app.plot_waves()
    assert len(app.fig_w.axes) == 1
    assert "max V(b)" in app.measure_var.get()
    assert app.nb.index(app.nb.select()) == 0


def test_op_ac_and_dc_results(app):
    app.load(os.path.join(ROOT, "circuits", "ce_amplifier.net"))
    app.run_netlist()
    _pump(app, lambda: app.result is not None and "op" in app.result and not app.busy)
    groups = app.op_tree.get_children()
    assert len(groups) == 2
    values = {app.op_tree.item(i, "text"): app.op_tree.item(i, "values")[0]
              for g in groups for i in app.op_tree.get_children(g)}
    assert values["V(c)"].startswith("8.99")              # collector bias ~8.997 V
    assert len(app.fig_ac.axes) == 2                      # magnitude + phase
    app.load(os.path.join(ROOT, "circuits", "diode_iv.net"))
    app.result = None
    app.run_netlist()
    _pump(app, lambda: app.result is not None and not app.busy)
    assert "dc" in app.result and len(app.fig_dc.axes) == 1


def test_syntax_error_goes_to_log(app):
    app.set_text("R1 a 0 xyz\n.op\n")
    app.run_netlist()
    _pump(app, lambda: not app.busy)
    assert "line 1" in app.log.get("1.0", "end")


def test_edit_save_and_reload(app, tmp_path):
    p = tmp_path / "x.net"
    app.set_text("V1 a 0 DC 2\nR1 a 0 1k\n.op\n")
    app.text.insert("end", "* edited\n")
    app.root.update()
    assert app.dirty
    app.path = str(p)
    app.save()
    assert not app.dirty and "* edited" in p.read_text()
    app.load(str(p))
    app.run_netlist()
    _pump(app, lambda: app.result is not None and not app.busy)
    assert app.result["op"]["voltages"]["a"] == pytest.approx(2.0)


def test_example_script_runs_and_shows_plots(app):
    app.run_script(os.path.join(ROOT, "examples", "07_digital_control_cblock.py"))
    _pump(app, lambda: app.proc is not None and app.proc.poll() is not None and not app.busy, timeout=120)
    log = app.log.get("1.0", "end")
    assert "07_digital_control_cblock.py: OK" in log, log
    assert app.img_list.size() >= 1 and app.img_list.get(0).endswith(".png")


def _peak(app, sig="V(b)"):
    return max(v for v in app.result["tran"]["signals"][sig] if v is not None)


def test_design_parameter_panel_edits_and_autoruns(app):
    labels = {app.ptree.item(i, "text") for g in app.ptree.get_children() for i in app.ptree.get_children(g)}
    assert {"value", "time step", "stop time"} <= labels
    app.run_netlist()
    _pump(app, lambda: app.result is not None and not app.busy)
    peak_r5 = _peak(app)                                  # R1 = 5 ohm: underdamped, ~44 % overshoot
    app.select_param("R1.value")
    assert app.p_value.get() == "5"
    app.result = None
    app.p_value.set("40")                                 # heavier damping
    assert app.apply_param()
    assert "R1 in a 40" in app.text.get("1.0", "end")
    _pump(app, lambda: app.result is not None and not app.busy)   # auto-run re-simulated
    assert _peak(app) < peak_r5 - 2
    assert "changed" in app.ptree.item("p:R1.value", "tags")
    # slider: nominal x10^0.5, applied on release
    app._scale_moved("0.5")
    assert app.p_value.get() == "15.8"
    # +/-10 % stepping and reset to the original value
    app.p_value.set("10")
    app.step_param(+1)
    assert app.p_value.get() == "11"
    app.reset_param()
    assert "R1 in a 5" in app.text.get("1.0", "end")
    _pump(app, lambda: not app.busy)


def test_parameter_sweep_overlays_runs(app):
    app.run_netlist()
    _pump(app, lambda: app.result is not None and not app.busy)
    app.select_param("R1.value")
    app.sw_from.set("2")
    app.sw_to.set("50")
    app.sw_n.set("4")
    app.sw_log.set(True)
    app.sw_signal.set("V(b)")
    app.run_sweep()
    _pump(app, lambda: app.sweep_data is not None and not app.busy, timeout=120)
    runs = app.sweep_data["runs"]
    assert [r["value"] for r in runs] == pytest.approx([2, 2 * 25 ** (1 / 3), 2 * 25 ** (2 / 3), 50])
    peaks = [r["max"] for r in runs]
    assert peaks == sorted(peaks, reverse=True)          # more damping, less overshoot
    assert len(app.fig_w.axes) == 1 and len(app.fig_w.axes[0].lines) == 4
    assert "sweep of R1.value" in app.log.get("1.0", "end")
    assert "R1 in a 5" in app.text.get("1.0", "end")      # the sweep does not modify the design


def test_script_and_simulation_busy_states_are_independent(app):
    app.run_script(os.path.join(ROOT, "examples", "07_digital_control_cblock.py"))
    assert app.script_busy and not app.sim_busy
    app.result = None
    app.run_netlist()                                     # must not be blocked by the script
    _pump(app, lambda: app.result is not None and not app.sim_busy)
    _pump(app, lambda: not app.script_busy, timeout=120)
