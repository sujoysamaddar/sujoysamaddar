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
