"""Tests for the schematic-editor server API and (when Node.js + Playwright
are available) an end-to-end browser test of the drag-and-drop editor."""
import json
import os
import shutil
import subprocess
import threading
import urllib.request

import pytest

from openpe.server import make_server, simulate_netlist

HERE = os.path.dirname(__file__)


@pytest.fixture(scope="module")
def server():
    httpd = make_server("127.0.0.1", 0)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/"
    httpd.shutdown()
    httpd.server_close()


def _post(url, payload):
    req = urllib.request.Request(url + "api/simulate", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def test_serves_editor_and_ping(server):
    with urllib.request.urlopen(server, timeout=10) as r:
        html = r.read().decode()
    assert "OpenPE Schematic Editor" in html and "window.OpenPE" in html
    with urllib.request.urlopen(server + "api/ping", timeout=10) as r:
        assert json.loads(r.read())["ok"] is True


def test_simulate_all_analyses(server):
    netlist = """* rc
V1 in 0 DC 1 AC 1 SIN(0 1 1k 0)
R1 in out 1k
C1 out 0 100n
.tran 2u 2m
.op
.ac dec 10 10 100k
.dc V1 0 2 0.5
.plot V(out) I(R1)
.measure max V(out)
.end
"""
    res = _post(server, {"netlist": netlist})
    assert res["ok"], res
    assert set(res["tran"]["signals"]) == {"V(out)", "I(R1)"}
    assert len(res["tran"]["t"]) == len(res["tran"]["signals"]["V(out)"]) <= 4000
    assert res["op"]["voltages"]["out"] == pytest.approx(1.0)
    tr = res["ac"]["traces"]["V(out)"]
    assert tr["f_high"] == pytest.approx(1591.5, rel=0.02)
    assert res["dc"]["x"] == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert res["dc"]["signals"]["V(out)"] == pytest.approx([0, 0.5, 1, 1.5, 2])
    assert "max V(out)" in res["measures"]


def test_simulate_reports_errors(server):
    res = _post(server, {"netlist": "R1 a 0 xyz\n.op\n"})
    assert res["ok"] is False and "line 1" in res["error"]
    res = _post(server, {"netlist": "R1 a 0 1\nV1 a 0 DC 1\n"})
    assert res["ok"] is False and "no analysis" in res["error"]


def test_transient_step_limit():
    with pytest.raises(ValueError, match="steps"):
        simulate_netlist("R1 a 0 1\nV1 a 0 DC 1\n.tran 1n 1\n")


def test_unknown_routes(server):
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(server + "nope", timeout=10)


def _have_playwright():
    if not shutil.which("node"):
        return False
    r = subprocess.run(["node", "-e", "require('playwright')"], cwd=HERE, capture_output=True)
    return r.returncode == 0


@pytest.mark.skipif(not _have_playwright(), reason="Node.js with Playwright not available")
def test_editor_end_to_end(server):
    """Drag parts from the palette, wire pins by dragging, move parts (wires
    follow), undo/redo, copy/paste, run every example through the server."""
    r = subprocess.run(["node", os.path.join(HERE, "e2e", "editor_e2e.js"), server],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "E2E OK" in r.stdout
