"""Local web server for the OpenPE schematic editor.

    python -m openpe editor            # opens http://127.0.0.1:8765 in a browser

GET  /              -> the drag-and-drop schematic editor (single HTML file)
GET  /api/ping      -> {"ok": true, "version": ...}
POST /api/simulate  -> body {"netlist": "..."}; runs every analysis in the
                       netlist (.tran/.op/.dc/.ac) and returns JSON results

The server binds to 127.0.0.1 by default and only parses netlist text (no
code execution), so it is safe to run on a workstation.
"""
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

EDITOR_HTML = os.path.join(os.path.dirname(__file__), "editor", "index.html")
MAX_STEPS = 5_000_000
MAX_BODY = 2_000_000


def _decimate(t, series, max_points):
    n = len(t)
    if n <= max_points:
        idx = slice(None)
    else:
        idx = np.linspace(0, n - 1, max_points).round().astype(int)
    return np.asarray(t)[idx], {k: np.asarray(v)[idx] for k, v in series.items()}


def _clean(a):
    """numpy array -> JSON-safe list (NaN/inf -> None)."""
    out = []
    for v in np.asarray(a, float).tolist():
        out.append(v if np.isfinite(v) else None)
    return out


def simulate_netlist(text, max_points=4000):
    """Run all analyses of a netlist and return a JSON-serialisable dict."""
    from . import __version__
    from .netlist import parse

    nl = parse(text)
    plot_names = [n for g in nl.plots for n in g]
    out = {"ok": True, "version": __version__, "title": nl.circuit.title, "plots": plot_names}

    for a in nl.analyses:
        if a[0] == "op":
            op = nl.op()
            out["op"] = {"voltages": op.voltages, "probes": op.probes}
        elif a[0] == "dc":
            r = nl.dc()
            names = [n for n in plot_names if n in r] or [k for k in r.names if k.startswith("V(")][:8]
            out["dc"] = {"source": a[1], "x": _clean(r.t),
                         "signals": {n: _clean(r[n]) for n in names}}
        elif a[0] == "ac":
            r = nl.ac()
            outs = [n for n in plot_names if n.startswith("V(")] or [f"V({n})" for n in list(r._node)[-4:]]
            traces = {}
            for o in outs:
                lo, hi = r.corners(o)
                f_pk, g_pk = r.peak(o)
                traces[o] = {"mag_db": _clean(r.mag_db(o)), "phase_deg": _clean(r.phase_deg(o)),
                             "peak_hz": f_pk, "peak_db": g_pk,
                             "f_low": lo if np.isfinite(lo) else None,
                             "f_high": hi if np.isfinite(hi) else None}
            out["ac"] = {"freq": _clean(r.freq), "traces": traces}

    if nl.tran:
        steps = nl.tran["t_end"] / nl.tran["dt"]
        if steps > MAX_STEPS:
            raise ValueError(f"transient needs {steps:.3g} steps (limit {MAX_STEPS:g}); "
                             f"increase the time step or shorten the simulation")
        res = nl.run()
        names = [n for n in plot_names if n in res] or [n for n in res.names if n.startswith("V(")][:8]
        t, sig = _decimate(res.t, {n: res[n] for n in names}, max_points)
        stats = dict(res.meta["stats"])
        out["tran"] = {"t": _clean(t), "signals": {n: _clean(v) for n, v in sig.items()},
                       "stats": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()}}
        if nl.measures:
            from .__main__ import _measure
            mres = res.last(nl.window) if nl.window else res
            out["measures"] = {" ".join(m): _measure(mres, m) for m in nl.measures}
    if not nl.tran and not nl.analyses:
        raise ValueError("the netlist has no analysis: enable Transient, Operating point, AC or DC sweep")
    return out


class _Handler(BaseHTTPRequestHandler):
    server_version = "OpenPE"

    def log_message(self, fmt, *args):  # keep the console quiet
        if os.environ.get("OPENPE_SERVER_LOG"):
            super().log_message(fmt, *args)

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            with open(EDITOR_HTML, "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif path == "/api/ping":
            from . import __version__
            self._send(200, {"ok": True, "version": __version__})
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/simulate":
            self._send(404, {"ok": False, "error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > MAX_BODY:
                raise ValueError("request too large")
            req = json.loads(self.rfile.read(n) or b"{}")
            result = simulate_netlist(str(req.get("netlist", "")))
            self._send(200, result)
        except Exception as exc:  # report any simulation error to the editor
            self._send(200, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def make_server(host="127.0.0.1", port=8765):
    return ThreadingHTTPServer((host, port), _Handler)


def serve(host="127.0.0.1", port=8765, open_browser=True):
    httpd = make_server(host, port)
    url = f"http://{host}:{httpd.server_address[1]}/"
    print(f"OpenPE schematic editor running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    # Allow "Run" on this file directly (e.g. from PyCharm).
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from openpe.server import serve as _serve
    _serve()
