"""Analyses built on top of transient simulation: parameter sweep and
AC sweep (frequency response by sinusoidal injection, which works for
switching circuits as well as linear ones)."""
import itertools
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .control import SineWave
from .simulator import Simulator


def _run_point(args):
    build, point, sim_kw, measure = args
    circuit = build(**point)
    res = Simulator(circuit, **sim_kw).run()
    out = dict(point)
    out.update(measure(res))
    return out


def parameter_sweep(build, params, dt, t_end, measure, workers=1, **sim_kw):
    """Run a simulation for every combination of parameter values.

    build   : function(**point) -> Circuit
    params  : {"name": [values...], ...}  (Cartesian product is swept)
    measure : function(Results) -> {"metric": value, ...}
    workers : >1 runs points in parallel processes (build/measure must be
              top-level picklable functions)
    Returns a list of dicts (parameters + metrics).
    """
    keys = list(params)
    points = [dict(zip(keys, combo)) for combo in itertools.product(*(params[k] for k in keys))]
    sim_kw = dict(sim_kw, dt=dt, t_end=t_end)
    jobs = [(build, p, sim_kw, measure) for p in points]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            return list(ex.map(_run_point, jobs))
    return [_run_point(j) for j in jobs]


def format_table(rows, floatfmt="{:.5g}"):
    if not rows:
        return ""
    cols = list(rows[0].keys())
    cells = [[floatfmt.format(r[c]) if isinstance(r[c], float) else str(r[c]) for c in cols] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    line = "  ".join(c.rjust(w) for c, w in zip(cols, widths))
    sep = "  ".join("-" * w for w in widths)
    body = ["  ".join(v.rjust(w) for v, w in zip(row, widths)) for row in cells]
    return "\n".join([line, sep] + body)


def ac_sweep(build, freqs, measure, perturb="ac_pert", amplitude=0.01, dt=1e-6,
             t_settle=5e-3, cycles=4, reference=None, **sim_kw):
    """Small-signal frequency response by sinusoidal injection.

    build     : function() -> Circuit.  The circuit must contain a block that
                drives the signal ``perturb`` (e.g. Constant('ac_pert', 0));
                it is replaced by a sine of the sweep frequency and summed
                into the loop / source by the user's circuit.
    measure   : signal (or probe) whose response is measured
    reference : signal used as the input of the transfer function; defaults
                to the injected perturbation itself.
    Returns dict with arrays 'freq', 'gain', 'mag_db', 'phase_deg'.
    """
    out = {"freq": [], "gain": [], "mag_db": [], "phase_deg": []}
    for f in freqs:
        c = build()
        blk = c.block(perturb)
        c.blocks.remove(blk)
        c.add(SineWave(perturb, amplitude, f))
        n_per = max(1, int(round(1.0 / (f * dt))))
        t_end = t_settle + cycles * n_per * dt
        t_end = math.ceil(t_end / (n_per * dt)) * n_per * dt
        res = Simulator(c, dt, t_end, t_record_start=t_end - cycles * n_per * dt - dt, **sim_kw).run()
        f_eff = 1.0 / (n_per * dt)
        am, pm = res.harmonic(measure, f_eff, 1, cycles)
        ar, pr = res.harmonic(reference or perturb, f_eff, 1, cycles)
        g = am / ar
        ph = (pm - pr + 180.0) % 360.0 - 180.0
        out["freq"].append(f)
        out["gain"].append(g)
        out["mag_db"].append(20 * math.log10(g) if g > 0 else -math.inf)
        out["phase_deg"].append(ph)
    return {k: np.array(v) for k, v in out.items()}


def plot_bode(resp, title="Frequency response", save=None, show=False, analytic=None):
    """Bode plot of an ac_sweep() result. ``analytic`` may be a dict of the
    same form to overlay (e.g. from a transfer-function model)."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5.5))
    a1.semilogx(resp["freq"], resp["mag_db"], "o-", label="simulated (injection)")
    a2.semilogx(resp["freq"], resp["phase_deg"], "o-")
    if analytic is not None:
        a1.semilogx(analytic["freq"], analytic["mag_db"], "--", label="analytic")
        a2.semilogx(analytic["freq"], analytic["phase_deg"], "--")
    a1.set_ylabel("magnitude (dB)")
    a2.set_ylabel("phase (deg)")
    a2.set_xlabel("frequency (Hz)")
    a1.legend()
    for a in (a1, a2):
        a.grid(True, which="both", alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    return fig


def tf_response(num, den, freqs):
    """Analytic frequency response of num(s)/den(s)."""
    s = 2j * np.pi * np.asarray(freqs)
    h = np.polyval(num, s) / np.polyval(den, s)
    return {"freq": np.asarray(freqs), "gain": np.abs(h), "mag_db": 20 * np.log10(np.abs(h)),
            "phase_deg": np.degrees(np.angle(h))}
