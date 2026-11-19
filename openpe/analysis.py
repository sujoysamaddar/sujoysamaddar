"""Analyses built on top of transient simulation: parameter sweep and
AC sweep (frequency response by sinusoidal injection, which works for
switching circuits as well as linear ones)."""
import itertools
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .control import SineWave
from .results import _finish, _subplots
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
    fig, (a1, a2) = _subplots(show, 2, 1, sharex=True, figsize=(8, 5.5))
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
    _finish(fig, show)
    return fig


def tf_response(num, den, freqs):
    """Analytic frequency response of num(s)/den(s)."""
    s = 2j * np.pi * np.asarray(freqs)
    h = np.polyval(num, s) / np.polyval(den, s)
    return {"freq": np.asarray(freqs), "gain": np.abs(h), "mag_db": 20 * np.log10(np.abs(h)),
            "phase_deg": np.degrees(np.angle(h))}


# ---------------------------------------------------------------------------
# Monte Carlo tolerance analysis
# ---------------------------------------------------------------------------
class MonteCarloResult:
    def __init__(self, rows, params, metrics):
        self.rows, self.params, self.metrics = rows, params, metrics

    def values(self, name):
        return np.array([r[name] for r in self.rows], float)

    def stats(self, metric):
        v = self.values(metric)
        return {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "min": float(np.min(v)), "max": float(np.max(v))}

    def yield_(self, metric, lo=-math.inf, hi=math.inf):
        """Fraction of runs with lo <= metric <= hi."""
        v = self.values(metric)
        return float(np.mean((v >= lo) & (v <= hi)))

    def summary(self):
        lines = [f"Monte Carlo: {len(self.rows)} runs",
                 f"{'metric':<16}{'mean':>12}{'std':>12}{'min':>12}{'max':>12}"]
        for m in self.metrics:
            st = self.stats(m)
            lines.append(f"{m:<16}{st['mean']:12.5g}{st['std']:12.5g}{st['min']:12.5g}{st['max']:12.5g}")
        return "\n".join(lines)

    def histogram(self, metric, bins=20, save=None, show=False, limits=None, title=None):
        v = self.values(metric)
        fig, ax = _subplots(show, figsize=(7, 3.8))
        ax.hist(v, bins=bins, edgecolor="white")
        if limits:
            for x in limits:
                ax.axvline(x, color="C3", linestyle="--")
        st = self.stats(metric)
        ax.set_title(title or f"{metric}: mean {st['mean']:.4g}, sigma {st['std']:.3g} ({len(v)} runs)")
        ax.set_xlabel(metric)
        ax.set_ylabel("count")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=120)
        _finish(fig, show)
        return fig


def monte_carlo(run, nominal, tolerances, n=100, seed=0, distribution="uniform"):
    """Monte Carlo tolerance analysis.

    run         : function(**params) -> {"metric": value, ...}; it builds the
                  circuit from the parameters and runs any analysis
                  (transient, .op, .ac ...)
    nominal     : {"R1": 10e3, "C1": 1e-9, ...}
    tolerances  : {"R1": 0.01, "C1": 0.05, ...} relative tolerance.  For
                  'uniform' values are spread over +/-tol; for 'gauss' tol
                  is the 3-sigma limit.
    """
    if distribution not in ("uniform", "gauss"):
        raise ValueError("distribution must be 'uniform' or 'gauss'")
    rng = np.random.default_rng(seed)
    rows, metrics = [], None
    for _ in range(n):
        p = {}
        for k, v in nominal.items():
            tol = tolerances.get(k, 0.0)
            f = rng.uniform(-tol, tol) if distribution == "uniform" else rng.normal(0.0, tol / 3.0)
            p[k] = v * (1.0 + f)
        res = run(**p)
        if metrics is None:
            metrics = list(res)
        row = dict(p)
        row.update(res)
        rows.append(row)
    return MonteCarloResult(rows, list(nominal), metrics or [])
