"""Design by target: find the value of one parameter that makes a measurement
reach a given number.

    sol = solve_for(text, "R1.value", "overshoot V(b)", target=20, lo=2, hi=50)
    print(sol.value, sol.measure)       # R1 for which the overshoot is 20 %
    open("tuned.net", "w").write(sol.text)

``lo`` / ``hi`` bracket the search; the measurement at both ends must lie on
opposite sides of the target.  The solver is the Illinois variant of
regula falsi (robust like bisection, usually much faster), optionally in
logarithmic parameter space for values spanning decades.  Every evaluation
is a full simulation of the netlist, so the number of runs is reported.
"""
import math
from dataclasses import dataclass, field

from .designparams import apply_overrides, format_value
from .measures import measure, split_spec
from .netlist import parse


@dataclass
class Solution:
    label: str
    spec: str
    target: float
    value: float
    measure: float
    iterations: int
    converged: bool
    history: list = field(default_factory=list)      # [(value, measure), ...] in evaluation order
    text: str = ""

    def __str__(self):
        return (f"{self.label} = {format_value(self.value, 6)}  ->  {self.spec} = {self.measure:.6g} "
                f"(target {self.target:g}, {self.iterations} runs{'' if self.converged else ', NOT converged'})")


def evaluate(text, label, value, spec):
    """Run ``text`` with ``label`` set to ``value`` and return the measurement."""
    t2 = apply_overrides(text, [f"{label}={format_value(value, 9)}"])
    nl = parse(t2)
    if not nl.tran:
        raise ValueError("solving needs a .tran analysis in the netlist")
    res = nl.run()
    return measure(res.last(nl.window) if nl.window else res, spec), t2


def solve_for(text, label, spec, target, lo, hi, log=False, rtol=1e-4, max_iter=40):
    spec = " ".join(split_spec(spec))
    target = float(target)
    if not (hi > lo):
        raise ValueError("solve_for: hi must be greater than lo")
    if log and lo <= 0:
        raise ValueError("a logarithmic search needs positive limits")
    to_x = math.log if log else (lambda v: v)
    from_x = math.exp if log else (lambda x: x)
    hist, cache = [], {}

    def f(v):
        m, t2 = evaluate(text, label, v, spec)
        hist.append((v, m))
        cache[v] = (m, t2)
        return m - target

    a, b = float(lo), float(hi)
    fa, fb = f(a), f(b)
    tol = rtol * max(abs(target), 1e-12)
    done = None
    if abs(fa) <= tol:
        done = a
    elif abs(fb) <= tol:
        done = b
    elif fa * fb > 0:
        raise ValueError(f"target {target:g} is not between the measurements at the ends of the range: "
                         f"{spec} = {fa + target:.6g} at {label}={format_value(a)} and "
                         f"{fb + target:.6g} at {label}={format_value(b)}")
    side = 0
    it = 0
    while done is None and it < max_iter:
        it += 1
        xa, xb = to_x(a), to_x(b)
        x = (xa * fb - xb * fa) / (fb - fa)
        v = from_x(x)
        if not (min(a, b) < v < max(a, b)):               # numerical safety: bisect
            v = from_x(0.5 * (xa + xb))
        fv = f(v)
        if abs(fv) <= tol or abs(b - a) <= 1e-9 * max(abs(a), abs(b)):
            done = v
            break
        if fv * fb > 0:                                    # root in [a, v]
            b, fb = v, fv
            if side == -1:
                fa *= 0.5
            side = -1
        else:                                              # root in [v, b]
            a, fa = v, fv
            if side == 1:
                fb *= 0.5
            side = 1
    converged = done is not None
    if done is None:                                       # not converged: report the closest run
        done = min(cache, key=lambda v: abs(cache[v][0] - target))
    m, solved = cache[done]
    return Solution(label, spec, target, done, m, len(hist), converged, hist, solved)
