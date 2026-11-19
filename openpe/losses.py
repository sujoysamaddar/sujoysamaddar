"""Loss and thermal analysis of a transient simulation.

    rep = res.last(0.02).loss_report(output="Rload", input="Vin")
    print(rep)                          # per-device conduction + switching loss, efficiency
    tj = res.junction_temperature("S1", rth=1.2, cth=0.05, ta=40)   # Tj(t) in deg C

Conduction loss of a device is the mean of v*i over the analysed window
(ron and the on-state drop vf are included).  Switching loss comes from the
turn-on / turn-off / reverse-recovery energies given on the device
(``eon``, ``eoff``, ``erec`` with ``vref`` / ``iref``, see
:class:`openpe.elements._Switching`): the recorded cumulative energy
``Esw(<name>)`` is differentiated over the window.
"""
import math
from dataclasses import dataclass, field

import numpy as np

SWITCHING_TYPES = ("Switch", "Diode", "Thyristor")


def _names(x):
    if x is None:
        return []
    return [x] if isinstance(x, str) else list(x)


@dataclass
class LossRow:
    name: str
    kind: str
    conduction: float          # W
    switching: float           # W

    @property
    def total(self):
        return self.conduction + self.switching


@dataclass
class LossReport:
    t0: float
    t1: float
    rows: list = field(default_factory=list)
    p_in: float = None         # W delivered by the input source(s)
    p_out: float = None        # W absorbed by the output element(s)
    p_storage: float = 0.0     # W of net energy going into capacitors / inductors (drift of the window)

    @property
    def device_loss(self):
        return sum(r.total for r in self.rows)

    @property
    def conduction_loss(self):
        return sum(r.conduction for r in self.rows)

    @property
    def switching_loss(self):
        return sum(r.switching for r in self.rows)

    @property
    def other_loss(self):
        """Circuit power not accounted for by the output, the conduction loss of
        the listed devices and the energy stored in capacitors and inductors
        during the window: loss in winding / ESR resistors and any other element,
        plus a discretisation error of the sampled power integrals that grows
        with the time step (about V * dI * dt / 2 per switching event; it halves
        when the step halves - use >= 200 steps per switching period for
        sub-percent figures)."""
        if self.p_in is None or self.p_out is None:
            return None
        return self.p_in - self.p_out - self.conduction_loss - self.p_storage

    @property
    def efficiency(self):
        """Output power / (input power - stored-energy drift + switching loss).
        The drift term makes the figure meaningful even when the window is not
        exactly periodic (a converter still settling).  Switching loss is a
        post-processed estimate the simulated circuit does not draw, so a real
        converter takes it from the source in addition."""
        if self.p_in is None or self.p_out is None:
            return None
        total_in = self.p_in - self.p_storage + self.switching_loss
        return self.p_out / total_in if total_in > 0 else None

    def to_dict(self):
        return {"window": [self.t0, self.t1],
                "rows": [{"name": r.name, "type": r.kind, "conduction": r.conduction,
                          "switching": r.switching, "total": r.total} for r in self.rows],
                "device_loss": self.device_loss, "conduction_loss": self.conduction_loss,
                "switching_loss": self.switching_loss, "p_storage": self.p_storage,
                "p_in": self.p_in, "p_out": self.p_out,
                "other_loss": self.other_loss, "efficiency": self.efficiency}

    def __str__(self):
        w = max([len(r.name) for r in self.rows] + [6])
        lines = [f"Loss report, {self.t0 * 1e3:.4g} .. {self.t1 * 1e3:.4g} ms",
                 f"{'device':<{w}}  {'type':<10}{'conduction':>12}{'switching':>12}{'total':>12}"]
        for r in self.rows:
            lines.append(f"{r.name:<{w}}  {r.kind:<10}{r.conduction:12.4g}{r.switching:12.4g}{r.total:12.4g}")
        lines.append(f"{'devices':<{w}}  {'':<10}{sum(r.conduction for r in self.rows):12.4g}"
                     f"{sum(r.switching for r in self.rows):12.4g}{self.device_loss:12.4g}  W")
        if self.p_in is not None and self.p_out is not None:
            lines += [f"input power {self.p_in:.5g} W (circuit) + {self.switching_loss:.4g} W switching loss "
                      f"- {self.p_storage:.4g} W stored in C/L = {self.p_in + self.switching_loss - self.p_storage:.5g} W    "
                      f"output power {self.p_out:.5g} W",
                      f"other (resistors, ... and sampling error that shrinks with dt): {self.other_loss:.4g} W",
                      f"efficiency  {100 * self.efficiency:.2f} %"]
        return "\n".join(lines)


def loss_report(res, devices=None, input=None, output=None):
    """Build a :class:`LossReport` for ``res`` (use ``res.window / res.last`` first
    to restrict it to steady state).

    devices : element names to report; default = every switch, diode and thyristor
    input   : source(s) delivering the power (efficiency denominator)
    output  : element(s) absorbing the useful power (efficiency numerator)
    """
    types = res.meta.get("element_types", {})
    names = _names(devices) or [n for n, t in types.items() if t in SWITCHING_TYPES]
    t = res.t
    span = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    rep = LossReport(float(t[0]), float(t[-1]))
    for n in names:
        cond = res.power(n)
        key = f"Esw({n})"
        sw = 0.0
        if key in res.data and span > 0:
            e = res.data[key]
            sw = float(e[-1] - e[0]) / span
        rep.rows.append(LossRow(n, types.get(n, "?"), float(cond), sw))
    if input is not None and output is not None:
        rep.p_in = -sum(res.power(n) for n in _names(input))
        rep.p_out = sum(res.power(n) for n in _names(output))
        rep.p_storage = storage_power(res)
    return rep


def storage_power(res):
    """Net power (W) flowing into the capacitors and inductors over the window:
    (energy at the end - energy at the start) / duration, with
    E = C v^2 / 2 and E = L i^2 / 2."""
    t = res.t
    span = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    if span <= 0:
        return 0.0
    de = 0.0
    for s in res.meta.get("storage", []):
        n = s["name"]
        if s["kind"] == "C":
            q = res.voltage_across(n)
            de += 0.5 * s["value"] * (float(q[-1]) ** 2 - float(q[0]) ** 2)
        elif f"I({n})" in res.data:
            i = res[f"I({n})"]
            de += 0.5 * s["value"] * (float(i[-1]) ** 2 - float(i[0]) ** 2)
    return de / span


def junction_temperature(res, device, rth=None, cth=None, ta=25.0, stages=None):
    """Junction temperature waveform (deg C) of a device from its dissipation.

    The power is the device's instantaneous conduction loss v*i plus its
    switching energy spread over the recording step.  The thermal model is a
    Foster network: either one stage (``rth`` K/W, ``cth`` J/K) or
    ``stages=[(r1, c1), (r2, c2), ...]``.  Each stage follows
    T' = (P*R - T) / (R*C) and the junction rise is the sum of the stages.
    With ``cth=None`` the device is a pure resistance (steady-state
    Tj = Ta + Rth * P(t)).

    ``device`` may also be an array of power samples (same length as ``res.t``).
    """
    t = np.asarray(res.t, float)
    if isinstance(device, str):
        p = np.asarray(res.voltage_across(device) * res[f"I({device})"], float)
        key = f"Esw({device})"
        if key in res.data and len(t) > 1:
            de = np.diff(res.data[key], prepend=res.data[key][0])
            dts = np.diff(t, prepend=t[0] - (t[1] - t[0]))
            p = p + de / np.where(dts > 0, dts, np.inf)
    else:
        p = np.asarray(device, float)
        if len(p) != len(t):
            raise ValueError("power array must have one sample per recorded time step")
    if stages is None:
        if rth is None:
            raise ValueError("give rth (and optionally cth) or stages=[(r, c), ...]")
        stages = [(rth, cth)]
    rise = np.zeros_like(p)
    dt = np.diff(t, prepend=t[0] - (t[1] - t[0]) if len(t) > 1 else t[0])
    for r, c in stages:
        if r <= 0:
            raise ValueError("thermal resistance must be > 0")
        if not c:                                  # no capacitance: instantaneous
            rise += p * r
            continue
        tau = r * c
        x = np.empty_like(p)
        state = 0.0
        for k in range(len(p)):
            a = math.exp(-dt[k] / tau)
            state = state * a + p[k] * r * (1.0 - a)
            x[k] = state
        rise += x
    return ta + rise
