"""Named measurements of a simulation result, shared by ``.measure`` lines,
the command line, the GUIs and the parameter solver.

A measurement is a list such as ``["max", "V(out)"]`` or the same text,
``"overshoot V(b) 0"``.  Available kinds:

    mean rms min max pp final        statistics of a signal
    thd SIG F0 [CYCLES]              total harmonic distortion (ratio)
    rise SIG [T_STEP]                10-90 % rise time of a step response
    delay SIG [T_STEP]               time to reach 50 % of the step
    overshoot SIG [T_STEP]           overshoot in percent of the step size
    peaktime SIG [T_STEP]            time of the first peak after the step
    settle SIG [T_STEP]              2 % settling time
    power ELEMENT                    mean power absorbed by an element (W)
    eff OUT_ELEMENT IN_SOURCE        circuit efficiency P_out / P_in
"""
from .units import parse_value

KINDS = ("mean", "rms", "min", "max", "pp", "final", "thd", "rise", "delay", "overshoot", "peaktime",
         "settle", "power", "eff")
_STEP = {"rise": "rise_time", "delay": "delay_time", "overshoot": "overshoot_pct",
         "peaktime": "peak_time", "settle": "settling_time"}


def split_spec(spec):
    return spec.split() if isinstance(spec, str) else list(spec)


def measure(res, spec):
    """Evaluate one measurement on a :class:`Results`."""
    spec = split_spec(spec)
    if len(spec) < 2:
        raise ValueError(f"measurement {' '.join(spec)!r} needs a signal, e.g. 'max V(out)'")
    kind, sig = spec[0].lower(), spec[1]
    if kind not in KINDS:
        raise ValueError(f"unknown measurement {kind!r}; choose from {', '.join(KINDS)}")
    if kind == "mean":
        return res.mean(sig)
    if kind == "rms":
        return res.rms(sig)
    if kind == "max":
        return res.max(sig)
    if kind == "min":
        return res.min(sig)
    if kind == "pp":
        return res.ripple(sig)
    if kind == "final":
        return res.final(sig)
    if kind == "thd":
        if len(spec) < 3:
            raise ValueError("thd needs the fundamental frequency: 'thd V(out) 50 [cycles]'")
        return res.thd(sig, parse_value(spec[2]), cycles=int(spec[3]) if len(spec) > 3 else 1)
    if kind == "power":
        return res.power(sig)
    if kind == "eff":
        if len(spec) < 3:
            raise ValueError("eff needs an output element and an input source: 'eff Rload Vin'")
        return res.efficiency(sig, spec[2])
    t_step = parse_value(spec[2]) if len(spec) > 2 else None
    return res.step_info(sig, t_step=t_step)[_STEP[kind]]
