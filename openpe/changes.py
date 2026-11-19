"""Value changes: set a component or control-block parameter, either
immediately (design time) or at a given simulation time (run time).

    circuit.at(5e-3, "RL", 2.0)              # load resistor drops to 2 ohm at 5 ms
    circuit.at(8e-3, "d", 0.5)               # control signal 'd' (a Constant) steps to 0.5
    circuit.at(0.3, "M1", t_load=4.0)        # DC-motor load torque step
    circuit.at(2e-3, "V1", freq=2e3)         # change the sine frequency of a source

    circuit.set_value("R1", 4.7e3)           # change before simulating

Netlist form:  ``.change 5m RL 2``   or   ``.change 0.3 M1 tload=4``.

A change takes effect for the first time step that starts at or after its
time.  Targets are element names (R1, V1, M1, T1_Lm ...) or control blocks,
named either by their output signal or by their netlist name (B4).
"""
import copy
from dataclasses import dataclass
from typing import Optional

from . import control as ctl
from . import elements as el
from .waveforms import DC

_ALIASES = {"tload": "t_load", "load": "t_load"}
_POSITIVE = ("r", "c", "l")


@dataclass
class ValueChange:
    t: float
    target: str
    attr: Optional[str]          # None = the target's default value
    value: float

    def to_dict(self):
        return {"t": self.t, "target": self.target, "attr": self.attr, "value": self.value}


def default_attr(obj):
    """Name of the attribute 'the value' of ``obj`` refers to, or None."""
    if isinstance(obj, el.Resistor):
        return "r"
    if isinstance(obj, el.Capacitor):
        return "c"
    if isinstance(obj, el.Inductor):
        return "l"
    if isinstance(obj, (el.VoltageSource, el.CurrentSource)):
        return "value"
    if isinstance(obj, ctl.Constant):
        return "value"
    if isinstance(obj, ctl.Gain):
        return "k"
    return None


def label(obj):
    return getattr(obj, "name", None) or (obj.outputs[0] if getattr(obj, "outputs", None) else repr(obj))


def adjustable_attrs(obj):
    """Numeric attributes that may be changed on ``obj``."""
    out = []
    for k, v in vars(obj).items():
        if k.startswith("_") or k in ("idx", "br", "n_branch", "nodes", "name"):
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(k)
    w = getattr(obj, "wave", None)
    if w is not None:
        out += [k for k, v in vars(w).items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return out


def find_target(elements, blocks, name):
    for e in elements:
        if e.name == name:
            return e
    for b in blocks:
        if name in b.outputs or getattr(b, "netname", None) == name:
            return b
    names = [e.name for e in elements]
    sigs = [o for b in blocks for o in b.outputs]
    raise ValueError(f"unknown target {name!r}. Elements: {', '.join(names) or '-'}; "
                     f"control signals: {', '.join(sigs) or '-'}")


def set_attr(obj, attr, value):
    """Set parameter ``attr`` (None = default value) of an element or block."""
    value = float(value)
    attr = _ALIASES.get(attr, attr) if attr else (default_attr(obj))
    if attr is None:
        raise ValueError(f"{label(obj)}: give the parameter to change, one of: "
                         f"{', '.join(adjustable_attrs(obj)) or '-'}")
    if isinstance(obj, (el.VoltageSource, el.CurrentSource)):
        if getattr(obj, "signal", None) is not None and attr in ("value", "dc"):
            raise ValueError(f"{obj.name} is driven by control signal {obj.signal!r}; change that signal instead")
        if attr in ("value", "dc"):
            obj.wave = DC(value)
            if obj.dc is not None:
                obj.dc = value
            return
        if attr not in vars(obj) and attr in vars(obj.wave):
            setattr(obj.wave, attr, value)
            return
    if attr not in vars(obj):
        raise ValueError(f"{label(obj)} has no parameter {attr!r}. Available: {', '.join(adjustable_attrs(obj)) or '-'}")
    if attr in _POSITIVE and not value > 0:
        raise ValueError(f"{label(obj)}: {attr} must be > 0 (got {value:g})")
    setattr(obj, attr, value)
    if isinstance(obj, el.CoupledInductors) and attr in ("l1", "l2", "k"):
        obj.m = obj.k * (obj.l1 * obj.l2) ** 0.5


def get_attr(obj, attr=None):
    attr = _ALIASES.get(attr, attr) if attr else default_attr(obj)
    if attr is None:
        raise ValueError(f"{label(obj)}: give the parameter, one of: {', '.join(adjustable_attrs(obj)) or '-'}")
    if isinstance(obj, (el.VoltageSource, el.CurrentSource)) and attr in ("value", "dc"):
        return obj.wave(0.0)
    if attr in vars(obj):
        return vars(obj)[attr]
    w = getattr(obj, "wave", None)
    if w is not None and attr in vars(w):
        return vars(w)[attr]
    raise ValueError(f"{label(obj)} has no parameter {attr!r}")


def resolve_changes(changes, elements, blocks):
    """Validate scheduled changes and bind them to objects.  Returns a list of
    (t, obj, attr, value, change) sorted by time (stable)."""
    out = []
    for ch in changes:
        obj = find_target(elements, blocks, ch.target)
        try:
            set_attr(copy.deepcopy(obj), ch.attr, ch.value)       # dry run: unknown attribute, bad value...
        except ValueError as exc:
            raise ValueError(f"change at t={ch.t:g}: {exc}") from None
        out.append((float(ch.t), obj, ch.attr, float(ch.value), ch))
    out.sort(key=lambda r: r[0])
    return out
