"""Text netlist format.

Power elements (SPICE-like, first letter = type):

    R<name> n1 n2 value
    L<name> n1 n2 value [ic=..]
    C<name> n1 n2 value [ic=..]
    V<name> n+ n- DC v | SIN amp freq [phase offset] | SQUARE v1 v2 freq [duty] | SIGNAL sig [gain]
    I<name> n1 n2 (same forms as V)
    D<name> anode cathode [vf=.. ron=.. roff=..]
    S<name> n1 n2 gate_signal [ron=.. body_diode=1]
    Q<name> anode cathode gate_signal          (thyristor)
    T<name> p1 p2 s1 s2 n [lm=.. llk=..]       (transformer, n = Np/Ns)
    K<name> a b c d L1 L2 k                     (coupled inductors)
    M<name> a b ra=.. la=.. ke=.. j=.. [bf=.. tload=..]   (DC motor)
    G<name> a b c n VLL freq                    (3-phase source, VLL rms)

Control blocks:

    B<anything> TYPE out in1 in2 ... key=value ...

    e.g.  B1 TRIANGLE carrier freq=20k
          B2 CONST vref value=12
          B3 SUM err vref vo signs=+-
          B4 PI d err kp=0.05 ki=200 lo=0 hi=0.95
          B5 COMPARATOR g1 d carrier
          B6 VSENSOR vo out 0
          B7 ISENSOR il L1

Directives:

    .tran dt t_end [method=trap|be] [start=t]
    .plot sig1 sig2,sig3 ...       (comma = overlay in the same subplot)
    .measure mean|rms|max|min|pp|thd signal [f0]
    .window duration               (measure over the last `duration` seconds)
    .param name=value              (use {name} anywhere later)
    .end

Lines starting with '*' or '#' are comments; '+' continues the previous line.
"""
import re

from . import control as ctl
from . import elements as el
from . import waveforms as wf
from .circuit import Circuit
from .units import parse_value

BLOCK_TYPES = {
    # name: (class, number of outputs, number of inputs or None for variadic)
    "CONST": (ctl.Constant, 1, 0),
    "SINE": (ctl.SineWave, 1, 0),
    "TRIANGLE": (ctl.Triangle, 1, 0),
    "SAWTOOTH": (ctl.Sawtooth, 1, 0),
    "SQUARE": (ctl.SquareWave, 1, 0),
    "STEP": (ctl.StepSignal, 1, 0),
    "RAMP": (ctl.Ramp, 1, 0),
    "GAIN": (ctl.Gain, 1, 1),
    "SUM": (ctl.Sum, 1, None),
    "PRODUCT": (ctl.Product, 1, None),
    "DIVIDE": (ctl.Divide, 1, 2),
    "ABS": (ctl.Abs, 1, 1),
    "LIMITER": (ctl.Limiter, 1, 1),
    "RATELIMIT": (ctl.RateLimiter, 1, 1),
    "COMPARATOR": (ctl.Comparator, 1, 2),
    "AND": (ctl.Logic, 1, None),
    "OR": (ctl.Logic, 1, None),
    "XOR": (ctl.Logic, 1, None),
    "NOT": (ctl.Logic, 1, 1),
    "DEADTIME": (ctl.DeadTime, 2, 1),
    "PWM": (ctl.PWM, 1, 1),
    "INTEGRATOR": (ctl.Integrator, 1, 1),
    "PI": (ctl.PI, 1, 1),
    "PID": (ctl.PID, 1, 1),
    "TF": (ctl.TransferFunction, 1, 1),
    "LOWPASS": (ctl.LowPass, 1, 1),
    "DELAY": (ctl.UnitDelay, 1, 1),
    "ZOH": (ctl.SampleHold, 1, 1),
    "AVERAGE": (ctl.MovingAverage, 1, 1),
    "RMS": (ctl.RMSMeter, 1, 1),
    "VSENSOR": None,
    "ISENSOR": None,
    "PROBE": None,
}


class Netlist:
    def __init__(self, circuit, tran=None, plots=None, measures=None):
        self.circuit = circuit
        self.tran = tran or {}
        self.plots = plots or []
        self.measures = measures or []

    @property
    def window(self):
        return self.tran.get("_window")

    def run(self, **overrides):
        kw = {k: v for k, v in self.tran.items() if not k.startswith("_")}
        kw.update(overrides)
        return self.circuit.simulate(**kw)


def _kv(tokens):
    pos, kw = [], {}
    for t in tokens:
        if "=" in t:
            k, v = t.split("=", 1)
            kw[k.lower()] = v
        else:
            pos.append(t)
    return pos, kw


def _num(v):
    if isinstance(v, str) and v.lower() in ("true", "yes", "on"):
        return 1.0
    if isinstance(v, str) and v.lower() in ("false", "no", "off"):
        return 0.0
    return parse_value(v)


def _numlist(v):
    return [parse_value(x) for x in v.strip("[]()").replace(";", ",").split(",") if x]


def _source_value(name, args):
    """Parse a source specification -> (value, signal, gain)."""
    if not args:
        raise ValueError(f"{name}: missing source value")
    kind = args[0].upper()
    if kind == "DC":
        return parse_value(args[1]), None, 1.0
    if kind == "SIN":
        vals = [parse_value(a) for a in args[1:]]
        return wf.Sine(*vals), None, 1.0
    if kind == "SQUARE":
        vals = [parse_value(a) for a in args[1:]]
        return wf.Square(*vals), None, 1.0
    if kind == "STEP":
        vals = [parse_value(a) for a in args[1:]]
        return wf.Step(*vals), None, 1.0
    if kind == "SIGNAL":
        gain = parse_value(args[2]) if len(args) > 2 else 1.0
        return 0.0, args[1], gain
    return parse_value(args[0]), None, 1.0


def parse(text, title=None):
    """Parse netlist text into a :class:`Netlist`."""
    raw = text.splitlines()
    lines = []
    for ln in raw:
        s = ln.split(";", 1)[0].rstrip()
        if not s.strip() or s.lstrip()[0] in "*#":
            continue
        if s.lstrip().startswith("+") and lines:
            lines[-1] += " " + s.lstrip()[1:]
        else:
            lines.append(s.strip())

    params = {}
    c = Circuit(title or "netlist")
    tran, plots, measures = {}, [], []

    def subst(s):
        return re.sub(r"\{(\w+)\}", lambda m: params[m.group(1)], s)

    for lineno, ln in enumerate(lines, 1):
        ln = subst(ln)
        tok = ln.split()
        head = tok[0]
        try:
            if head.startswith("."):
                d = head.lower()
                if d == ".param":
                    for t in tok[1:]:
                        k, v = t.split("=", 1)
                        params[k] = v
                elif d == ".tran":
                    pos, kw = _kv(tok[1:])
                    tran["dt"] = parse_value(pos[0])
                    tran["t_end"] = parse_value(pos[1])
                    if "method" in kw:
                        tran["method"] = kw["method"].lower()
                    if "start" in kw:
                        tran["t_record_start"] = parse_value(kw["start"])
                    if "every" in kw:
                        tran["record_every"] = int(parse_value(kw["every"]))
                elif d == ".plot":
                    plots.extend([_split_overlay(g) for g in tok[1:]])
                elif d == ".window":
                    tran["_window"] = parse_value(tok[1])
                elif d == ".measure":
                    measures.append(tok[1:])
                elif d == ".title":
                    c.title = " ".join(tok[1:])
                elif d == ".end":
                    break
                else:
                    raise ValueError(f"unknown directive {head}")
                continue

            kind = head[0].upper()
            pos, kw = _kv(tok[1:])
            if kind == "R":
                c.add(el.Resistor(head, pos[0], pos[1], parse_value(pos[2])))
            elif kind == "L":
                c.add(el.Inductor(head, pos[0], pos[1], parse_value(pos[2]), _num(kw.get("ic", 0))))
            elif kind == "C":
                c.add(el.Capacitor(head, pos[0], pos[1], parse_value(pos[2]), _num(kw.get("ic", 0))))
            elif kind in "VI":
                val, sig, gain = _source_value(head, pos[2:])
                cls = el.VoltageSource if kind == "V" else el.CurrentSource
                c.add(cls(head, pos[0], pos[1], val, signal=sig, gain=gain))
            elif kind == "D":
                c.add(el.Diode(head, pos[0], pos[1], **{k: _num(v) for k, v in kw.items()}))
            elif kind == "S":
                opts = {k: _num(v) for k, v in kw.items()}
                if "body_diode" in opts:
                    opts["body_diode"] = bool(opts["body_diode"])
                c.add(el.Switch(head, pos[0], pos[1], pos[2], **opts))
            elif kind == "Q":
                c.add(el.Thyristor(head, pos[0], pos[1], pos[2], **{k: _num(v) for k, v in kw.items()}))
            elif kind == "T":
                c.add(el.Transformer(head, *pos[:4], parse_value(pos[4]), **{k: _num(v) for k, v in kw.items()}))
            elif kind == "K":
                c.add(el.CoupledInductors(head, *pos[:4], *(parse_value(p) for p in pos[4:7])))
            elif kind == "M":
                opts = {k: (v if k == "tload" and not re.match(r"^[-+.\d]", v) else _num(v))
                        for k, v in kw.items()}
                if "tload" in opts:
                    opts["t_load"] = opts.pop("tload")
                c.add(el.DCMotor(head, pos[0], pos[1], **opts))
            elif kind == "G":
                c.add(el.ThreePhaseSource(head, *pos[:4], parse_value(pos[4]), parse_value(pos[5]),
                                          *(parse_value(p) for p in pos[6:7])))
            elif kind == "B":
                c.add(_parse_block(pos, kw))
            else:
                raise ValueError(f"unknown element type {head!r}")
        except (IndexError, KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"netlist line {lineno}: {ln!r}: {exc}") from exc
    return Netlist(c, tran, plots, measures)


def _split_overlay(group):
    """Split 'V(out),I(L1)' on commas that are not inside parentheses."""
    parts, depth, cur = [], 0, ""
    for ch in group:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p for p in parts if p]


def _parse_block(pos, kw):
    btype = pos[0].upper()
    args = pos[1:]
    if btype not in BLOCK_TYPES:
        raise ValueError(f"unknown block type {btype}; known: {', '.join(BLOCK_TYPES)}")
    gain = _num(kw.pop("gain", 1))
    if btype == "VSENSOR":
        return ctl.VoltageSensor(args[0], args[1], args[2] if len(args) > 2 else "0", gain=gain)
    if btype == "ISENSOR":
        return ctl.CurrentSensor(args[0], args[1], gain=gain)
    if btype == "PROBE":
        return ctl.ProbeSensor(args[0], args[1], gain=gain)
    if gain != 1:
        kw["gain"] = str(gain)
    cls, n_out, n_in = BLOCK_TYPES[btype]
    outs, ins = args[:n_out], args[n_out:]
    if n_in is not None and len(ins) != n_in:
        raise ValueError(f"{btype} expects {n_in} input(s), got {len(ins)}")
    params = {}
    for k, v in kw.items():
        if k in ("num", "den"):
            params[k] = _numlist(v)
        elif k in ("signs", "carrier", "op"):
            params[k] = v
        elif k == "out_comp":
            params[k] = v
        else:
            params[k] = _num(v)
    if btype in ("AND", "OR", "XOR", "NOT"):
        return cls(outs[0], ins, btype.lower())
    if btype in ("SUM", "PRODUCT"):
        return cls(outs[0], ins, **params)
    if btype == "DEADTIME":
        return cls(outs[0], outs[1], ins[0], **params)
    if btype == "COMPARATOR":
        return cls(outs[0], ins[0], ins[1], **params)
    if btype == "DIVIDE":
        return cls(outs[0], ins[0], ins[1], **params)
    if n_in == 0:
        return cls(outs[0], **params)
    return cls(outs[0], ins[0], **params)


def load(path):
    with open(path) as f:
        text = f.read()
    import os
    return parse(text, title=os.path.splitext(os.path.basename(path))[0])
