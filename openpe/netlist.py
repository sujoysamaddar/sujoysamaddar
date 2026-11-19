"""Text netlist format.

Circuit elements (SPICE-like, first letter = type):

    R<name> n1 n2 value
    L<name> n1 n2 value [ic=..]
    L<name> n1 n2 value isat=.. [lsat=..] [ic=..]   (saturable inductor: L falls from `value` towards lsat above isat)
    C<name> n1 n2 value [ic=..]
    V<name> n+ n- [DC v] [AC mag [phase]] [SIN(offset amp freq [phase]) | SQUARE v1 v2 freq [duty]
                  | STEP t v0 v1 | SIGNAL sig [gain]]
    I<name> n1 n2 (same forms as V)
    D<name> anode cathode [vf=.. ron=.. roff=..]            (ideal switching diode)
    D<name> anode cathode is=.. [n=.. rs=.. bv=.. cj=..]    (Shockley / Zener diode)
    S<name> n1 n2 gate_signal [ron=.. body_diode=1]
    Q<name> c b e NPN|PNP [is=.. bf=.. br=.. vaf=..]       (BJT)
    Q<name> anode cathode gate_signal                       (thyristor)
    M<name> d g s NMOS|PMOS [vto=.. kp=.. w=.. l=.. lambda=..]   (MOSFET)
    M<name> a b ra=.. la=.. ke=.. j=.. [bf=.. tload=..]     (DC motor)
    E<name> o+ o- i+ i- gain                                (VCVS)
    G<name> o+ o- i+ i- gm                                  (VCCS)
    F<name> o+ o- Vctrl gain                                (CCCS)
    H<name> o+ o- Vctrl r                                   (CCVS)
    X<name> OPAMP in+ in- out [gain=.. gbw=.. vmax=.. vmin=.. rout=..]
    T<name> p1 p2 s1 s2 n [lm=.. llk=..]       (transformer, n = Np/Ns)
    K<name> a b c d L1 L2 k                     (coupled inductors)
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

    .tran dt t_end [method=trap|be] [start=t] [every=n] [init=ic|op]
    .subckt NAME port1 port2 ... [param=default ...]   ...   .ends      (reusable sub-circuit)
    X<name> node1 node2 ... NAME [param=value ...]                    (instance; names become X1.R1, X1.sw ...)
    .losses [devices=S1,D1] [in=Vin] [out=Rload]   (loss and efficiency report after the run)
    .change time target value      (change a component/block value at a simulation time;
    .change time target key=value   e.g. '.change 5m RL 2'  or  '.change 0.3 M1 tload=4')
    .op                            (DC operating point)
    .dc Vsrc start stop step       (DC sweep of an independent source)
    .ac dec|oct|lin n fstart fstop (AC small-signal analysis)
    .plot sig1 sig2,sig3 ...       (comma = overlay in the same subplot)
    .measure mean|rms|max|min|pp|thd signal [f0]
    .window duration               (measure over the last `duration` seconds)
    .param name=value              (use {name} anywhere later)
    .end

Lines starting with '*' or '#' are comments; '+' continues the previous line.
"""
import re

import numpy as np

from . import control as ctl
from .changes import ValueChange
from . import devices as dv
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
    "LOOKUP": (ctl.LookupTable, 1, 1),
    "SVPWM": (ctl.SVPWM, 3, 3),
    "AVERAGE": (ctl.MovingAverage, 1, 1),
    "RMS": (ctl.RMSMeter, 1, 1),
    "VSENSOR": None,
    "ISENSOR": None,
    "PROBE": None,
}


class Netlist:
    def __init__(self, circuit, tran=None, plots=None, measures=None, analyses=None, warnings=None, losses=None):
        self.circuit = circuit
        self.losses = losses             # None, or {"devices": [...], "input": [...], "output": [...]} from '.losses'
        self.warnings = warnings or []   # suspicious but legal input (e.g. '1M' = milli)
        self.tran = tran or {}
        self.plots = plots or []
        self.measures = measures or []
        self.analyses = analyses or []   # [("op",), ("dc", src, values), ("ac", freqs)]

    def op(self):
        return self.circuit.op()

    def dc(self):
        a = next(a for a in self.analyses if a[0] == "dc")
        return self.circuit.dc(a[1], a[2])

    def ac(self):
        a = next(a for a in self.analyses if a[0] == "ac")
        return self.circuit.ac(a[1])

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


def _isnum(t):
    try:
        parse_value(t)
        return True
    except ValueError:
        return False


def _source_value(name, args):
    """Parse a source specification -> dict(value, signal, gain, ac, ac_phase)."""
    if not args:
        raise ValueError(f"{name}: missing source value")
    out = {"value": None, "signal": None, "gain": 1.0, "ac": 0.0, "ac_phase": 0.0, "dc": None}
    # SIN follows SPICE argument order: offset amplitude freq [phase_deg]
    waves = {"SIN": lambda off, amp, f, ph=0.0: wf.Sine(amp, f, ph, off),
             "SQUARE": wf.Square, "STEP": wf.Step}
    i = 0
    while i < len(args):
        k = args[i].upper()
        if k == "DC":
            out["dc"] = parse_value(args[i + 1])
            i += 2
        elif k == "AC":
            out["ac"] = parse_value(args[i + 1])
            i += 2
            if i < len(args) and _isnum(args[i]):
                out["ac_phase"] = parse_value(args[i])
                i += 1
        elif k in waves:
            j = i + 1
            while j < len(args) and _isnum(args[j]):
                j += 1
            out["value"] = waves[k](*(parse_value(a) for a in args[i + 1:j]))
            i = j
        elif k == "SIGNAL":
            out["signal"] = args[i + 1]
            i += 2
            if i < len(args) and _isnum(args[i]):
                out["gain"] = parse_value(args[i])
                i += 1
        elif _isnum(args[i]):
            out["value"] = parse_value(args[i])
            i += 1
        else:
            raise ValueError(f"{name}: unexpected source token {args[i]!r}")
    if out["value"] is None:          # only DC given: it is the transient value too
        out["value"] = out["dc"] if out["dc"] is not None else 0.0
        out["dc"] = None
    return out


def _freqs(kind, n, f1, f2):
    kind = kind.lower()
    n, f1, f2 = int(parse_value(n)), parse_value(f1), parse_value(f2)
    if kind == "lin":
        return np.linspace(f1, f2, n)
    per = {"dec": 10.0, "oct": 2.0}[kind]
    count = int(round(n * np.log(f2 / f1) / np.log(per))) + 1
    return np.logspace(np.log10(f1), np.log10(f2), count)


def parse(text, title=None, _subckts=None):
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

    # ---- sub-circuit definitions:  .subckt NAME port1 port2 ... [param=default ...]  ...  .ends
    subckts = dict(_subckts or {})
    outer = []
    i = 0
    while i < len(lines):
        tk = lines[i].split()
        if tk[0].lower() == ".subckt":
            if len(tk) < 2:
                raise ValueError(f"netlist line {i + 1}: .subckt needs a name")
            ports = [t for t in tk[2:] if "=" not in t]
            defaults = dict(t.split("=", 1) for t in tk[2:] if "=" in t)
            body = []
            i += 1
            while i < len(lines) and lines[i].split()[0].lower() != ".ends":
                body.append(lines[i])
                i += 1
            if i >= len(lines):
                raise ValueError(f".subckt {tk[1]} has no matching .ends")
            subckts[tk[1].lower()] = (tk[1], ports, defaults, body)
        else:
            outer.append(lines[i])
        i += 1
    lines = outer

    params = {}
    losses = None
    c = Circuit(title or "netlist")
    tran, plots, measures, analyses, warnings = {}, [], [], [], []
    milli = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?M(?!EG)[a-zA-Z]*$")

    def subst(s):
        return re.sub(r"\{(\w+)\}", lambda m: params[m.group(1)], s)

    for lineno, ln in enumerate(lines, 1):
        ln = subst(ln)
        tok = ln.split()
        head = tok[0]
        for t in tok[1:]:
            v = t.split("=", 1)[-1]
            if milli.match(v) and not v.upper().startswith("MEG", len(v.rstrip("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))):
                warnings.append(f"line {lineno}: '{v}' means {parse_value(v):g} (SPICE: M = milli); "
                                f"write '{v.split('M')[0]}meg' for mega")
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
                    if "init" in kw:
                        tran["initial"] = kw["init"].lower()   # 'ic' (default) or 'op'

                elif d == ".losses":
                    _, lkw = _kv(tok[1:])
                    losses = {"devices": [x for x in lkw.get("devices", "").split(",") if x],
                              "input": [x for x in lkw.get("in", "").split(",") if x],
                              "output": [x for x in lkw.get("out", "").split(",") if x]}
                elif d == ".change":
                    pos, kw = _kv(tok[1:])
                    if len(pos) < 2 or (len(pos) < 3 and not kw):
                        raise ValueError("expected '.change time target value' or '.change time target key=value'")
                    t_chg, target = parse_value(pos[0]), pos[1]
                    if len(pos) >= 3:
                        c.changes.append(ValueChange(t_chg, target, None, parse_value(pos[2])))
                    for k, v in kw.items():
                        c.changes.append(ValueChange(t_chg, target, k, _num(v)))
                elif d == ".op":
                    analyses.append(("op",))
                elif d == ".dc":
                    start, stop, step = (parse_value(v) for v in tok[2:5])
                    nstep = int(round((stop - start) / step))
                    analyses.append(("dc", tok[1], start + step * np.arange(nstep + 1)))
                elif d == ".ac":
                    analyses.append(("ac", _freqs(*tok[1:5])))
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
            elif kind == "L" and "isat" in kw:
                c.add(dv.SaturableInductor(head, pos[0], pos[1], parse_value(pos[2]), _num(kw["isat"]),
                                           lsat=_num(kw["lsat"]) if "lsat" in kw else None, ic=_num(kw.get("ic", 0))))
            elif kind == "L":
                c.add(el.Inductor(head, pos[0], pos[1], parse_value(pos[2]), _num(kw.get("ic", 0))))
            elif kind == "C":
                c.add(el.Capacitor(head, pos[0], pos[1], parse_value(pos[2]), _num(kw.get("ic", 0))))
            elif kind in "VI":
                spec = _source_value(head, re.sub(r"[()]", " ", " ".join(pos[2:])).split())
                cls = el.VoltageSource if kind == "V" else el.CurrentSource
                c.add(cls(head, pos[0], pos[1], spec.pop("value"), **spec))
            elif kind == "D":
                opts = {k: _num(v) for k, v in kw.items()}
                if set(opts) & {"is", "n", "rs", "bv", "ibv", "cj"}:
                    if "is" in opts:
                        opts["is_"] = opts.pop("is")
                    c.add(dv.PNDiode(head, pos[0], pos[1], **opts))
                else:
                    c.add(el.Diode(head, pos[0], pos[1], **opts))
            elif kind == "E":
                c.add(dv.VCVS(head, *pos[:4], parse_value(pos[4])))
            elif kind == "G" and len(pos) == 5:
                c.add(dv.VCCS(head, *pos[:4], parse_value(pos[4])))
            elif kind == "F":
                c.add(dv.CCCS(head, pos[0], pos[1], pos[2], parse_value(pos[3])))
            elif kind == "H":
                c.add(dv.CCVS(head, pos[0], pos[1], pos[2], parse_value(pos[3])))
            elif kind == "X" and pos and pos[-1].lower() in subckts:
                sname, ports, defaults, body = subckts[pos[-1].lower()]
                nodes = pos[:-1]
                if len(nodes) != len(ports):
                    raise ValueError(f"subcircuit {sname} has {len(ports)} port(s) ({', '.join(ports)}) "
                                     f"but {len(nodes)} node(s) were given")
                vals = dict(defaults)
                lower = {k.lower(): k for k in defaults}
                for k, v in kw.items():
                    if k not in lower:
                        raise ValueError(f"subcircuit {sname} has no parameter {k!r}"
                                         + (f" (parameters: {', '.join(defaults)})" if defaults else ""))
                    vals[lower[k]] = v
                head_params = {**params, **vals}
                sub_text = ".param " + " ".join(f"{k}={v}" for k, v in head_params.items()) + "\n" + "\n".join(body) \
                    if head_params else "\n".join(body)
                sub = parse(sub_text, title=head, _subckts=subckts)
                c.subcircuit(head, sub.circuit, dict(zip(ports, nodes)))
                warnings.extend(f"{head}: {w}" for w in sub.warnings)
            elif kind == "X":
                if pos[0].upper() != "OPAMP":
                    raise ValueError("'X<name> OPAMP in+ in- out ...' is the built-in op-amp; for a sub-circuit "
                                     "write 'X<name> node1 node2 ... SUBCKT_NAME' with a matching .subckt definition")
                c.add(dv.OpAmp(head, pos[1], pos[2], pos[3], **{k: _num(v) for k, v in kw.items()}))
            elif kind == "Q" and len(pos) >= 4 and pos[3].upper() in ("NPN", "PNP"):
                opts = {k: _num(v) for k, v in kw.items()}
                if "is" in opts:
                    opts["is_"] = opts.pop("is")
                c.add(dv.BJT(head, pos[0], pos[1], pos[2], pos[3], **opts))
            elif kind == "M" and len(pos) >= 4 and pos[3].upper() in ("NMOS", "PMOS"):
                opts = {k: _num(v) for k, v in kw.items()}
                if "lambda" in opts:
                    opts["lam"] = opts.pop("lambda")
                c.add(dv.MOSFET(head, pos[0], pos[1], pos[2], pos[3], **opts))
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
                blk = _parse_block(pos, kw)
                blk.netname = head
                c.add(blk)
            else:
                raise ValueError(f"unknown element type {head!r}")
        except (IndexError, KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"netlist line {lineno}: {ln!r}: {exc}") from exc
    return Netlist(c, tran, plots, measures, analyses, warnings, losses)


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
        if k in ("num", "den", "xs", "ys"):
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
    if btype == "SVPWM":
        return cls(outs[0], outs[1], outs[2], ins[0], ins[1], ins[2], **params)
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
    with open(path, encoding="utf-8") as f:
        text = f.read()
    import os
    return parse(text, title=os.path.splitext(os.path.basename(path))[0])
