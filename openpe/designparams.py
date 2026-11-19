"""Design-time parameters: find every adjustable numeric value in a netlist
and change it in place, so values can be tuned (and swept) without editing
the text by hand.

    params = extract_params(text)          # [DesignParam, ...]
    text = set_param(text, params[0], "4.7k")

Recognised values:
  * ``.param name=value``                       -> group ".param"
  * R/L/C values, transformer ratio, coupling   -> e.g. "R1" / "value"
  * source DC, AC and SIN(offset amp freq phase) values
  * every ``key=value`` option of an element or control block (kp, ki, freq, ...)
  * E/G/F/H gains, 3-phase source voltage and frequency
  * ``.tran`` time step and stop time, ``.ac`` and ``.dc`` limits
  * ``.change`` events: the time and the new value of each scheduled change
Values written as ``{name}`` refer to a ``.param`` and are not listed
separately: change the ``.param`` instead.
"""
import math
import re
from dataclasses import dataclass

from .units import parse_value

_TOKEN = re.compile(r"[^\s()]+")


@dataclass
class DesignParam:
    group: str        # element / block name, ".param", ".tran", ...
    name: str         # "value", "kp", "dc", "SIN amp", ...
    value: str        # text as written in the netlist
    line: int         # 0-based line index
    start: int        # character span of the value inside the line
    end: int

    @property
    def label(self):
        return f"{self.group}.{self.name}" if self.group != ".param" else self.name

    @property
    def number(self):
        try:
            return parse_value(self.value)
        except ValueError:
            return None


def _tokens(line):
    """(text, start, end) tokens, ignoring parentheses and ';' comments."""
    code = line.split(";", 1)[0]
    return [(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(code)]


def _numeric(s):
    try:
        parse_value(s)
        return True
    except ValueError:
        return False


def extract_params(text):
    """Return the adjustable numeric values of a netlist, in file order."""
    out = []
    owner = None                                  # element a '+' continuation belongs to
    n_change = 0
    for i, raw in enumerate(text.splitlines()):
        s = raw.strip()
        if not s or s[0] in "*#":
            continue
        toks = _tokens(raw)
        if not toks:
            continue
        head = toks[0][0]
        if head.startswith("+"):
            if owner:
                out += _kv(owner, toks, i, skip_first=True)
            continue

        def add(group, name, tok):
            text_, a, b = tok
            if _numeric(text_):
                out.append(DesignParam(group, name, text_, i, a, b))

        if head.startswith("."):
            d = head.lower()
            owner = None
            if d == ".param":
                for t, a, b in toks[1:]:
                    if "=" in t:
                        k, v = t.split("=", 1)
                        if _numeric(v):
                            out.append(DesignParam(".param", k, v, i, a + len(k) + 1, b))
            elif d == ".tran" and len(toks) >= 3:
                add(".tran", "time step", toks[1])
                add(".tran", "stop time", toks[2])
                out += _kv(".tran", toks, i, skip_first=True, keys={"start"})
            elif d == ".change" and len(toks) >= 3:
                n_change += 1
                g = f".change{n_change} {toks[2][0]}"
                add(g, "time", toks[1])
                if len(toks) >= 4 and "=" not in toks[3][0]:
                    add(g, "value", toks[3])
                out += _kv(g, toks, i, skip_first=True)
            elif d == ".ac" and len(toks) >= 5:
                add(".ac", "points", toks[2])
                add(".ac", "start freq", toks[3])
                add(".ac", "stop freq", toks[4])
            elif d == ".dc" and len(toks) >= 5:
                add(".dc", "start", toks[2])
                add(".dc", "stop", toks[3])
                add(".dc", "step", toks[4])
            continue

        name = head
        kind = head[0].upper()
        owner = name
        pos = [t for t in toks[1:] if "=" not in t[0]]
        if kind in "RLC" and len(pos) >= 3:
            add(name, "value", pos[2])
        elif kind in "VI":
            words = pos[2:]
            j = 0
            while j < len(words):
                w = words[j][0].upper()
                if w in ("DC", "AC") and j + 1 < len(words):
                    add(name, w.lower(), words[j + 1])
                    j += 2
                    if w == "AC" and j < len(words) and _numeric(words[j][0]):
                        add(name, "ac phase", words[j])
                        j += 1
                elif w in ("SIN", "SQUARE", "STEP"):
                    labels = {"SIN": ["offset", "amp", "freq", "phase"],
                              "SQUARE": ["low", "high", "freq", "duty"],
                              "STEP": ["time", "before", "after"]}[w]
                    j += 1
                    k = 0
                    while j < len(words) and _numeric(words[j][0]):
                        add(name, f"{w} {labels[k] if k < len(labels) else k}", words[j])
                        j += 1
                        k += 1
                elif w == "SIGNAL":
                    j += 2
                    if j < len(words) and _numeric(words[j][0]):
                        add(name, "gain", words[j])
                        j += 1
                elif _numeric(words[j][0]):
                    add(name, "value", words[j])
                    j += 1
                else:
                    j += 1
        elif kind == "T" and len(pos) >= 5:
            add(name, "turns ratio", pos[4])
        elif kind == "K" and len(pos) >= 7:
            for lbl, t in zip(("L1", "L2", "k"), pos[4:7]):
                add(name, lbl, t)
        elif kind == "E" and len(pos) >= 5:
            add(name, "gain", pos[4])
        elif kind == "G" and len(pos) == 5:
            add(name, "gm", pos[4])
        elif kind == "G" and len(pos) >= 6:
            add(name, "VLL rms", pos[4])
            add(name, "freq", pos[5])
        elif kind in "FH" and len(pos) >= 4:
            add(name, "gain" if kind == "F" else "r", pos[3])
        out += _kv(name, toks, i, skip_first=True)
    return out


def _kv(group, toks, line, skip_first=False, keys=None):
    res = []
    for t, a, b in toks[1 if skip_first else 0:]:
        if "=" not in t:
            continue
        k, v = t.split("=", 1)
        if keys is not None and k.lower() not in keys:
            continue
        if _numeric(v):
            res.append(DesignParam(group, k, v, line, a + len(k) + 1, b))
    return res


def set_param(text, p, value):
    """Return ``text`` with parameter ``p`` replaced by ``value`` (a string or
    number).  Raises ValueError for a non-numeric value."""
    value = format_value(value) if isinstance(value, (int, float)) else str(value).strip()
    parse_value(value)                             # validate
    lines = text.split("\n")
    ln = lines[p.line]
    if ln[p.start:p.end] != p.value:
        raise ValueError(f"netlist changed: expected {p.value!r} at line {p.line + 1}")
    lines[p.line] = ln[:p.start] + value + ln[p.end:]
    return "\n".join(lines)


def find(params, label):
    for p in params:
        if p.label == label:
            return p
    raise KeyError(label)


_PREFIX = [(1e12, "T"), (1e9, "G"), (1e6, "meg"), (1e3, "k"), (1.0, ""), (1e-3, "m"), (1e-6, "u"),
           (1e-9, "n"), (1e-12, "p"), (1e-15, "f")]


def format_value(v, digits=4):
    """Number -> SPICE text with an SI suffix (2200 -> '2.2k', 1e6 -> '1meg')."""
    if v == 0 or not math.isfinite(v):
        return "0" if v == 0 else repr(v)
    a = abs(v)
    for mult, suf in _PREFIX:
        if a >= mult * 0.9999999:
            s = f"{v / mult:.{digits}g}"
            return s + suf
    return f"{v:.{digits}g}"


def sweep_values(start, stop, n, log=False):
    """n values from start to stop (inclusive), linear or logarithmic."""
    a, b = parse_value(start) if isinstance(start, str) else start, parse_value(stop) if isinstance(stop, str) else stop
    n = max(2, int(n))
    if log:
        if a <= 0 or b <= 0:
            raise ValueError("a logarithmic sweep needs positive limits")
        return [a * (b / a) ** (k / (n - 1)) for k in range(n)]
    return [a + (b - a) * k / (n - 1) for k in range(n)]


# ---------------------------------------------------------------------------
# command-line style overrides and standard-value snapping
# ---------------------------------------------------------------------------
def apply_overrides(text, assignments):
    """Apply ``["R1.value=4.7k", "vin=24", ...]`` to a netlist.  A label is
    the one shown by :func:`extract_params` (``R1.value``, ``B4.kp``,
    ``.tran.stop time``, or just ``vin`` for a ``.param``).  Matching is
    case-insensitive; an unknown label lists close matches."""
    import difflib
    for a in assignments:
        if "=" not in a:
            raise ValueError(f"override {a!r} must look like LABEL=VALUE, e.g. R1.value=4.7k")
        label, value = a.split("=", 1)
        label = label.strip()
        params = extract_params(text)          # re-extract: earlier edits may move offsets
        by_label = {p.label: p for p in params}
        match = by_label.get(label) or next((p for k, p in by_label.items() if k.lower() == label.lower()), None)
        if match is None and "." not in label:     # 'R1' alone: use its main value if unambiguous
            cand = [p for p in params if p.group.lower() == label.lower() and p.name in ("value", "dc")]
            if len(cand) == 1:
                match = cand[0]
        if match is None:
            close = difflib.get_close_matches(label, list(by_label), n=4, cutoff=0.5)
            raise ValueError(f"unknown parameter {label!r}" + (f"; did you mean: {', '.join(close)}?" if close
                                                                else " (use --list-params to see them)"))
        text = set_param(text, match, value.strip())
    return text


def params_table(params):
    """Human-readable table of parameters (label, value)."""
    if not params:
        return "(no adjustable values found)"
    w = max(len(p.label) for p in params)
    return "\n".join(f"{p.label:<{w}}  {p.value}" for p in params)


_KIND_GROUP = {"R": "R", "C": "C", "L": "L"}


def snap_to_series(text, series="E24", kinds="RC"):
    """Replace the values of R / C / L elements (the letters in ``kinds``) with
    the nearest standard value of an E-series.  Returns ``(new_text, changes)``
    where ``changes`` is a list of ``(element, old, new, error_percent)`` for
    the values that actually changed.  Values given as ``{param}`` references
    are left alone."""
    from . import eseries
    changed = []
    for p in extract_params(text):
        if p.name != "value" or p.group[:1].upper() not in kinds.upper() or p.group.startswith("."):
            continue
        v = p.number
        if v is None or v <= 0:
            continue
        snapped = eseries.nearest(v, series)
        if abs(snapped - v) <= 1e-9 * v:
            continue
        new = format_value(snapped)
        text = set_param(text, find(extract_params(text), p.label), new)
        changed.append((p.group, p.value, new, 100.0 * (snapped - v) / v))
    return text, changed
