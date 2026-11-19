"""SI-suffix value parsing (SPICE style): 10k, 4.7u, 100uH, 2meg, 1e-3 ..."""
import re

_SUFFIX = {
    "f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6,
    "m": 1e-3, "k": 1e3, "meg": 1e6, "g": 1e9, "t": 1e12,
}
_RE = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)(meg|[fpnuµmkgt])?([a-zA-Z]*)$",
                 re.IGNORECASE)


def parse_value(text):
    """Convert a number with an optional SI suffix to float.

    >>> parse_value("4.7u")
    4.7e-06
    >>> parse_value("10kHz")
    10000.0
    """
    if isinstance(text, (int, float)):
        return float(text)
    s = str(text).strip()
    m = _RE.match(s)
    if not m:
        raise ValueError(f"cannot parse value {text!r}")
    num, suf, _unit = m.groups()
    mult = _SUFFIX[suf.lower()] if suf else 1.0
    return float(num) * mult
