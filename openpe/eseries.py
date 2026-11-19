"""Standard component values: the IEC 60063 E-series (E6 ... E96).

    nearest(4.6e3, "E24")      -> 4700.0      closest standard value
    step(4.7e3, +1, "E24")     -> 5100.0      next standard value up
    step(4.7e3, -1, "E24")     -> 4300.0      next standard value down
    error_percent(4.6e3, "E24")               relative error of snapping
"""
import math

E6 = [1.0, 1.5, 2.2, 3.3, 4.7, 6.8]
E12 = [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2]
E24 = [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.3,
       4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1]
# E48 and E96 are the rounded geometric series 10**(k/N) with 3 significant digits
E48 = [round(10 ** (k / 48), 2) for k in range(48)]
E96 = [round(10 ** (k / 96), 2) for k in range(96)]

SERIES = {"E6": E6, "E12": E12, "E24": E24, "E48": E48, "E96": E96}


def series_values(series="E24"):
    """Mantissas (1.0 ... 9.x) of a series."""
    try:
        return list(SERIES[series.upper()])
    except KeyError:
        raise ValueError(f"unknown series {series!r}; choose from {', '.join(SERIES)}") from None


def _split(v):
    if not (v > 0) or not math.isfinite(v):
        raise ValueError("E-series values must be positive and finite")
    dec = math.floor(math.log10(v))
    mant = v / 10.0 ** dec
    if mant >= 10.0 - 1e-12:           # guard against log10 rounding at exact decades
        dec, mant = dec + 1, mant / 10.0
    return dec, mant


def _join(mant, dec):
    return float(f"{mant * 10.0 ** dec:.6g}")


def nearest(v, series="E24"):
    """Closest standard value (nearest in the logarithmic sense)."""
    vals = series_values(series) + [10.0]
    dec, mant = _split(v)
    best = min(vals, key=lambda s: abs(math.log(s / mant)))
    return _join(best, dec)


def step(v, direction=+1, series="E24"):
    """The standard value above (direction > 0) or below (< 0) ``v``.  A value
    that is already standard moves to its neighbour; any other value moves to
    the closest standard value in that direction."""
    vals = series_values(series)
    dec, mant = _split(v)
    eps = 1e-9
    if direction > 0:
        for s in vals:
            if s > mant * (1 + eps):
                return _join(s, dec)
        return _join(vals[0], dec + 1)
    for s in reversed(vals):
        if s < mant * (1 - eps):
            return _join(s, dec)
    return _join(vals[-1], dec - 1)


def error_percent(v, series="E24"):
    """Relative error (%) made by replacing ``v`` with its nearest standard value."""
    return 100.0 * (nearest(v, series) - v) / v


def is_standard(v, series="E24", tol=1e-6):
    return abs(nearest(v, series) - v) <= tol * abs(v)
