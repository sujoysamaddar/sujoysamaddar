"""Time-domain source waveforms used by independent voltage/current sources."""
import math

import numpy as np


class Waveform:
    def __call__(self, t):
        raise NotImplementedError


class DC(Waveform):
    def __init__(self, value):
        self.value = float(value)

    def __call__(self, t):
        return self.value


class Sine(Waveform):
    """offset + amp * sin(2*pi*freq*t + phase_deg)."""

    def __init__(self, amp, freq, phase=0.0, offset=0.0):
        self.amp, self.freq, self.phase, self.offset = float(amp), float(freq), float(phase), float(offset)

    def __call__(self, t):
        return self.offset + self.amp * math.sin(2 * math.pi * self.freq * t + math.radians(self.phase))


class Square(Waveform):
    """Rectangular pulse train between v1 (low) and v2 (high)."""

    def __init__(self, v1, v2, freq, duty=0.5, delay=0.0):
        self.v1, self.v2, self.freq, self.duty, self.delay = float(v1), float(v2), float(freq), float(duty), float(delay)

    def __call__(self, t):
        if t < self.delay:
            return self.v1
        frac = ((t - self.delay) * self.freq) % 1.0
        return self.v2 if frac < self.duty else self.v1


class Step(Waveform):
    def __init__(self, t_step, before=0.0, after=1.0):
        self.t_step, self.before, self.after = float(t_step), float(before), float(after)

    def __call__(self, t):
        return self.after if t >= self.t_step else self.before


class PWL(Waveform):
    """Piecewise linear: points = [(t0, v0), (t1, v1), ...]; held constant outside."""

    def __init__(self, points):
        pts = sorted((float(a), float(b)) for a, b in points)
        self.ts = np.array([p[0] for p in pts])
        self.vs = np.array([p[1] for p in pts])

    def __call__(self, t):
        return float(np.interp(t, self.ts, self.vs))


def as_waveform(value):
    """Accept a number, a Waveform or any callable f(t)."""
    if isinstance(value, Waveform) or callable(value):
        return value
    return DC(value)
