"""Control-circuit blocks.

The control circuit is a signal-flow graph evaluated once per time step,
after the power circuit has been solved (as in PSIM, the power circuit sees
control outputs from the previous step).  Each block reads named input
signals and writes one or more named output signals.

Blocks with ``feedthrough = False`` (e.g. UnitDelay) break algebraic loops.
"""
import math

import numpy as np


class Block:
    feedthrough = True
    netname = None            # name used in a netlist ('B4'), set by the parser

    def __init__(self, out, inputs=()):
        self.outputs = [out] if isinstance(out, str) else list(out)
        self.inputs = [inputs] if isinstance(inputs, str) else list(inputs)

    @property
    def name(self):
        return self.outputs[0]

    def init(self, ctx):
        pass

    def step(self, u, ctx):
        """Return output value(s) for inputs u at time ctx.t."""
        raise NotImplementedError

    # for non-feedthrough blocks
    def output(self, ctx):
        raise NotImplementedError

    def update(self, u, ctx):
        pass

    def __repr__(self):
        return f"{type(self).__name__}({self.outputs} <- {self.inputs})"


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
class Constant(Block):
    def __init__(self, out, value):
        super().__init__(out)
        self.value = float(value)

    def step(self, u, ctx):
        return self.value


class SineWave(Block):
    def __init__(self, out, amp=1.0, freq=50.0, phase=0.0, offset=0.0):
        super().__init__(out)
        self.amp, self.freq, self.phase, self.offset = float(amp), float(freq), float(phase), float(offset)

    def step(self, u, ctx):
        return self.offset + self.amp * math.sin(2 * math.pi * self.freq * ctx.t + math.radians(self.phase))


def _tc(ctx):
    """Carrier sampling time.  A gate computed at time t drives the switches
    over the interval (t, t+dt]; sampling the carrier at the interval's
    midpoint removes the systematic duty-cycle bias of naive sampling."""
    return ctx.t + 0.5 * ctx.dt


class Triangle(Block):
    """Triangular carrier between vmin and vmax (starts at vmin)."""

    def __init__(self, out, freq, vmin=0.0, vmax=1.0, phase=0.0):
        super().__init__(out)
        self.freq, self.vmin, self.vmax, self.phase = float(freq), float(vmin), float(vmax), float(phase)

    def step(self, u, ctx):
        frac = (_tc(ctx) * self.freq + self.phase / 360.0) % 1.0
        tri = 2 * frac if frac < 0.5 else 2 - 2 * frac
        return self.vmin + (self.vmax - self.vmin) * tri


class Sawtooth(Block):
    def __init__(self, out, freq, vmin=0.0, vmax=1.0):
        super().__init__(out)
        self.freq, self.vmin, self.vmax = float(freq), float(vmin), float(vmax)

    def step(self, u, ctx):
        return self.vmin + (self.vmax - self.vmin) * ((_tc(ctx) * self.freq) % 1.0)


class SquareWave(Block):
    def __init__(self, out, freq, duty=0.5, vmin=0.0, vmax=1.0, delay=0.0):
        super().__init__(out)
        self.freq, self.duty, self.vmin, self.vmax, self.delay = map(float, (freq, duty, vmin, vmax, delay))

    def step(self, u, ctx):
        t = _tc(ctx)
        if t < self.delay:
            return self.vmin
        return self.vmax if ((t - self.delay) * self.freq) % 1.0 < self.duty else self.vmin


class StepSignal(Block):
    def __init__(self, out, t_step, before=0.0, after=1.0):
        super().__init__(out)
        self.t_step, self.before, self.after = float(t_step), float(before), float(after)

    def step(self, u, ctx):
        return self.after if ctx.t >= self.t_step else self.before


class Ramp(Block):
    def __init__(self, out, slope, t_start=0.0, initial=0.0, final=None):
        super().__init__(out)
        self.slope, self.t_start, self.initial, self.final = float(slope), float(t_start), float(initial), final

    def step(self, u, ctx):
        y = self.initial + self.slope * max(0.0, ctx.t - self.t_start)
        if self.final is not None:
            y = min(y, self.final) if self.slope >= 0 else max(y, self.final)
        return y


# ---------------------------------------------------------------------------
# Sensors: link from power circuit to control circuit
# ---------------------------------------------------------------------------
class VoltageSensor(Block):
    """out = gain * (V(n1) - V(n2))."""

    def __init__(self, out, n1, n2="0", gain=1.0):
        super().__init__(out)
        self.n1, self.n2, self.gain = str(n1), str(n2), float(gain)

    def step(self, u, ctx):
        return self.gain * (ctx.node_voltage(self.n1) - ctx.node_voltage(self.n2))


class CurrentSensor(Block):
    """out = gain * current of a power element."""

    def __init__(self, out, element, gain=1.0):
        super().__init__(out)
        self.element, self.gain = element, float(gain)

    def step(self, u, ctx):
        return self.gain * ctx.probe(f"I({self.element})")


class ProbeSensor(Block):
    """Read any recorded power-circuit quantity, e.g. 'speed(M1)'."""

    def __init__(self, out, key, gain=1.0):
        super().__init__(out)
        self.key, self.gain = key, float(gain)

    def step(self, u, ctx):
        return self.gain * ctx.probe(self.key)


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------
class Gain(Block):
    def __init__(self, out, inp, k):
        super().__init__(out, inp)
        self.k = float(k)

    def step(self, u, ctx):
        return self.k * u[0]


class Sum(Block):
    """Weighted sum; signs like '+-' or a list of numbers."""

    def __init__(self, out, inputs, signs=None):
        super().__init__(out, inputs)
        if signs is None:
            signs = "+" * len(self.inputs)
        if isinstance(signs, str):
            signs = [1.0 if s == "+" else -1.0 for s in signs]
        if len(signs) != len(self.inputs):
            raise ValueError(f"Sum {out}: {len(signs)} signs for {len(self.inputs)} inputs")
        self.signs = [float(s) for s in signs]

    def step(self, u, ctx):
        return sum(s * v for s, v in zip(self.signs, u))


class Product(Block):
    def __init__(self, out, inputs):
        super().__init__(out, inputs)

    def step(self, u, ctx):
        p = 1.0
        for v in u:
            p *= v
        return p


class Divide(Block):
    def __init__(self, out, num, den, eps=1e-12):
        super().__init__(out, [num, den])
        self.eps = eps

    def step(self, u, ctx):
        d = u[1] if abs(u[1]) > self.eps else math.copysign(self.eps, u[1] or 1.0)
        return u[0] / d


class Abs(Block):
    def __init__(self, out, inp):
        super().__init__(out, inp)

    def step(self, u, ctx):
        return abs(u[0])


class Limiter(Block):
    def __init__(self, out, inp, lo, hi):
        super().__init__(out, inp)
        self.lo, self.hi = float(lo), float(hi)

    def step(self, u, ctx):
        return min(self.hi, max(self.lo, u[0]))


class RateLimiter(Block):
    def __init__(self, out, inp, rising, falling=None, ic=None):
        super().__init__(out, inp)
        self.rise = float(rising)
        self.fall = float(falling if falling is not None else rising)
        self.ic = ic

    def init(self, ctx):
        self.y = None if self.ic is None else float(self.ic)

    def step(self, u, ctx):
        if self.y is None:
            self.y = u[0]
        dy = u[0] - self.y
        dy = min(dy, self.rise * ctx.dt) if dy > 0 else max(dy, -self.fall * ctx.dt)
        self.y += dy
        return self.y


class Function(Block):
    """Arbitrary algebraic function: out = fn(*inputs) (or fn(t, *inputs)
    if with_time=True)."""

    def __init__(self, out, inputs, fn, with_time=False):
        super().__init__(out, inputs)
        self.fn, self.with_time = fn, with_time

    def step(self, u, ctx):
        return self.fn(ctx.t, *u) if self.with_time else self.fn(*u)


class PythonBlock(Block):
    """General-purpose user code block (equivalent of PSIM's C block).

    fn(t, dt, inputs, state) -> outputs   (scalar or sequence)
    ``state`` is a dict that persists between calls.
    """

    def __init__(self, outputs, inputs, fn, state=None):
        super().__init__(outputs, inputs)
        self.fn = fn
        self._state0 = dict(state or {})

    def init(self, ctx):
        self.state = dict(self._state0)

    def step(self, u, ctx):
        return self.fn(ctx.t, ctx.dt, list(u), self.state)


# ---------------------------------------------------------------------------
# Logic / comparators
# ---------------------------------------------------------------------------
class Comparator(Block):
    """1 if a > b else 0, with optional hysteresis band."""

    def __init__(self, out, a, b, hysteresis=0.0):
        super().__init__(out, [a, b])
        self.h = float(hysteresis)

    def init(self, ctx):
        self.y = 0.0

    def step(self, u, ctx):
        d = u[0] - u[1]
        if self.h == 0.0:
            self.y = 1.0 if d > 0 else 0.0
        elif d > self.h / 2:
            self.y = 1.0
        elif d < -self.h / 2:
            self.y = 0.0
        return self.y


class Logic(Block):
    """op: 'and', 'or', 'xor', 'not', 'nand', 'nor' (inputs > 0.5 are true)."""

    def __init__(self, out, inputs, op):
        super().__init__(out, inputs)
        self.op = op.lower()

    def step(self, u, ctx):
        b = [v > 0.5 for v in u]
        r = {
            "and": lambda: all(b), "or": lambda: any(b), "xor": lambda: sum(b) % 2 == 1,
            "not": lambda: not b[0], "nand": lambda: not all(b), "nor": lambda: not any(b),
        }[self.op]()
        return 1.0 if r else 0.0


class DeadTime(Block):
    """From one gate command produce complementary high/low-side gates with
    a turn-on delay of td seconds on each edge."""

    def __init__(self, out_high, out_low, inp, td):
        super().__init__([out_high, out_low], inp)
        self.td = float(td)

    def init(self, ctx):
        self.state = None
        self.t_change = -1e9

    def step(self, u, ctx):
        s = u[0] > 0.5
        if s != self.state:
            self.state, self.t_change = s, ctx.t
        ready = ctx.t - self.t_change >= self.td - 1e-15
        hi = 1.0 if (s and ready) else 0.0
        lo = 1.0 if ((not s) and ready) else 0.0
        return hi, lo


class PWM(Block):
    """Carrier-based PWM modulator.

    out = 1 when the duty/modulation input > carrier.
    carrier: 'triangle' (centre-aligned) or 'sawtooth' (edge-aligned).
    If out_comp is given a complementary output is produced too.
    """

    def __init__(self, out, duty, freq, carrier="triangle", vmin=0.0, vmax=1.0, out_comp=None, phase=0.0):
        outs = [out] if out_comp is None else [out, out_comp]
        super().__init__(outs, duty)
        self.freq, self.vmin, self.vmax, self.phase = float(freq), float(vmin), float(vmax), float(phase)
        self.carrier = carrier

    def step(self, u, ctx):
        frac = (_tc(ctx) * self.freq + self.phase / 360.0) % 1.0
        if self.carrier == "triangle":
            c = 2 * frac if frac < 0.5 else 2 - 2 * frac
        else:
            c = frac
        c = self.vmin + (self.vmax - self.vmin) * c
        g = 1.0 if u[0] > c else 0.0
        return g if len(self.outputs) == 1 else (g, 1.0 - g)


# ---------------------------------------------------------------------------
# Dynamic blocks
# ---------------------------------------------------------------------------
class Integrator(Block):
    """Trapezoidal integrator k*∫u dt with optional output clamping."""

    def __init__(self, out, inp, k=1.0, ic=0.0, lo=None, hi=None):
        super().__init__(out, inp)
        self.k, self.ic, self.lo, self.hi = float(k), float(ic), lo, hi

    def init(self, ctx):
        self.y, self.u_prev = self.ic, None

    def step(self, u, ctx):
        up = u[0] if self.u_prev is None else self.u_prev
        self.y += self.k * ctx.dt * 0.5 * (u[0] + up)
        if self.lo is not None:
            self.y = max(self.lo, self.y)
        if self.hi is not None:
            self.y = min(self.hi, self.y)
        self.u_prev = u[0]
        return self.y


class PI(Block):
    """PI controller kp*e + ki*∫e dt with output limits and anti-windup
    (integration is stopped while the output saturates)."""

    def __init__(self, out, inp, kp, ki, lo=-math.inf, hi=math.inf, ic=0.0):
        super().__init__(out, inp)
        self.kp, self.ki, self.lo, self.hi, self.ic = float(kp), float(ki), float(lo), float(hi), float(ic)

    def init(self, ctx):
        self.xi, self.e_prev = self.ic, None

    def step(self, u, ctx):
        e = u[0]
        ep = e if self.e_prev is None else self.e_prev
        xi_new = self.xi + self.ki * ctx.dt * 0.5 * (e + ep)
        y = self.kp * e + xi_new
        if y > self.hi:
            y = self.hi
            if e < 0:
                self.xi = xi_new
        elif y < self.lo:
            y = self.lo
            if e > 0:
                self.xi = xi_new
        else:
            self.xi = xi_new
        self.e_prev = e
        return y


class PID(Block):
    """PID with filtered derivative: kp*e + ki∫e + kd*s/(1+s*tf) e."""

    def __init__(self, out, inp, kp, ki, kd, tf=None, lo=-math.inf, hi=math.inf):
        super().__init__(out, inp)
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.tf, self.lo, self.hi = tf, float(lo), float(hi)

    def init(self, ctx):
        self.xi, self.e_prev, self.d = 0.0, None, 0.0
        self.tf_ = self.tf if self.tf is not None else 10 * ctx.dt

    def step(self, u, ctx):
        e = u[0]
        ep = e if self.e_prev is None else self.e_prev
        a = ctx.dt / (self.tf_ + ctx.dt)
        self.d = (1 - a) * self.d + a * self.kd * (e - ep) / ctx.dt
        xi_new = self.xi + self.ki * ctx.dt * 0.5 * (e + ep)
        y = self.kp * e + xi_new + self.d
        sat = min(self.hi, max(self.lo, y))
        if sat == y or (y > self.hi and e < 0) or (y < self.lo and e > 0):
            self.xi = xi_new
        self.e_prev = e
        return sat


def _tustin(num, den, dt):
    """Discretise H(s)=num/den (descending powers of s) with the bilinear
    transform. Returns (b, a) of H(z) with a[0] = 1."""
    num = np.atleast_1d(np.asarray(num, float))
    den = np.atleast_1d(np.asarray(den, float))
    n = len(den) - 1
    if len(num) - 1 > n:
        raise ValueError("transfer function must be proper")
    num = np.concatenate([np.zeros(n + 1 - len(num)), num])
    k = 2.0 / dt
    zm = np.poly1d([1.0, -1.0])  # (z - 1)
    zp = np.poly1d([1.0, 1.0])   # (z + 1)
    b = np.poly1d([0.0])
    a = np.poly1d([0.0])
    for i in range(n + 1):
        p = n - i  # power of s
        term = (k ** p) * (zm ** p) * (zp ** (n - p))
        b = b + num[i] * term
        a = a + den[i] * term
    b = np.concatenate([np.zeros(n + 1 - len(b.coeffs)), b.coeffs])
    a = np.concatenate([np.zeros(n + 1 - len(a.coeffs)), a.coeffs])
    return b / a[0], a / a[0]


class TransferFunction(Block):
    """Continuous transfer function num(s)/den(s), coefficients in
    descending powers of s, discretised with Tustin's method."""

    def __init__(self, out, inp, num, den):
        super().__init__(out, inp)
        self.num, self.den = list(num), list(den)

    def init(self, ctx):
        self.b, self.a = _tustin(self.num, self.den, ctx.dt)
        self.z = np.zeros(max(len(self.a), len(self.b)) - 1)

    def step(self, u, ctx):
        x = u[0]
        b, a, z = self.b, self.a, self.z
        y = b[0] * x + (z[0] if len(z) else 0.0)
        for i in range(len(z)):
            nxt = z[i + 1] if i + 1 < len(z) else 0.0
            z[i] = nxt + b[i + 1] * x - a[i + 1] * y
        return float(y)


class LowPass(TransferFunction):
    """First-order low-pass filter with cut-off fc (Hz)."""

    def __init__(self, out, inp, fc):
        wc = 2 * math.pi * float(fc)
        super().__init__(out, inp, [wc], [1.0, wc])


class UnitDelay(Block):
    """Output equals the input of the previous time step (breaks loops)."""
    feedthrough = False

    def __init__(self, out, inp, ic=0.0):
        super().__init__(out, inp)
        self.ic = float(ic)

    def init(self, ctx):
        self.y = self.ic

    def output(self, ctx):
        return self.y

    def update(self, u, ctx):
        self.y = u[0]


class SampleHold(Block):
    """Zero-order hold sampling at fs (Hz): models a digital controller's ADC."""

    def __init__(self, out, inp, fs, ic=0.0):
        super().__init__(out, inp)
        self.ts, self.ic = 1.0 / float(fs), float(ic)

    def init(self, ctx):
        self.y, self.next_t = self.ic, 0.0

    def step(self, u, ctx):
        if ctx.t + 1e-12 >= self.next_t:
            self.y = u[0]
            self.next_t += self.ts
            while self.next_t <= ctx.t:
                self.next_t += self.ts
        return self.y


class MovingAverage(Block):
    """Average of the input over one period of freq (e.g. a mean meter)."""

    def __init__(self, out, inp, freq):
        super().__init__(out, inp)
        self.freq = float(freq)

    def init(self, ctx):
        n = max(1, int(round(1.0 / (self.freq * ctx.dt))))
        self.buf = np.zeros(n)
        self.k, self.s, self.count = 0, 0.0, 0

    def step(self, u, ctx):
        self.s += u[0] - self.buf[self.k]
        self.buf[self.k] = u[0]
        self.k = (self.k + 1) % len(self.buf)
        self.count = min(self.count + 1, len(self.buf))
        return self.s / self.count


class RMSMeter(MovingAverage):
    """Running RMS over one period of freq."""

    def step(self, u, ctx):
        return math.sqrt(max(0.0, super().step([u[0] * u[0]], ctx)))


# ---------------------------------------------------------------------------
# Coordinate transforms (motor drives / grid inverters)
# ---------------------------------------------------------------------------
class ABCtoDQ(Block):
    """Amplitude-invariant Park transform. outputs (d, q, 0)."""

    def __init__(self, out_d, out_q, a, b, c, theta, out_0=None):
        outs = [out_d, out_q] + ([out_0] if out_0 else [])
        super().__init__(outs, [a, b, c, theta])

    def step(self, u, ctx):
        a, b, c, th = u
        k = 2.0 / 3.0
        d = k * (a * math.cos(th) + b * math.cos(th - 2 * math.pi / 3) + c * math.cos(th + 2 * math.pi / 3))
        q = -k * (a * math.sin(th) + b * math.sin(th - 2 * math.pi / 3) + c * math.sin(th + 2 * math.pi / 3))
        z = (a + b + c) / 3.0
        return (d, q, z) if len(self.outputs) == 3 else (d, q)


class DQtoABC(Block):
    """Inverse Park transform. outputs (a, b, c)."""

    def __init__(self, out_a, out_b, out_c, d, q, theta):
        super().__init__([out_a, out_b, out_c], [d, q, theta])

    def step(self, u, ctx):
        d, q, th = u
        return tuple(d * math.cos(th - s) - q * math.sin(th - s) for s in (0.0, 2 * math.pi / 3, -2 * math.pi / 3))


class PLL(Block):
    """Simple single-phase-input SRF-free PLL for a sine signal of nominal
    frequency f0: outputs (theta, freq). Uses a quadrature (sin*cos) phase
    detector followed by a PI loop filter and a 2*f0 notch via averaging."""

    def __init__(self, out_theta, out_freq, inp, f0, kp=None, ki=None):
        super().__init__([out_theta, out_freq], inp)
        self.f0 = float(f0)
        self.kp = kp if kp is not None else 2 * math.pi * self.f0 * 0.5
        self.ki = ki if ki is not None else (2 * math.pi * self.f0) ** 2 * 0.05

    def init(self, ctx):
        self.theta, self.xi = 0.0, 0.0
        n = max(1, int(round(1.0 / (2 * self.f0 * ctx.dt))))
        self.buf, self.k, self.s = np.zeros(n), 0, 0.0

    def step(self, u, ctx):
        err = u[0] * math.cos(self.theta)
        self.s += err - self.buf[self.k]
        self.buf[self.k] = err
        self.k = (self.k + 1) % len(self.buf)
        e = self.s / len(self.buf)
        self.xi += self.ki * e * ctx.dt
        w = 2 * math.pi * self.f0 + self.kp * e + self.xi
        self.theta = (self.theta + w * ctx.dt) % (2 * math.pi)
        return self.theta, w / (2 * math.pi)
