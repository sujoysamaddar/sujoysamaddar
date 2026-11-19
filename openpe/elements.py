"""Power-circuit elements.

All elements are stamped into a Modified Nodal Analysis (MNA) system using
companion models (trapezoidal or backward-Euler).  Switching devices are
piecewise-linear (on = small resistance, off = large resistance), which is
what makes fixed-step simulation of switching converters fast: every distinct
combination of switch states gives one matrix, and each matrix is factorised
only once and then cached.

Sign convention: an element's current is the current flowing from its first
node to its second node *through the element* (passive sign convention), so
v*i > 0 means power absorbed.  For sources this means delivered power is
negative.
"""
from .waveforms import as_waveform

GROUND_NAMES = {"0", "gnd", "GND", "ground"}


def stamp_g(A, a, b, g):
    """Conductance g between matrix indices a and b (-1 = ground)."""
    if a >= 0:
        A[a, a] += g
    if b >= 0:
        A[b, b] += g
    if a >= 0 and b >= 0:
        A[a, b] -= g
        A[b, a] -= g


def stamp_i(rhs, a, b, i):
    """Ideal current source of value i flowing from node a to node b
    through the source (i.e. injected into node b)."""
    if a >= 0:
        rhs[a] -= i
    if b >= 0:
        rhs[b] += i


def _v(x, k):
    return x[k] if k >= 0 else 0.0


class Element:
    """Base class for power-circuit elements."""

    n_branch = 0          # extra MNA unknowns (branch currents, speeds, ...)
    is_switching = False  # True for devices with on/off state

    def __init__(self, name, *nodes):
        self.name = name
        self.nodes = [str(n) for n in nodes]
        self.idx = []
        self.br = []

    # -- hooks -----------------------------------------------------------
    def expand(self):
        """Return primitive elements this element is made of (default: itself)."""
        return [self]

    def init(self, ctx):
        pass

    def stamp_A(self, A, ctx):
        pass

    def stamp_b(self, rhs, ctx):
        pass

    def update(self, x, ctx):
        """Called once a time step has been accepted."""

    def current(self, x):
        return 0.0

    def voltage(self, x):
        return _v(x, self.idx[0]) - _v(x, self.idx[1])

    def probes(self, x):
        return {f"I({self.name})": self.current(x)}

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r}, {', '.join(self.nodes)})"


# ---------------------------------------------------------------------------
# Passive elements
# ---------------------------------------------------------------------------
class Resistor(Element):
    def __init__(self, name, a, b, r):
        super().__init__(name, a, b)
        self.r = float(r)
        if self.r <= 0:
            raise ValueError(f"{name}: resistance must be > 0")

    def stamp_A(self, A, ctx):
        stamp_g(A, self.idx[0], self.idx[1], 1.0 / self.r)

    def current(self, x):
        return self.voltage(x) / self.r


class Capacitor(Element):
    def __init__(self, name, a, b, c, ic=0.0):
        super().__init__(name, a, b)
        self.c, self.ic = float(c), float(ic)

    def init(self, ctx):
        self.v_prev, self.i_prev = self.ic, 0.0

    def _geq(self, ctx):
        return (2.0 if ctx.method == "trap" else 1.0) * self.c / ctx.dt

    def _ieq(self, ctx):
        g = self._geq(ctx)
        return g * self.v_prev + (self.i_prev if ctx.method == "trap" else 0.0)

    def stamp_A(self, A, ctx):
        stamp_g(A, self.idx[0], self.idx[1], self._geq(ctx))

    def stamp_b(self, rhs, ctx):
        stamp_i(rhs, self.idx[1], self.idx[0], self._ieq(ctx))

    def update(self, x, ctx):
        v = self.voltage(x)
        self.i_prev = self._geq(ctx) * v - self._ieq(ctx)
        self.v_prev = v

    def current(self, x):
        return self.i_prev  # valid after update()


class Inductor(Element):
    def __init__(self, name, a, b, l, ic=0.0):
        super().__init__(name, a, b)
        self.l, self.ic = float(l), float(ic)

    def init(self, ctx):
        self.v_prev, self.i_prev = 0.0, self.ic

    def _geq(self, ctx):
        return ctx.dt / ((2.0 if ctx.method == "trap" else 1.0) * self.l)

    def _ihist(self, ctx):
        if ctx.method == "trap":
            return self.i_prev + self._geq(ctx) * self.v_prev
        return self.i_prev

    def stamp_A(self, A, ctx):
        stamp_g(A, self.idx[0], self.idx[1], self._geq(ctx))

    def stamp_b(self, rhs, ctx):
        stamp_i(rhs, self.idx[0], self.idx[1], self._ihist(ctx))

    def update(self, x, ctx):
        v = self.voltage(x)
        self.i_prev = self._geq(ctx) * v + self._ihist(ctx)
        self.v_prev = v

    def current(self, x):
        return self.i_prev


class CoupledInductors(Element):
    """Two magnetically coupled inductors (L1: a-b, L2: c-d, coupling k).

    Modelled with two branch currents as unknowns:
        v1 = L1 di1/dt + M di2/dt,   v2 = M di1/dt + L2 di2/dt
    """
    n_branch = 2

    def __init__(self, name, a, b, c, d, l1, l2, k=0.99):
        super().__init__(name, a, b, c, d)
        self.l1, self.l2, self.k = float(l1), float(l2), float(k)
        self.m = self.k * (self.l1 * self.l2) ** 0.5

    def init(self, ctx):
        self.i1 = self.i2 = 0.0
        self.v1 = self.v2 = 0.0

    def stamp_A(self, A, ctx):
        a, b, c, d = self.idx
        k1, k2 = self.br
        f = (2.0 if ctx.method == "trap" else 1.0) / ctx.dt
        for n, s in ((a, 1), (b, -1)):
            if n >= 0:
                A[n, k1] += s
                A[k1, n] += s
        for n, s in ((c, 1), (d, -1)):
            if n >= 0:
                A[n, k2] += s
                A[k2, n] += s
        A[k1, k1] -= f * self.l1
        A[k1, k2] -= f * self.m
        A[k2, k1] -= f * self.m
        A[k2, k2] -= f * self.l2

    def stamp_b(self, rhs, ctx):
        k1, k2 = self.br
        f = (2.0 if ctx.method == "trap" else 1.0) / ctx.dt
        r1 = -f * (self.l1 * self.i1 + self.m * self.i2)
        r2 = -f * (self.m * self.i1 + self.l2 * self.i2)
        if ctx.method == "trap":
            r1 -= self.v1
            r2 -= self.v2
        rhs[k1] += r1
        rhs[k2] += r2

    def update(self, x, ctx):
        a, b, c, d = self.idx
        self.i1, self.i2 = x[self.br[0]], x[self.br[1]]
        self.v1 = _v(x, a) - _v(x, b)
        self.v2 = _v(x, c) - _v(x, d)

    def current(self, x):
        return self.i1

    def probes(self, x):
        return {f"I1({self.name})": self.i1, f"I2({self.name})": self.i2}


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
class VoltageSource(Element):
    """Independent or signal-controlled voltage source (+ node a, - node b).

    value  : number, Waveform or f(t)
    signal : name of a control signal; if given, V = gain * signal
             (PSIM-style link from the control circuit to the power circuit)
    """
    n_branch = 1

    def __init__(self, name, a, b, value=0.0, signal=None, gain=1.0):
        super().__init__(name, a, b)
        self.wave = as_waveform(value)
        self.signal, self.gain = signal, float(gain)

    def value(self, ctx):
        if self.signal is not None:
            return self.gain * ctx.signals[self.signal]
        return self.wave(ctx.t)

    def stamp_A(self, A, ctx):
        a, b = self.idx
        k = self.br[0]
        if a >= 0:
            A[a, k] += 1
            A[k, a] += 1
        if b >= 0:
            A[b, k] -= 1
            A[k, b] -= 1

    def stamp_b(self, rhs, ctx):
        rhs[self.br[0]] += self.value(ctx)

    def current(self, x):
        return x[self.br[0]]


class CurrentSource(Element):
    """Independent or signal-controlled current source; current flows from a
    to b through the source (out of node b into the external circuit)."""

    def __init__(self, name, a, b, value=0.0, signal=None, gain=1.0):
        super().__init__(name, a, b)
        self.wave = as_waveform(value)
        self.signal, self.gain = signal, float(gain)
        self._i = 0.0

    def value(self, ctx):
        if self.signal is not None:
            return self.gain * ctx.signals[self.signal]
        return self.wave(ctx.t)

    def stamp_b(self, rhs, ctx):
        self._i = self.value(ctx)
        stamp_i(rhs, self.idx[0], self.idx[1], self._i)

    def current(self, x):
        return self._i


class ThreePhaseSource(Element):
    """Balanced 3-phase Y-connected sine source: phases a, b, c and neutral n.
    vll_rms is the line-to-line RMS voltage."""

    def __init__(self, name, a, b, c, n, vll_rms, freq, phase=0.0):
        super().__init__(name, a, b, c, n)
        self.vll_rms, self.freq, self.phase = float(vll_rms), float(freq), float(phase)

    def expand(self):
        from .waveforms import Sine
        amp = self.vll_rms * (2.0 / 3.0) ** 0.5
        n = self.nodes[3]
        return [VoltageSource(f"{self.name}_{p}", node, n, Sine(amp, self.freq, self.phase + sh))
                for p, node, sh in zip("abc", self.nodes[:3], (0, -120, 120))]


# ---------------------------------------------------------------------------
# Switching devices
# ---------------------------------------------------------------------------
class _Switching(Element):
    is_switching = True

    def __init__(self, name, a, b, ron=1e-3, roff=1e6, vf=0.0):
        super().__init__(name, a, b)
        self.ron, self.roff, self.vf = float(ron), float(roff), float(vf)
        self.on = False
        self._i = 0.0

    def stamp_A(self, A, ctx):
        stamp_g(A, self.idx[0], self.idx[1], 1.0 / (self.ron if self.on else self.roff))

    def stamp_b(self, rhs, ctx):
        if self.on and self.vf:
            stamp_i(rhs, self.idx[1], self.idx[0], self.vf / self.ron)

    def _current(self, x):
        v = self.voltage(x)
        if self.on:
            return (v - self.vf) / self.ron
        return v / self.roff

    def set_gate(self, ctx):
        """Update state from control signals at the start of a step."""

    def violation(self, x, ctx):
        """How strongly (in volts) solution x contradicts the present state;
        0 if consistent.  Self-commutated devices override this."""
        return 0.0

    def check(self, x, ctx):
        """Flip state if inconsistent with solution x; return True if flipped."""
        if self.violation(x, ctx) > 0.0:
            self.on = not self.on
            return True
        return False

    def update(self, x, ctx):
        self._i = self._current(x)

    def current(self, x):
        return self._i

    def probes(self, x):
        return {f"I({self.name})": self._i}


class Diode(_Switching):
    """Ideal diode, anode a, cathode b, with optional forward drop vf."""

    def violation(self, x, ctx):
        if self.on:
            return max(0.0, -self._current(x) * self.ron - 1e-12)
        return max(0.0, self.voltage(x) - self.vf - 1e-9)


class Switch(_Switching):
    """Gate-controlled ideal switch (MOSFET/IGBT/GTO style).

    gate        : control signal name; device is on while signal > 0.5
    body_diode  : add an anti-parallel diode (b -> a), as in a MOSFET or an
                  IGBT module
    """

    def __init__(self, name, a, b, gate, ron=1e-3, roff=1e6, body_diode=False, vf_diode=0.0):
        super().__init__(name, a, b, ron, roff)
        self.gate = gate
        self.body_diode = body_diode
        self.vf_diode = vf_diode

    def expand(self):
        out = [self]
        if self.body_diode:
            out.append(Diode(f"{self.name}_D", self.nodes[1], self.nodes[0],
                             ron=self.ron, roff=self.roff, vf=self.vf_diode))
        return out

    def set_gate(self, ctx):
        self.on = ctx.signals[self.gate] > 0.5


class Thyristor(_Switching):
    """SCR: turns on when gated with positive anode-cathode voltage, stays on
    until its current falls to zero."""

    def __init__(self, name, a, b, gate, ron=1e-3, roff=1e6, vf=0.0):
        super().__init__(name, a, b, ron, roff, vf)
        self.gate = gate
        self._g = False

    def set_gate(self, ctx):
        self._g = ctx.signals[self.gate] > 0.5

    def violation(self, x, ctx):
        if self.on:
            return max(0.0, -self._current(x) * self.ron - 1e-12)
        if self._g:
            return max(0.0, self.voltage(x) - self.vf - 1e-9)
        return 0.0


# ---------------------------------------------------------------------------
# Magnetics
# ---------------------------------------------------------------------------
class IdealTransformer(Element):
    """Ideal 2-winding transformer, primary p1-p2, secondary s1-s2,
    n = Np/Ns.  Dot on p1 and s1."""
    n_branch = 1

    def __init__(self, name, p1, p2, s1, s2, n):
        super().__init__(name, p1, p2, s1, s2)
        self.n = float(n)

    def stamp_A(self, A, ctx):
        p1, p2, s1, s2 = self.idx
        k = self.br[0]
        for node, coef in ((p1, 1.0), (p2, -1.0), (s1, -self.n), (s2, self.n)):
            if node >= 0:
                A[node, k] += coef
                A[k, node] += coef

    def current(self, x):
        return x[self.br[0]]

    def probes(self, x):
        return {f"I({self.name})": x[self.br[0]], f"Is({self.name})": -self.n * x[self.br[0]]}


class Transformer(Element):
    """Practical transformer: ideal transformer + optional magnetising
    inductance (lm, on primary) and leakage inductances / winding resistances."""

    def __init__(self, name, p1, p2, s1, s2, n, lm=None, llk=None, rp=None, rs=None):
        super().__init__(name, p1, p2, s1, s2)
        self.n, self.lm, self.llk, self.rp, self.rs = n, lm, llk, rp, rs

    def expand(self):
        p1, p2, s1, s2 = self.nodes
        out = []
        pin = p1
        if self.rp:
            out.append(Resistor(f"{self.name}_Rp", pin, f"{self.name}_np1", self.rp))
            pin = f"{self.name}_np1"
        if self.llk:
            out.append(Inductor(f"{self.name}_Llk", pin, f"{self.name}_np2", self.llk))
            pin = f"{self.name}_np2"
        sin_ = s1
        if self.rs:
            out.append(Resistor(f"{self.name}_Rs", f"{self.name}_ns", s1, self.rs))
            sin_ = f"{self.name}_ns"
        if self.lm:
            out.append(Inductor(f"{self.name}_Lm", pin, p2, self.lm))
        out.append(IdealTransformer(self.name, pin, p2, sin_, s2, self.n))
        return out


# ---------------------------------------------------------------------------
# Electrical machine (motor drive module)
# ---------------------------------------------------------------------------
class DCMotor(Element):
    """Separately-excited / PM DC machine with mechanical load.

        v = Ra*i + La di/dt + Ke*w
        J dw/dt = Kt*i - B*w - T_load

    Both electrical current i and speed w are MNA unknowns, so the
    electro-mechanical coupling is solved fully implicitly.

    t_load : number, f(t, w) or a control-signal name.
    Probes: I(M), speed(M) [rad/s], rpm(M), Te(M) [N.m].
    """
    n_branch = 2

    def __init__(self, name, a, b, ra, la, ke, j, bf=0.0, t_load=0.0, kt=None, w0=0.0):
        super().__init__(name, a, b)
        self.ra, self.la, self.ke, self.j, self.bf = map(float, (ra, la, ke, j, bf))
        self.kt = float(kt) if kt is not None else self.ke
        self.t_load = t_load
        self.w0 = float(w0)

    def init(self, ctx):
        self.i_prev, self.w_prev, self.v_prev = 0.0, self.w0, 0.0
        self.tl_prev = self._tl(ctx)

    def _tl(self, ctx):
        tl = self.t_load
        if isinstance(tl, str):
            return ctx.signals.get(tl, 0.0)
        if callable(tl):
            return float(tl(ctx.t, getattr(self, "w_prev", self.w0)))
        return float(tl)

    def stamp_A(self, A, ctx):
        a, b = self.idx
        ki, kw = self.br
        trap = ctx.method == "trap"
        if a >= 0:
            A[a, ki] += 1
            A[ki, a] += 1
        if b >= 0:
            A[b, ki] -= 1
            A[ki, b] -= 1
        if trap:
            A[ki, ki] -= self.ra + 2 * self.la / ctx.dt
            A[kw, ki] -= self.kt / 2
            A[kw, kw] += self.j / ctx.dt + self.bf / 2
        else:
            A[ki, ki] -= self.ra + self.la / ctx.dt
            A[kw, ki] -= self.kt
            A[kw, kw] += self.j / ctx.dt + self.bf
        A[ki, kw] -= self.ke

    def stamp_b(self, rhs, ctx):
        ki, kw = self.br
        tl = self._tl(ctx)
        self._tl_now = tl
        if ctx.method == "trap":
            rhs[ki] += -self.v_prev + (self.ra - 2 * self.la / ctx.dt) * self.i_prev + self.ke * self.w_prev
            rhs[kw] += self.kt / 2 * self.i_prev + (self.j / ctx.dt - self.bf / 2) * self.w_prev \
                - 0.5 * (tl + self.tl_prev)
        else:
            rhs[ki] += -self.la / ctx.dt * self.i_prev
            rhs[kw] += self.j / ctx.dt * self.w_prev - tl

    def update(self, x, ctx):
        self.i_prev = x[self.br[0]]
        self.w_prev = x[self.br[1]]
        self.v_prev = self.voltage(x)
        self.tl_prev = self._tl_now

    def current(self, x):
        return self.i_prev

    def probes(self, x):
        import math
        return {
            f"I({self.name})": self.i_prev,
            f"speed({self.name})": self.w_prev,
            f"rpm({self.name})": self.w_prev * 60 / (2 * math.pi),
            f"Te({self.name})": self.kt * self.i_prev,
        }
