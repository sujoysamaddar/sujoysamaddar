"""Analog / electronic devices: nonlinear semiconductors (Shockley diode,
Zener, BJT, MOSFET), SPICE controlled sources (E, G, F, H) and an
operational amplifier with gain-bandwidth and output-rail saturation.

Nonlinear devices are solved with Newton-Raphson: at every iteration
``stamp_nl`` stamps the device's linearisation (small-signal conductances
plus an equivalent current source) about the present solution estimate.
Junction voltages are limited between iterations (SPICE ``pnjlim``) so
the exponential characteristics converge from any starting point.
"""
import math

from .elements import Capacitor, Element, Resistor, _v, stamp_g, stamp_i

K_BOLTZ = 1.380649e-23
Q_ELEC = 1.602176634e-19
GMIN = 1e-12


def thermal_voltage(temp_c=27.0):
    return K_BOLTZ * (temp_c + 273.15) / Q_ELEC


def pnjlim(vnew, vold, vt, vcrit):
    """SPICE junction-voltage limiting. Returns (limited_voltage, limited?)."""
    if vnew > vcrit and abs(vnew - vold) > 2 * vt:
        if vold > 0:
            arg = 1 + (vnew - vold) / vt
            vnew = vold + vt * math.log(arg) if arg > 0 else vcrit
        else:
            vnew = vt * math.log(vnew / vt)
        return vnew, True
    return vnew, False


def _exp(x):
    return math.exp(min(x, 200.0))


def _lin_current(A, rhs, node, const, terms):
    """Stamp a linearised current I = const + sum(coef * V[n]) flowing INTO
    the device at ``node`` (i.e. leaving the circuit node)."""
    if node < 0:
        return
    for n, g in terms:
        if n >= 0:
            A[node, n] += g
    rhs[node] -= const


# ---------------------------------------------------------------------------
# Diodes
# ---------------------------------------------------------------------------
class PNDiode(Element):
    """Shockley diode  i = Is (exp(v / (n Vt)) - 1)  with optional series
    resistance rs, junction capacitance cj and reverse breakdown bv (Zener).

    anode a, cathode k.  Probes: I(D), Vd(D).
    """
    is_nonlinear = True

    def __init__(self, name, a, k, is_=1e-14, n=1.0, rs=0.0, bv=None, ibv=1e-3, cj=0.0, temp=27.0):
        super().__init__(name, a, k)
        self.is_, self.n, self.rs, self.cj = float(is_), float(n), float(rs), float(cj)
        self.bv = None if bv is None else float(bv)
        self.ibv = float(ibv)
        self.vt = self.n * thermal_voltage(temp)
        self.vcrit = self.vt * math.log(self.vt / (math.sqrt(2) * self.is_))
        self.vcrit_z = self.vt * math.log(self.vt / (math.sqrt(2) * self.ibv))
        self.limited = False

    def expand(self):
        out = []
        a, k = self.nodes
        if self.rs:
            j = f"{self.name}#j"
            out.append(Resistor(f"{self.name}_rs", a, j, self.rs))
            self.nodes = [j, k]
        out.append(self)
        if self.cj:
            out.append(Capacitor(f"{self.name}_cj", self.nodes[0], k, self.cj))
        return out

    def init(self, ctx):
        self.vd_it = 0.0
        self._i = 0.0
        self._vd = 0.0

    def iv(self, vd):
        e = _exp(vd / self.vt)
        i = self.is_ * (e - 1.0) + GMIN * vd
        g = self.is_ / self.vt * e + GMIN
        if self.bv is not None:
            ez = _exp(-(vd + self.bv) / self.vt)
            i -= self.ibv * ez
            g += self.ibv / self.vt * ez
        return i, g

    def stamp_nl(self, A, rhs, x, ctx):
        vd, lim = pnjlim(self.voltage(x), self.vd_it, self.vt, self.vcrit)
        if self.bv is not None and not lim:
            vz, lim = pnjlim(-(vd + self.bv), -(self.vd_it + self.bv), self.vt, self.vcrit_z)
            if lim:
                vd = -vz - self.bv
        self.vd_it, self.limited = vd, lim
        i, g = self.iv(vd)
        a, k = self.idx
        stamp_g(A, a, k, g)
        stamp_i(rhs, a, k, i - g * vd)

    def update(self, x, ctx):
        self._vd = self.voltage(x)
        self._i = self.iv(self._vd)[0]
        self.vd_it = self._vd

    def current(self, x):
        return self._i

    def probes(self, x):
        return {f"I({self.name})": self._i, f"Vd({self.name})": self._vd}


def Zener(name, a, k, bv, ibv=1e-3, **kw):
    """Zener diode (anode a, cathode k) with breakdown voltage bv."""
    return PNDiode(name, a, k, bv=bv, ibv=ibv, **kw)


# ---------------------------------------------------------------------------
# Bipolar junction transistor
# ---------------------------------------------------------------------------
class BJT(Element):
    """Ebers-Moll (transport) bipolar transistor with Early effect.

    nodes: collector c, base b, emitter e;  polarity 'npn' or 'pnp'.
    Parameters: is_ (A), bf, br, vaf (Early voltage, V; None = infinite),
    cje / cjc (linear junction capacitances).
    Probes: I(Q)=Ic, Ib(Q), Vbe(Q), Vce(Q), P(Q), gm(Q).
    """
    is_nonlinear = True

    def __init__(self, name, c, b, e, polarity="npn", is_=1e-15, bf=100.0, br=1.0, vaf=None,
                 cje=0.0, cjc=0.0, temp=27.0):
        super().__init__(name, c, b, e)
        pol = polarity.lower()
        if pol not in ("npn", "pnp"):
            raise ValueError(f"{name}: polarity must be 'npn' or 'pnp'")
        self.p = 1.0 if pol == "npn" else -1.0
        self.is_, self.beta_f, self.beta_r = float(is_), float(bf), float(br)
        self.vaf = None if vaf in (None, 0) else float(vaf)
        self.cje, self.cjc = float(cje), float(cjc)
        self.vt = thermal_voltage(temp)
        self.vcrit = self.vt * math.log(self.vt / (math.sqrt(2) * self.is_))
        self.limited = False

    def expand(self):
        c, b, e = self.nodes
        out = [self]
        if self.cje:
            out.append(Capacitor(f"{self.name}_cje", b, e, self.cje))
        if self.cjc:
            out.append(Capacitor(f"{self.name}_cjc", b, c, self.cjc))
        return out

    def init(self, ctx):
        self.vbe_it = self.vbc_it = 0.0
        self._op = dict(ic=0.0, ib=0.0, vbe=0.0, vce=0.0, gm=0.0)

    def _model(self, vbe, vbc):
        ef, er = _exp(vbe / self.vt), _exp(vbc / self.vt)
        i_f, i_r = self.is_ * (ef - 1), self.is_ * (er - 1)
        gif, gir = self.is_ / self.vt * ef, self.is_ / self.vt * er
        q = 1.0 - vbc / self.vaf if self.vaf else 1.0
        ict = (i_f - i_r) * q
        ic = ict - i_r / self.beta_r
        ib = i_f / self.beta_f + i_r / self.beta_r + GMIN * (vbe + vbc)
        gc1 = gif * q
        gc2 = -gir * q - gir / self.beta_r - ((i_f - i_r) / self.vaf if self.vaf else 0.0)
        gb1 = gif / self.beta_f + GMIN
        gb2 = gir / self.beta_r + GMIN
        return ic, ib, gc1, gc2, gb1, gb2

    def _junctions(self, x):
        c, b, e = self.idx
        vb = _v(x, b)
        return self.p * (vb - _v(x, e)), self.p * (vb - _v(x, c))

    def stamp_nl(self, A, rhs, x, ctx):
        vbe_raw, vbc_raw = self._junctions(x)
        vbe, l1 = pnjlim(vbe_raw, self.vbe_it, self.vt, self.vcrit)
        vbc, l2 = pnjlim(vbc_raw, self.vbc_it, self.vt, self.vcrit)
        self.vbe_it, self.vbc_it, self.limited = vbe, vbc, (l1 or l2)
        ic, ib, gc1, gc2, gb1, gb2 = self._model(vbe, vbc)
        c, b, e = self.idx
        p = self.p
        kc = p * (ic - gc1 * vbe - gc2 * vbc)
        kb = p * (ib - gb1 * vbe - gb2 * vbc)
        # I = k + g1*(Vb - Ve) + g2*(Vb - Vc)
        _lin_current(A, rhs, c, kc, [(b, gc1 + gc2), (e, -gc1), (c, -gc2)])
        _lin_current(A, rhs, b, kb, [(b, gb1 + gb2), (e, -gb1), (c, -gb2)])
        _lin_current(A, rhs, e, -(kc + kb), [(b, -(gc1 + gc2 + gb1 + gb2)), (e, gc1 + gb1), (c, gc2 + gb2)])

    def update(self, x, ctx):
        vbe, vbc = self._junctions(x)
        self.vbe_it, self.vbc_it = vbe, vbc
        ic, ib, gc1, _, _, _ = self._model(vbe, vbc)
        c, b, e = self.idx
        self._op = dict(ic=self.p * ic, ib=self.p * ib, vbe=self.p * vbe,
                        vce=_v(x, c) - _v(x, e), gm=gc1)

    def current(self, x):
        return self._op["ic"]

    def probes(self, x):
        o, n = self._op, self.name
        return {f"I({n})": o["ic"], f"Ib({n})": o["ib"], f"Vbe({n})": o["vbe"], f"Vce({n})": o["vce"],
                f"P({n})": o["vce"] * o["ic"] + o["vbe"] * o["ib"], f"gm({n})": o["gm"]}


# ---------------------------------------------------------------------------
# MOSFET (level 1, Shichman-Hodges)
# ---------------------------------------------------------------------------
class MOSFET(Element):
    """Level-1 (square-law) MOSFET, body tied to source.

    nodes: drain d, gate g, source s;  polarity 'nmos' or 'pmos'.
    vto  : threshold voltage (SPICE sign convention: negative for PMOS)
    kp   : transconductance parameter (A/V^2); beta = kp * w / l
    lam  : channel-length modulation (1/V)
    Probes: I(M)=Id (into drain), Vgs(M), Vds(M), gm(M), gds(M), P(M), region(M)
    (0 = cut-off, 1 = triode, 2 = saturation).
    """
    is_nonlinear = True

    def __init__(self, name, d, g, s, polarity="nmos", vto=None, kp=2e-5, w=1.0, l=1.0, lam=0.0,
                 cgs=0.0, cgd=0.0):
        super().__init__(name, d, g, s)
        pol = polarity.lower()
        if pol not in ("nmos", "pmos"):
            raise ValueError(f"{name}: polarity must be 'nmos' or 'pmos'")
        self.p = 1.0 if pol == "nmos" else -1.0
        if vto is None:
            vto = 1.0 * self.p
        self.vth = self.p * float(vto)
        self.beta = float(kp) * float(w) / float(l)
        self.lam = float(lam)
        self.cgs, self.cgd = float(cgs), float(cgd)
        self.limited = False

    def expand(self):
        d, g, s = self.nodes
        out = [self]
        if self.cgs:
            out.append(Capacitor(f"{self.name}_cgs", g, s, self.cgs))
        if self.cgd:
            out.append(Capacitor(f"{self.name}_cgd", g, d, self.cgd))
        return out

    def init(self, ctx):
        self.vgs_it = self.vds_it = 0.0
        self._op = dict(id=0.0, vgs=0.0, vds=0.0, gm=0.0, gds=0.0, region=0)

    def _fwd(self, vgs, vds):
        vov = vgs - self.vth
        if vov <= 0:
            return 0.0, 0.0, 0.0, 0
        cl = 1 + self.lam * vds
        if vds < vov:
            core = vov * vds - 0.5 * vds * vds
            return (self.beta * core * cl, self.beta * vds * cl,
                    self.beta * (vov - vds) * cl + self.beta * core * self.lam, 1)
        return (0.5 * self.beta * vov * vov * cl, self.beta * vov * cl,
                0.5 * self.beta * vov * vov * self.lam, 2)

    def model(self, vgs, vds):
        """Drain current (d->s, NMOS orientation) and its derivatives."""
        if vds >= 0:
            i, gm, gds, reg = self._fwd(vgs, vds)
        else:  # source and drain exchange roles
            i2, gm2, gds2, reg = self._fwd(vgs - vds, -vds)
            i, gm, gds = -i2, -gm2, gm2 + gds2
        return i + GMIN * vds, gm, gds + GMIN, reg

    def _terminal(self, x):
        d, g, s = self.idx
        vs = _v(x, s)
        return self.p * (_v(x, g) - vs), self.p * (_v(x, d) - vs)

    def stamp_nl(self, A, rhs, x, ctx):
        vgs_raw, vds_raw = self._terminal(x)
        # simple FET step limiting
        dmax_g = 0.5 + 0.5 * abs(self.vgs_it - self.vth)
        dmax_d = 1.0 + 0.5 * abs(self.vds_it)
        vgs = self.vgs_it + max(-dmax_g, min(dmax_g, vgs_raw - self.vgs_it))
        vds = self.vds_it + max(-dmax_d, min(dmax_d, vds_raw - self.vds_it))
        self.limited = abs(vgs - vgs_raw) > 1e-12 or abs(vds - vds_raw) > 1e-12
        self.vgs_it, self.vds_it = vgs, vds
        i, gm, gds, _ = self.model(vgs, vds)
        d, g, s = self.idx
        k = self.p * (i - gm * vgs - gds * vds)
        # Id(into drain) = k + gm*(Vg - Vs) + gds*(Vd - Vs)
        _lin_current(A, rhs, d, k, [(g, gm), (d, gds), (s, -(gm + gds))])
        _lin_current(A, rhs, s, -k, [(g, -gm), (d, -gds), (s, gm + gds)])

    def update(self, x, ctx):
        vgs, vds = self._terminal(x)
        self.vgs_it, self.vds_it = vgs, vds
        i, gm, gds, reg = self.model(vgs, vds)
        self._op = dict(id=self.p * i, vgs=self.p * vgs, vds=self.p * vds, gm=gm, gds=gds, region=reg)

    def current(self, x):
        return self._op["id"]

    def probes(self, x):
        o, n = self._op, self.name
        return {f"I({n})": o["id"], f"Vgs({n})": o["vgs"], f"Vds({n})": o["vds"], f"gm({n})": o["gm"],
                f"gds({n})": o["gds"], f"P({n})": o["vds"] * o["id"], f"region({n})": float(o["region"])}


# ---------------------------------------------------------------------------
# Controlled sources (SPICE E, G, F, H)
# ---------------------------------------------------------------------------
def _stamp_vsrc_kcl(A, k, a, b):
    if a >= 0:
        A[a, k] += 1
        A[k, a] += 1
    if b >= 0:
        A[b, k] -= 1
        A[k, b] -= 1


class VCVS(Element):
    """E: V(o+, o-) = gain * V(i+, i-)."""
    n_branch = 1

    def __init__(self, name, op, on, ip, inn, gain):
        super().__init__(name, op, on, ip, inn)
        self.gain = float(gain)

    def stamp_A(self, A, ctx):
        op, on, ip, inn = self.idx
        k = self.br[0]
        _stamp_vsrc_kcl(A, k, op, on)
        if ip >= 0:
            A[k, ip] -= self.gain
        if inn >= 0:
            A[k, inn] += self.gain

    def current(self, x):
        return x[self.br[0]]


class VCCS(Element):
    """G: current gm * V(i+, i-) flows from o+ through the source to o-."""

    def __init__(self, name, op, on, ip, inn, gm):
        super().__init__(name, op, on, ip, inn)
        self.gm = float(gm)
        self._i = 0.0

    def stamp_A(self, A, ctx):
        op, on, ip, inn = self.idx
        for row, sr in ((op, 1.0), (on, -1.0)):
            if row < 0:
                continue
            if ip >= 0:
                A[row, ip] += sr * self.gm
            if inn >= 0:
                A[row, inn] -= sr * self.gm

    def update(self, x, ctx):
        _, _, ip, inn = self.idx
        self._i = self.gm * (_v(x, ip) - _v(x, inn))

    def current(self, x):
        return self._i


class CCCS(Element):
    """F: current gain * I(ctrl) flows from o+ through the source to o-.
    ``ctrl`` names an element with a branch current (voltage source,
    inductor, E/H source...)."""

    def __init__(self, name, op, on, ctrl, gain):
        super().__init__(name, op, on)
        self.ctrl, self.gain = ctrl, float(gain)
        self._i = 0.0

    def resolve(self, lookup):
        el = lookup(self.ctrl)
        if not el.br:
            raise ValueError(f"{self.name}: controlling element {self.ctrl!r} has no branch current")
        self.kc = el.br[0]

    def stamp_A(self, A, ctx):
        op, on = self.idx
        if op >= 0:
            A[op, self.kc] += self.gain
        if on >= 0:
            A[on, self.kc] -= self.gain

    def update(self, x, ctx):
        self._i = self.gain * x[self.kc]

    def current(self, x):
        return self._i


class CCVS(Element):
    """H: V(o+, o-) = r * I(ctrl)."""
    n_branch = 1

    def __init__(self, name, op, on, ctrl, r):
        super().__init__(name, op, on)
        self.ctrl, self.r = ctrl, float(r)

    def resolve(self, lookup):
        el = lookup(self.ctrl)
        if not el.br:
            raise ValueError(f"{self.name}: controlling element {self.ctrl!r} has no branch current")
        self.kc = el.br[0]

    def stamp_A(self, A, ctx):
        op, on = self.idx
        k = self.br[0]
        _stamp_vsrc_kcl(A, k, op, on)
        A[k, self.kc] -= self.r

    def current(self, x):
        return x[self.br[0]]


class _SaturatingVCVS(Element):
    """V(o+, o-) = sat(gain * V(i+, i-)) with smooth (tanh) rail limiting."""
    n_branch = 1
    is_nonlinear = True

    def __init__(self, name, op, on, ip, inn, gain, vmin, vmax):
        super().__init__(name, op, on, ip, inn)
        self.gain, self.vmin, self.vmax = float(gain), float(vmin), float(vmax)
        self.m = 0.05 * (self.vmax - self.vmin)
        self.limited = False

    def _sat(self, y):
        hi, lo, m = self.vmax - self.m, self.vmin + self.m, self.m
        if y > hi:
            t = math.tanh((y - hi) / m)
            return hi + m * t, 1 - t * t
        if y < lo:
            t = math.tanh((y - lo) / m)
            return lo + m * t, 1 - t * t
        return y, 1.0

    def stamp_A(self, A, ctx):
        op, on, _, _ = self.idx
        _stamp_vsrc_kcl(A, self.br[0], op, on)

    def init(self, ctx):
        self.y_it = 0.0

    def stamp_nl(self, A, rhs, x, ctx):
        _, _, ip, inn = self.idx
        k = self.br[0]
        u = _v(x, ip) - _v(x, inn)
        # limit the change of the internal (pre-saturation) output between
        # Newton iterations so the iteration cannot jump across the knee;
        # deep in saturation the exact value no longer matters
        span = self.vmax - self.vmin
        y = min(self.vmax + span, max(self.vmin - span, self.gain * u))
        step = 0.5 * span
        y_lim = self.y_it + max(-step, min(step, y - self.y_it))
        self.limited = abs(y_lim - y) > 1e-12 * span
        self.y_it = y_lim
        u = y_lim / self.gain
        f, df = self._sat(y_lim)
        fp = self.gain * df
        if ip >= 0:
            A[k, ip] -= fp
        if inn >= 0:
            A[k, inn] += fp
        rhs[k] += f - fp * u

    def current(self, x):
        return x[self.br[0]]


class OpAmp(Element):
    """Operational amplifier (output referred to ground).

    gain : open-loop DC gain
    gbw  : gain-bandwidth product in Hz (None = infinite bandwidth)
    vmax, vmin : output rails (None = no saturation)
    rout : output resistance
    """

    def __init__(self, name, inp, inn, out, gain=1e5, gbw=None, vmax=None, vmin=None, rout=0.0):
        super().__init__(name, inp, inn, out)
        self.gain, self.gbw, self.rout = float(gain), gbw, float(rout)
        self.vmax, self.vmin = vmax, vmin
        if (vmax is None) != (vmin is None):
            raise ValueError(f"{name}: give both vmax and vmin, or neither")

    def expand(self):
        inp, inn, out = self.nodes
        n = self.name
        prims = []
        drv = out if not self.rout else f"{n}#o"
        if self.gbw:
            x = f"{n}#p"
            gm = 1e-3
            r = self.gain / gm
            c = gm / (2 * math.pi * float(self.gbw))
            prims += [VCCS(f"{n}_gm", "0", x, inp, inn, gm), Resistor(f"{n}_rp", x, "0", r),
                      Capacitor(f"{n}_cp", x, "0", c)]
            src_p, src_n, g = x, "0", 1.0
        else:
            src_p, src_n, g = inp, inn, self.gain
        if self.vmax is not None:
            prims.append(_SaturatingVCVS(n, drv, "0", src_p, src_n, g, self.vmin, self.vmax))
        else:
            prims.append(VCVS(n, drv, "0", src_p, src_n, g))
        if self.rout:
            prims.append(Resistor(f"{n}_ro", drv, out, self.rout))
        return prims
