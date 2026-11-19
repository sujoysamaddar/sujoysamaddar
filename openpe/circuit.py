"""Circuit container: a power circuit (elements) plus a control circuit (blocks)."""
from . import changes as chg
from . import control as ctl
from . import devices as dv
from . import elements as el


class Circuit:
    """Holds power elements and control blocks.

    Elements and blocks can be added with ``add`` or the convenience methods
    (``c.R(...)``, ``c.L(...)``, ...) which mirror a schematic editor's parts
    palette.
    """

    def __init__(self, title="circuit"):
        self.title = title
        self.elements = []
        self.blocks = []
        self.changes = []          # scheduled run-time value changes (see openpe.changes)

    # ------------------------------------------------------------------
    def add(self, *items):
        for item in items:
            if isinstance(item, el.Element):
                if any(e.name == item.name for e in self.elements):
                    raise ValueError(f"duplicate element name {item.name!r}")
                self.elements.append(item)
            elif isinstance(item, ctl.Block):
                taken = {o for b in self.blocks for o in b.outputs}
                dup = taken.intersection(item.outputs)
                if dup:
                    raise ValueError(f"signal(s) {sorted(dup)} already driven by another block")
                self.blocks.append(item)
            else:
                raise TypeError(f"cannot add {item!r}")
        return items[0] if len(items) == 1 else items

    def element(self, name):
        for e in self.elements:
            if e.name == name:
                return e
        raise KeyError(name)

    def block(self, signal):
        for b in self.blocks:
            if signal in b.outputs:
                return b
        raise KeyError(signal)

    def copy(self):
        import copy
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # value changes
    def _target(self, name):
        return chg.find_target(self.elements, self.blocks, name)

    def set_value(self, target, value=None, **attrs):
        """Change a component / control-block value now (before simulating).

            c.set_value("R1", 4.7e3)              # the target's main value
            c.set_value("B1", kp=0.2, ki=150)     # named parameters
        """
        obj = self._target(target)
        if value is None and not attrs:
            raise ValueError("give a value or keyword parameters")
        if value is not None:
            chg.set_attr(obj, None, value)
        for k, v in attrs.items():
            chg.set_attr(obj, k, v)
        return self

    def get_value(self, target, attr=None):
        """Current value of a component / block parameter."""
        return chg.get_attr(self._target(target), attr)

    def at(self, t, target, value=None, **attrs):
        """Schedule a value change at simulation time ``t`` (seconds).

            c.at(5e-3, "RL", 2.0)           # load resistor becomes 2 ohm at 5 ms
            c.at(8e-3, "d", 0.5)            # control signal 'd' (a Constant) steps to 0.5
            c.at(0.3, "M1", t_load=4.0)     # motor load torque
        """
        if value is None and not attrs:
            raise ValueError("give a value or keyword parameters")
        self._target(target)               # fail early on a wrong name
        if value is not None:
            self.changes.append(chg.ValueChange(float(t), target, None, float(value)))
        for k, v in attrs.items():
            self.changes.append(chg.ValueChange(float(t), target, k, float(v)))
        return self

    def subcircuit(self, prefix, other, port_map):
        """Insert a copy of circuit ``other`` with its node/signal names
        prefixed by ``prefix.``.  ``port_map`` maps other's node names to
        nodes of this circuit (e.g. {'in': 'vbus', 'out': 'load'}).  Ground
        is always shared."""
        import copy
        sub = copy.deepcopy(other)

        def rn(n):
            if n in el.GROUND_NAMES:
                return n
            return port_map.get(n, f"{prefix}.{n}")

        def rs(s):
            return port_map.get(s, f"{prefix}.{s}")

        for e in sub.elements:
            e.name = f"{prefix}.{e.name}"
            e.nodes = [rn(n) for n in e.nodes]
            for attr in ("gate", "signal"):
                if getattr(e, attr, None):
                    setattr(e, attr, rs(getattr(e, attr)))
            self.add(e)
        for b in sub.blocks:
            b.outputs = [rs(o) for o in b.outputs]
            b.inputs = [rs(i) for i in b.inputs]
            for attr in ("n1", "n2"):
                if hasattr(b, attr):
                    setattr(b, attr, rn(getattr(b, attr)))
            if hasattr(b, "element"):
                b.element = f"{prefix}.{b.element}"
            if getattr(b, "netname", None):
                b.netname = f"{prefix}.{b.netname}"
            self.add(b)

    # ------------------------------------------------------------------
    # Palette shortcuts
    def R(self, name, a, b, r):
        return self.add(el.Resistor(name, a, b, r))

    def L(self, name, a, b, l, ic=0.0):
        return self.add(el.Inductor(name, a, b, l, ic))

    def C(self, name, a, b, c, ic=0.0):
        return self.add(el.Capacitor(name, a, b, c, ic))

    def V(self, name, a, b, value=0.0, **kw):
        return self.add(el.VoltageSource(name, a, b, value, **kw))

    def I(self, name, a, b, value=0.0, **kw):  # noqa: E743
        return self.add(el.CurrentSource(name, a, b, value, **kw))

    def D(self, name, a, b, **kw):
        return self.add(el.Diode(name, a, b, **kw))

    def S(self, name, a, b, gate, **kw):
        return self.add(el.Switch(name, a, b, gate, **kw))

    def SCR(self, name, a, b, gate, **kw):
        return self.add(el.Thyristor(name, a, b, gate, **kw))

    def T(self, name, p1, p2, s1, s2, n, **kw):
        return self.add(el.Transformer(name, p1, p2, s1, s2, n, **kw))

    # analog / electronic parts
    def Dpn(self, name, a, k, **kw):
        """Shockley (exponential) diode."""
        return self.add(dv.PNDiode(name, a, k, **kw))

    def Z(self, name, a, k, bv, **kw):
        return self.add(dv.Zener(name, a, k, bv, **kw))

    def Q(self, name, c, b, e, polarity="npn", **kw):
        return self.add(dv.BJT(name, c, b, e, polarity, **kw))

    def M(self, name, d, g, s, polarity="nmos", **kw):
        return self.add(dv.MOSFET(name, d, g, s, polarity, **kw))

    def E(self, name, op, on, ip, inn, gain):
        return self.add(dv.VCVS(name, op, on, ip, inn, gain))

    def G(self, name, op, on, ip, inn, gm):
        return self.add(dv.VCCS(name, op, on, ip, inn, gm))

    def F(self, name, op, on, ctrl, gain):
        return self.add(dv.CCCS(name, op, on, ctrl, gain))

    def H(self, name, op, on, ctrl, r):
        return self.add(dv.CCVS(name, op, on, ctrl, r))

    def opamp(self, name, inp, inn, out, **kw):
        return self.add(dv.OpAmp(name, inp, inn, out, **kw))

    # ------------------------------------------------------------------
    # analyses
    def simulate(self, dt, t_end, **kw):
        """Transient analysis (.tran)."""
        from .simulator import Simulator
        return Simulator(self, dt, t_end, **kw).run()

    def op(self, **kw):
        """DC operating point (.op)."""
        from .simulator import Simulator
        return Simulator(self, 1e-6, 0.0, **kw).operating_point()

    def dc(self, source, values, **kw):
        """DC sweep (.dc) of an independent source."""
        from .simulator import Simulator
        return Simulator(self, 1e-6, 0.0, **kw).dc_sweep(source, values)

    def ac(self, freqs, source=None, **kw):
        """AC small-signal analysis (.ac)."""
        from .simulator import Simulator
        return Simulator(self, 1e-6, 0.0, **kw).ac_analysis(freqs, source)

    def __repr__(self):
        return f"Circuit({self.title!r}, {len(self.elements)} elements, {len(self.blocks)} blocks)"
