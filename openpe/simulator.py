"""Fixed time-step transient simulator.

Per time step:
  1. Gate-controlled switches take their state from control signals
     computed in the previous step.
  2. The power circuit (MNA) is solved; diodes/thyristors are iterated until
     their states are consistent with the solution.
  3. If the switch topology changed and the method is trapezoidal, the step
     is recomputed with backward Euler to suppress the numerical ringing the
     trapezoidal rule produces at discontinuities.
  4. Element histories are updated, sensors are read and the control
     circuit is evaluated.

Every distinct switch-state combination produces a matrix that is inverted
once and cached, so a converter that cycles through a handful of topologies
costs a single matrix-vector product per step.
"""
import time
import warnings
from collections import OrderedDict

import numpy as np

from . import elements as el
from .results import Results, _finish, _subplots


class _Ctx:
    """Simulation context handed to every element and block."""

    def __init__(self, sim):
        self._sim = sim
        self.t = 0.0
        self.dt = sim.dt
        self.method = sim.method
        self.signals = {}
        self.x = None
        self._probes = {}
        self.source_scale = 1.0

    def node_voltage(self, node):
        if node in el.GROUND_NAMES:
            return 0.0
        return float(self.x[self._sim.node_index[node]])

    def probe(self, key):
        return self._probes[key]


class Simulator:
    def __init__(self, circuit, dt, t_end, method="trap", record_every=1,
                 t_record_start=0.0, max_iter=50, damp_switching=True,
                 cache_size=4096, record=None, gmin=1e-12, initial="ic",
                 nr_max=100, reltol=1e-4, vntol=1e-6, abstol=1e-9):
        if method not in ("trap", "be"):
            raise ValueError("method must be 'trap' or 'be'")
        self.circuit = circuit.copy()
        self.dt, self.t_end = float(dt), float(t_end)
        self.method = method
        self.record_every = max(1, int(record_every))
        self.t_record_start = float(t_record_start)
        self.max_iter = max_iter
        self.damp_switching = damp_switching
        self.cache_size = cache_size
        self.record_filter = record
        self.gmin = gmin
        if initial not in ("ic", "op"):
            raise ValueError("initial must be 'ic' (initial conditions) or 'op' (DC operating point)")
        self.initial = initial
        self.nr_max, self.reltol = nr_max, reltol
        self.vntol, self.abstol = vntol, abstol
        self._compile()

    # ------------------------------------------------------------------
    def _compile(self):
        def flatten(e, depth=0):
            if depth > 20:
                raise RecursionError(f"element expansion too deep at {e!r}")
            out = []
            for s in e.expand():
                out += [s] if s is e else flatten(s, depth + 1)
            return out

        prims = [p for e in self.circuit.elements for p in flatten(e)]
        names = [e.name for e in prims]
        if len(set(names)) != len(names):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate element names after expansion: {dup}")
        self.elements = prims

        nodes = []
        for e in prims:
            for n in e.nodes:
                if n not in el.GROUND_NAMES and n not in nodes:
                    nodes.append(n)
        if not any(n in el.GROUND_NAMES for e in prims for n in e.nodes):
            raise ValueError("circuit has no ground node ('0' or 'gnd')")
        self.nodes = nodes
        self.node_index = {n: i for i, n in enumerate(nodes)}
        k = len(nodes)
        for e in prims:
            e.idx = [-1 if n in el.GROUND_NAMES else self.node_index[n] for n in e.nodes]
            e.br = list(range(k, k + e.n_branch))
            k += e.n_branch
        self.size = k
        # Newton absolute tolerance: volts for node unknowns, amps for branches
        self._atol = np.full(k, self.abstol)
        self._atol[:len(nodes)] = self.vntol
        by_name = {e.name: e for e in prims}

        def lookup(name):
            if name not in by_name:
                raise ValueError(f"unknown controlling element {name!r}")
            return by_name[name]

        for e in prims:
            e.resolve(lookup)
        self.nonlinear = [e for e in prims if e.is_nonlinear]
        self.switching = [e for e in prims if e.is_switching]
        self.gated = [e for e in self.switching if hasattr(e, "gate")]
        self.self_commutated = [e for e in self.switching if type(e).violation is not el._Switching.violation]

        self.blocks = self._order_blocks(self.circuit.blocks)
        from .changes import resolve_changes
        self._events = resolve_changes(self.circuit.changes, prims, self.circuit.blocks)
        self.direct = [b for b in self.blocks if b.feedthrough]
        self.delayed = [b for b in self.blocks if not b.feedthrough]

        driven = {o for b in self.blocks for o in b.outputs}
        for e in prims:
            for attr in ("gate", "signal"):
                s = getattr(e, attr, None)
                if s is not None and s not in driven:
                    raise ValueError(f"{e.name}: control signal {s!r} is not produced by any block")
            if isinstance(getattr(e, "t_load", None), str) and e.t_load not in driven:
                raise ValueError(f"{e.name}: load-torque signal {e.t_load!r} is not produced by any block")

    @staticmethod
    def _order_blocks(blocks):
        producer = {}
        for b in blocks:
            for o in b.outputs:
                producer[o] = b
        for b in blocks:
            for i in b.inputs:
                if i not in producer:
                    raise ValueError(f"block {b!r}: input signal {i!r} is not produced by any block")
        order, state = [], {}

        def visit(b, path):
            s = state.get(id(b))
            if s == 2:
                return
            if s == 1:
                cyc = " -> ".join(p.name for p in path + [b])
                raise ValueError(f"algebraic loop in control circuit: {cyc}. Insert a UnitDelay.")
            state[id(b)] = 1
            if b.feedthrough:
                for i in b.inputs:
                    visit(producer[i], path + [b])
            state[id(b)] = 2
            order.append(b)

        for b in blocks:
            visit(b, [])
        return order

    # ------------------------------------------------------------------
    def _new_stats(self):
        self._cache = OrderedDict()
        self.stats = {"steps": 0, "factorizations": 0, "nonconverged": 0, "damped_steps": 0,
                      "ambiguous_steps": 0, "newton_iterations": 0, "substepped": 0, "value_changes": 0}

    def _matrix(self, key, ctx, want_inv):
        """Linear part of the MNA matrix for a switch-state key (cached),
        plus its inverse when the circuit has no nonlinear devices."""
        entry = self._cache.get(key)
        if entry is None:
            A = np.zeros((self.size, self.size))
            for e in self.elements:
                e.stamp_A(A, ctx)
            for i in range(len(self.nodes)):
                A[i, i] += self.gmin
            entry = [A, None]
            self._cache[key] = entry
            self.stats["factorizations"] += 1
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        if want_inv and entry[1] is None:
            try:
                entry[1] = np.linalg.inv(entry[0])
            except np.linalg.LinAlgError as exc:
                raise RuntimeError(self._singular_msg(ctx)) from exc
        return entry

    @staticmethod
    def _singular_msg(ctx):
        msg = (f"singular circuit matrix at t={ctx.t:g}s (floating node, loop of voltage sources "
               f"or inductor-only cut-set?)")
        if ctx.method == "dc":
            msg += (". In DC analysis inductors are short circuits and capacitors open circuits: "
                    "check for a voltage source shorted by an inductor or transformer winding, or a "
                    "node connected only through capacitors")
        return msg

    def _newton(self, A_lin, rhs_lin, ctx):
        """Newton-Raphson on the nonlinear devices. Returns (x, converged)."""
        x = self._x_guess
        hist = []          # recent iterates, for limit-cycle detection
        for _ in range(self.nr_max):
            A = A_lin.copy()
            rhs = rhs_lin.copy()
            for e in self.nonlinear:
                e.limited = False
                e.stamp_nl(A, rhs, x, ctx)
            try:
                xn = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError(self._singular_msg(ctx)) from exc
            self.stats["newton_iterations"] += 1
            limited = any(e.limited for e in self.nonlinear)
            tol = self.reltol * np.maximum(np.abs(xn), np.abs(x)) + self._atol
            done = not limited and np.all(np.abs(xn - x) <= tol)
            if not done and not limited:
                for k, xh in enumerate(reversed(hist)):
                    if np.all(np.abs(xn - xh) <= tol):
                        # round-off limit cycle of length k+2 (ill-conditioned
                        # matrix, e.g. a galvanically isolated winding with all
                        # diodes off): accept the mean of the cycle
                        cyc = hist[len(hist) - 1 - k:] + [x]
                        self.stats["cycle_accepted"] = self.stats.get("cycle_accepted", 0) + 1
                        return np.mean(cyc, axis=0), True
            hist = (hist + [x])[-5:]
            x = xn
            if done:
                return x, True
        return x, False

    def _point(self, key, ctx):
        """Solve the circuit for fixed switch states."""
        entry = self._matrix(key, ctx, want_inv=not self.nonlinear)
        rhs = np.zeros(self.size)
        for e in self.elements:
            e.stamp_b(rhs, ctx)
        if not self.nonlinear:
            A, inv = entry
            x = inv @ rhs
            # one step of iterative refinement: the cached explicit inverse
            # loses accuracy on ill-conditioned circuits (floating islands
            # tied by large resistances), and companion-model currents
            # (C/dt * dv) amplify that error
            x += inv @ (rhs - A @ x)
            return x, True
        return self._newton(entry[0], rhs, ctx)

    def _key(self, ctx):
        return (ctx.method, ctx.dt) + tuple(e.on for e in self.switching)

    def _solve(self, ctx):
        """Solve one step with Newton (nonlinear devices) and diode /
        thyristor state iteration. Returns (x, key, converged)."""
        seen = set()
        best = None
        for it in range(self.max_iter):
            key = self._key(ctx)
            x, nr_ok = self._point(key, ctx)
            viol = [(e.violation(x, ctx), k) for k, e in enumerate(self.self_commutated)]
            viol = [vk for vk in viol if vk[0] > 0.0]
            if not viol:
                return x, key, nr_ok
            worst = max(viol)[0]
            if best is None or worst < best[0]:
                best = (worst, x, key, [e.on for e in self.switching], nr_ok)
            if key in seen and it >= 2 * len(self.self_commutated) + 4:
                # No state is consistent for the whole step: a device current
                # crosses zero inside it (e.g. discontinuous conduction).
                # Accept the least-violating state (error <= one time step).
                _, x, key, states, nr_ok = best
                for e, st in zip(self.switching, states):
                    e.on = st
                self.stats["ambiguous_steps"] += 1
                return x, key, nr_ok
            if key in seen or it >= 8:
                # cycling: flip only the most strongly violated device
                self.self_commutated[max(viol)[1]].check(x, ctx)
            else:
                for _, k in viol:
                    self.self_commutated[k].check(x, ctx)
            seen.add(key)
        return x, key, False

    def _substeps(self, ctx):
        """Newton failed for a full step: retry it as 2, 4, ... 32 backward-
        Euler substeps (element states are restored between attempts).
        Returns (x, key, converged, element_states_updated)."""
        t_end, dt = ctx.t, self.dt
        t0 = t_end - dt
        snap = [dict(e.__dict__) for e in self.elements]
        guess = self._x_guess.copy()
        x, key = guess, None
        for m in (2, 4, 8, 16, 32):
            ctx.dt, ctx.method = dt / m, "be"
            ok = True
            for j in range(1, m + 1):
                ctx.t = t0 + j * ctx.dt
                x, key, ok = self._solve(ctx)
                if not ok:
                    break
                self._x_guess = x
                for e in self.elements:
                    e.update(x, ctx)
            if ok:
                ctx.dt, ctx.t = dt, t_end
                self.stats["substepped"] += 1
                return x, key, True, True
            for e, d in zip(self.elements, snap):
                e.__dict__.clear()
                e.__dict__.update(d)
            self._x_guess = guess.copy()
        ctx.dt, ctx.t = dt, t_end
        return x, key, False, False

    def _initial_solution(self, ctx):
        """Operating point at t=0 honouring capacitor-voltage and
        inductor-current initial conditions: a backward-Euler solve with a
        vanishingly small step turns capacitors into voltage sources (ic)
        and inductors into current sources (ic).  Element states are not
        advanced."""
        dt, ctx.dt, ctx.method = ctx.dt, ctx.dt * 1e-6, "be"
        try:
            x, _, _ = self._solve(ctx)
        except RuntimeError:
            x = np.zeros(self.size)
        finally:
            ctx.dt, ctx.method = dt, self.method
        return x

    def _dc_solve(self, ctx):
        """DC solution (capacitors open, inductors shorted) with source
        stepping as a fallback for hard-to-converge nonlinear circuits."""
        ctx.method = "dc"
        guess = self._x_guess.copy()
        # DC solutions are cheap: use 1000x tighter Newton tolerances
        saved = self.reltol, self._atol
        self.reltol, self._atol = self.reltol * 1e-3, self._atol * 1e-3
        try:
            return self._dc_solve_inner(ctx, guess)
        finally:
            self.reltol, self._atol = saved

    def _dc_solve_inner(self, ctx, guess):
        x, key, ok = self._solve(ctx)
        if not ok and self.nonlinear:
            self._x_guess = np.zeros(self.size)
            for e in self.nonlinear:
                e.init(ctx)
            for lam in np.linspace(0.05, 1.0, 20):
                ctx.source_scale = float(lam)
                x, key, ok = self._solve(ctx)
                self._x_guess = x
            ctx.source_scale = 1.0
            x, key, ok = self._solve(ctx)
        if not ok:
            self._x_guess = guess
            raise RuntimeError("DC operating point did not converge")
        return x, key

    def _apply_due(self, t_from):
        """Apply scheduled value changes whose time is <= t_from (the start of
        the step about to be solved).  Matrices depend on component values, so
        the cache is dropped when anything changed."""
        n = 0
        while self._pending and self._pending[0][0] <= t_from + 1e-9 * self.dt:
            _, obj, attr, value, _ = self._pending.pop(0)
            from .changes import set_attr
            set_attr(obj, attr, value)
            n += 1
        if n:
            self._cache.clear()
            self.stats["value_changes"] += n

    def _prepare(self):
        """Fresh context with initialised elements and control blocks."""
        self._new_stats()
        ctx = _Ctx(self)
        ctx.x = np.zeros(self.size)
        self._x_guess = np.zeros(self.size)
        for e in self.elements:
            e.init(ctx)
        for e in self.switching:
            e.on = False
        for b in self.blocks:
            b.init(ctx)
        for b in self.blocks:
            for o in b.outputs:
                ctx.signals.setdefault(o, 0.0)
        return ctx

    def _operating_point(self, ctx):
        """Solve the DC operating point in ``ctx`` and commit element states."""
        ctx._probes = self._collect_probes(ctx.x)
        self._eval_control(ctx)
        for e in self.gated:
            e.set_gate(ctx)
        x, _ = self._dc_solve(ctx)
        ctx.x = x
        self._x_guess = x
        for e in self.elements:
            e.update(x, ctx)
        ctx._probes = self._collect_probes(x)
        return x

    # ------------------------------------------------------------------
    # SPICE-style analyses
    def operating_point(self):
        """DC operating point (.op). Returns an :class:`OPResult`."""
        ctx = self._prepare()
        x = self._operating_point(ctx)
        return OPResult(self, x, ctx._probes)

    def dc_sweep(self, source, values):
        """DC sweep (.dc) of an independent source's value. Returns
        :class:`Results` with the swept value as the x axis."""
        from .waveforms import DC
        src = next((e for e in self.elements if e.name == source), None)
        if src is None or not hasattr(src, "wave"):
            raise ValueError(f"{source!r} is not an independent source")
        ctx = self._prepare()
        rows, names = [], None
        for v in values:
            src.wave = DC(v)
            if getattr(src, "dc", None) is not None:
                src.dc = float(v)          # a separate DC value would otherwise win
            x = self._operating_point(ctx)
            if names is None:
                names = list(ctx._probes)
            rows.append(list(x[:len(self.nodes)]) + [ctx._probes[k] for k in names])
        arr = np.array(rows)
        data = OrderedDict()
        for j, n in enumerate(self.nodes):
            data[f"V({n})"] = arr[:, j]
        for j, k in enumerate(names):
            data[k] = arr[:, len(self.nodes) + j]
        meta = {"title": self.circuit.title, "xlabel": f"{source} value",
                "element_nodes": {e.name: e.nodes for e in self.elements}, "stats": dict(self.stats)}
        return Results(np.asarray(values, float), data, meta)

    def ac_analysis(self, freqs, source=None):
        """Small-signal AC analysis (.ac) about the DC operating point.

        Excitation comes from sources with a non-zero ``ac`` magnitude; if
        ``source`` is given that source gets ac=1 (all others 0).
        Returns an :class:`ACResult`.
        """
        if source is not None:
            for e in self.elements:
                if hasattr(e, "ac"):
                    e.ac = 1.0 if e.name == source else 0.0
        if not any(getattr(e, "ac", 0.0) for e in self.elements):
            raise ValueError("no AC excitation: set ac=1 on a source or pass source=")
        ctx = self._prepare()
        self._operating_point(ctx)
        ctx.method = "dc"
        freqs = np.asarray(freqs, float)
        X = np.zeros((len(freqs), self.size), complex)
        for n, f in enumerate(freqs):
            w = 2 * np.pi * f
            Y = np.zeros((self.size, self.size), complex)
            rhs = np.zeros(self.size, complex)
            for e in self.elements:
                e.stamp_ac(Y, rhs, w, ctx)
            for i in range(len(self.nodes)):
                Y[i, i] += self.gmin
            X[n] = np.linalg.solve(Y, rhs)
        return ACResult(self, freqs, X)

    def _eval_control(self, ctx):
        sig = ctx.signals
        for b in self.delayed:
            sig[b.outputs[0]] = b.output(ctx)
        for b in self.direct:
            y = b.step([sig[i] for i in b.inputs], ctx)
            if len(b.outputs) == 1:
                sig[b.outputs[0]] = float(y)
            else:
                for o, v in zip(b.outputs, y):
                    sig[o] = float(v)
        for b in self.delayed:
            b.update([sig[i] for i in b.inputs], ctx)

    def _collect_probes(self, x):
        pr = {}
        for e in self.elements:
            pr.update(e.probes(x))
        return pr

    # ------------------------------------------------------------------
    def run(self, progress=False):
        t0 = time.perf_counter()
        ctx = self._prepare()
        self._pending = list(self._events)
        self._apply_due(0.0)                  # changes scheduled at t <= 0 act on the initial state
        if self.initial == "op":
            x = self._operating_point(ctx)
            ctx.method = self.method
        else:
            x = self._initial_solution(ctx)
        ctx.x = x
        self._x_guess = x
        ctx._probes = self._collect_probes(x)
        self._eval_control(ctx)

        n_steps = int(round(self.t_end / self.dt))
        rec_t, rec = [], []
        self._rec_names = None

        def record(t):
            if self._rec_names is None:
                self._rec_names = self._record_names(ctx)
                self._vidx = [self.node_index[n_] for n_ in self._rec_names[0]]
            nm = self._rec_names
            row = [ctx.x[i] for i in self._vidx]
            row += [ctx._probes[k] for k in nm[1]]
            row += [ctx.signals[k] for k in nm[2]]
            rec_t.append(t)
            rec.append(row)

        if self.t_record_start <= 0:
            record(0.0)
        prev_key = None
        next_report = 0.1
        for n in range(1, n_steps + 1):
            ctx.t = n * self.dt
            self._apply_due(ctx.t - self.dt)
            for e in self.gated:
                e.set_gate(ctx)
            # first step uses backward Euler: initial capacitor currents /
            # inductor voltages are unknown, which trapezoidal would mis-use
            ctx.method = "be" if n == 1 and self.initial == "ic" else self.method
            x, key, ok = self._solve(ctx)
            if ctx.method == "trap" and (not ok or (
                    self.damp_switching and prev_key is not None and key[2:] != prev_key[2:])):
                # topology changed (or trapezoidal found no consistent switch
                # state): redo the step with backward Euler, keeping the new
                # switch states as the starting guess
                ctx.method = "be"
                x, key, ok = self._solve(ctx)
                self.stats["damped_steps"] += 1
            updated = False
            if not ok and self.nonlinear:
                x, key, ok, updated = self._substeps(ctx)
            if not ok:
                self.stats["nonconverged"] += 1
            ctx.x = x
            self._x_guess = x
            if not updated:
                for e in self.elements:
                    e.update(x, ctx)
            prev_key = key
            ctx._probes = self._collect_probes(x)
            self._eval_control(ctx)

            if ctx.t >= self.t_record_start - 1e-15 and n % self.record_every == 0:
                record(ctx.t)
            if progress and n / n_steps >= next_report:
                print(f"  {100 * n / n_steps:5.1f}%  t={ctx.t:.4g}s")
                next_report += 0.1
        self.stats["steps"] = n_steps
        self.stats["cpu_time"] = time.perf_counter() - t0
        if self.stats["nonconverged"]:
            warnings.warn(f"switch-state iteration did not converge in {self.stats['nonconverged']} steps")

        data = OrderedDict()
        arr = np.array(rec) if rec else np.zeros((0, 0))
        names = self._rec_names
        if names is not None:
            cols = [f"V({n_})" for n_ in names[0]] + list(names[1]) + list(names[2])
            for j, c in enumerate(cols):
                data[c] = arr[:, j]
        meta = {
            "title": self.circuit.title,
            "dt": self.dt,
            "element_nodes": {e.name: e.nodes for e in self.elements},
            "stats": dict(self.stats),
            "changes": [c.to_dict() for c in self.circuit.changes],
        }
        return Results(np.array(rec_t), data, meta)

    def _record_names(self, ctx):
        f = self.record_filter
        nodes = list(self.nodes)
        probes = list(ctx._probes.keys())
        sigs = list(ctx.signals.keys())
        if f is not None:
            keep = set(f)
            nodes = [n for n in nodes if f"V({n})" in keep or n in keep]
            probes = [p for p in probes if p in keep]
            sigs = [s for s in sigs if s in keep]
        return nodes, probes, sigs


class OPResult:
    """DC operating point: node voltages, branch currents and device data."""

    def __init__(self, sim, x, probes):
        self.voltages = OrderedDict((n, float(x[i])) for n, i in sim.node_index.items())
        self.probes = OrderedDict((k, float(v)) for k, v in probes.items())
        self.title = sim.circuit.title

    def __getitem__(self, key):
        if key in self.probes:
            return self.probes[key]
        if key.startswith("V(") and key.endswith(")"):
            inner = key[2:-1]
            if "," in inner:
                a, b = (t.strip() for t in inner.split(","))
                return self._v(a) - self._v(b)
            return self._v(inner)
        return self._v(key)

    def _v(self, n):
        return 0.0 if n in el.GROUND_NAMES else self.voltages[n]

    def table(self):
        lines = [f"Operating point: {self.title}", "", f"{'node':<16}{'voltage':>14}"]
        lines += [f"{'V(' + n + ')':<16}{v:14.6g}" for n, v in self.voltages.items()]
        lines += ["", f"{'quantity':<16}{'value':>14}"]
        lines += [f"{k:<16}{v:14.6g}" for k, v in self.probes.items()]
        return "\n".join(lines)

    def __repr__(self):
        return self.table()


class ACResult:
    """Complex small-signal response vs frequency."""

    def __init__(self, sim, freqs, X):
        self.freq = freqs
        self.title = sim.circuit.title
        self._node = sim.node_index
        self._x = X
        self._branch = {e.name: e.br[0] for e in sim.elements if e.br}

    def __getitem__(self, key):
        """'V(out)', 'V(a,b)', 'out' or 'I(V1)' -> complex array."""
        if key.startswith("I(") and key.endswith(")"):
            return self._x[:, self._branch[key[2:-1]]]
        inner = key[2:-1] if key.startswith("V(") and key.endswith(")") else key
        if "," in inner:
            a, b = (t.strip() for t in inner.split(","))
            return self._nv(a) - self._nv(b)
        return self._nv(inner)

    def _nv(self, n):
        if n in el.GROUND_NAMES:
            return np.zeros(len(self.freq), complex)
        return self._x[:, self._node[n]]

    def transfer(self, out, ref=None):
        h = self[out]
        return h / self[ref] if ref else h

    def mag_db(self, out, ref=None):
        return 20 * np.log10(np.abs(self.transfer(out, ref)) + 1e-300)

    def phase_deg(self, out, ref=None):
        return np.degrees(np.unwrap(np.angle(self.transfer(out, ref))))

    def bode(self, out, ref=None):
        """Dict compatible with analysis.plot_bode()."""
        h = self.transfer(out, ref)
        return {"freq": self.freq, "gain": np.abs(h), "mag_db": self.mag_db(out, ref),
                "phase_deg": self.phase_deg(out, ref)}

    def bandwidth(self, out, ref=None, drop_db=3.0103):
        """First frequency where the gain falls drop_db below its value at
        the lowest analysed frequency (linear interpolation in log f)."""
        m = self.mag_db(out, ref)
        target = m[0] - drop_db
        idx = np.where(m < target)[0]
        if not len(idx):
            return float("nan")
        k = idx[0]
        if k == 0:
            return float(self.freq[0])
        f0, f1, m0, m1 = np.log10(self.freq[k - 1]), np.log10(self.freq[k]), m[k - 1], m[k]
        return float(10 ** (f0 + (target - m0) * (f1 - f0) / (m1 - m0)))

    def corners(self, out, ref=None, drop_db=3.0103):
        """(f_low, f_high): frequencies where the gain is drop_db below its
        peak, on either side of the peak (nan if not reached in range)."""
        m = self.mag_db(out, ref)
        k = int(np.argmax(m))
        target = m[k] - drop_db
        lf = np.log10(self.freq)

        def cross(i0, i1):
            return float(10 ** (lf[i0] + (target - m[i0]) * (lf[i1] - lf[i0]) / (m[i1] - m[i0])))

        lo = hi = float("nan")
        below = np.where(m[:k] < target)[0]
        if len(below):
            i = below[-1]
            lo = cross(i, i + 1)
        above = np.where(m[k:] < target)[0]
        if len(above):
            i = k + above[0]
            hi = cross(i - 1, i)
        return lo, hi

    def peak(self, out, ref=None):
        """(frequency, gain dB) of the maximum response."""
        m = self.mag_db(out, ref)
        k = int(np.argmax(m))
        return float(self.freq[k]), float(m[k])

    def plot(self, *outs, ref=None, title=None, save=None, show=False):
        fig, (a1, a2) = _subplots(show, 2, 1, sharex=True, figsize=(8, 5.5))
        for o in outs:
            lbl = o if ref is None else f"{o}/{ref}"
            a1.semilogx(self.freq, self.mag_db(o, ref), label=lbl)
            a2.semilogx(self.freq, self.phase_deg(o, ref), label=lbl)
        a1.set_ylabel("magnitude (dB)")
        a2.set_ylabel("phase (deg)")
        a2.set_xlabel("frequency (Hz)")
        a1.legend(fontsize=8)
        for a in (a1, a2):
            a.grid(True, which="both", alpha=0.3)
        fig.suptitle(title or f"AC analysis: {self.title}")
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=120)
        _finish(fig, show)
        return fig
