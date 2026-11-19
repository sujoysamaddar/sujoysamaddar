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
from .results import Results


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

    def node_voltage(self, node):
        if node in el.GROUND_NAMES:
            return 0.0
        return float(self.x[self._sim.node_index[node]])

    def probe(self, key):
        return self._probes[key]


class Simulator:
    def __init__(self, circuit, dt, t_end, method="trap", record_every=1,
                 t_record_start=0.0, max_iter=50, damp_switching=True,
                 cache_size=4096, record=None, gmin=1e-12):
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
        self.switching = [e for e in prims if e.is_switching]
        self.gated = [e for e in self.switching if hasattr(e, "gate")]
        self.self_commutated = [e for e in self.switching if type(e).violation is not el._Switching.violation]

        self.blocks = self._order_blocks(self.circuit.blocks)
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
    def _matrix_inv(self, key, ctx):
        inv = self._cache.get(key)
        if inv is not None:
            self._cache.move_to_end(key)
            return inv
        A = np.zeros((self.size, self.size))
        for e in self.elements:
            e.stamp_A(A, ctx)
        for i in range(len(self.nodes)):
            A[i, i] += self.gmin
        try:
            inv = np.linalg.inv(A)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                f"singular circuit matrix at t={ctx.t:g}s (floating node, loop of voltage sources "
                f"or inductor-only cut-set?)") from exc
        self._cache[key] = inv
        self.stats["factorizations"] += 1
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return inv

    def _solve(self, ctx):
        """Solve one step with diode/thyristor iteration. Returns x and key."""
        seen = set()
        best = None
        for it in range(self.max_iter):
            key = (ctx.method,) + tuple(e.on for e in self.switching)
            inv = self._matrix_inv(key, ctx)
            rhs = np.zeros(self.size)
            for e in self.elements:
                e.stamp_b(rhs, ctx)
            x = inv @ rhs
            viol = [(e.violation(x, ctx), k) for k, e in enumerate(self.self_commutated)]
            viol = [vk for vk in viol if vk[0] > 0.0]
            if not viol:
                return x, key, True
            worst = max(viol)[0]
            if best is None or worst < best[0]:
                best = (worst, x, key, [e.on for e in self.switching])
            if key in seen and it >= 2 * len(self.self_commutated) + 4:
                # No state is consistent for the whole step: a device current
                # crosses zero inside it (e.g. discontinuous conduction).
                # Accept the least-violating state (error <= one time step).
                _, x, key, states = best
                for e, st in zip(self.switching, states):
                    e.on = st
                self.stats["ambiguous_steps"] += 1
                return x, key, True
            if key in seen or it >= 8:
                # cycling: flip only the most strongly violated device
                self.self_commutated[max(viol)[1]].check(x, ctx)
            else:
                for _, k in viol:
                    self.self_commutated[k].check(x, ctx)
            seen.add(key)
        return x, key, False

    def _initial_solution(self, ctx):
        """Operating point at t=0 honouring capacitor-voltage and
        inductor-current initial conditions: a backward-Euler solve with a
        vanishingly small step turns capacitors into voltage sources (ic)
        and inductors into current sources (ic).  Element states are not
        advanced."""
        dt, ctx.dt, ctx.method = ctx.dt, ctx.dt * 1e-6, "be"
        try:
            A = np.zeros((self.size, self.size))
            x = np.zeros(self.size)
            for _ in range(self.max_iter):
                A[:] = 0.0
                for e in self.elements:
                    e.stamp_A(A, ctx)
                for i in range(len(self.nodes)):
                    A[i, i] += self.gmin
                rhs = np.zeros(self.size)
                for e in self.elements:
                    e.stamp_b(rhs, ctx)
                x = np.linalg.solve(A, rhs)
                if not any([e.check(x, ctx) for e in self.self_commutated]):
                    break
        except np.linalg.LinAlgError:
            x = np.zeros(self.size)
        finally:
            ctx.dt, ctx.method = dt, self.method
        return x

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
        self._cache = OrderedDict()
        self.stats = {"steps": 0, "factorizations": 0, "nonconverged": 0, "damped_steps": 0,
                      "ambiguous_steps": 0}
        ctx = _Ctx(self)
        ctx.x = np.zeros(self.size)
        for e in self.elements:
            e.init(ctx)
        for e in self.switching:
            e.on = False
        for b in self.blocks:
            b.init(ctx)
        for b in self.blocks:
            for o in b.outputs:
                ctx.signals.setdefault(o, 0.0)
        x = self._initial_solution(ctx)
        ctx.x = x
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
            for e in self.gated:
                e.set_gate(ctx)
            # first step uses backward Euler: initial capacitor currents /
            # inductor voltages are unknown, which trapezoidal would mis-use
            ctx.method = "be" if n == 1 else self.method
            x, key, ok = self._solve(ctx)
            if ctx.method == "trap" and (not ok or (
                    self.damp_switching and prev_key is not None and key[1:] != prev_key[1:])):
                # topology changed (or trapezoidal found no consistent switch
                # state): redo the step with backward Euler, keeping the new
                # switch states as the starting guess
                ctx.method = "be"
                x, key, ok = self._solve(ctx)
                self.stats["damped_steps"] += 1
            if not ok:
                self.stats["nonconverged"] += 1
            ctx.x = x
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
