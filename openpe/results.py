"""Simulation results and waveform post-processing (a SIMVIEW-like toolbox):
windowing, mean / RMS / ripple, FFT, THD, power factor, average power,
efficiency, plotting and CSV export."""
import csv
import math

import numpy as np


class Results:
    def __init__(self, t, data, meta=None):
        self.t = np.asarray(t)
        self.data = data
        self.meta = meta or {}

    # ------------------------------------------------------------------
    def __getitem__(self, name):
        if name in self.data:
            return self.data[name]
        if name == "t" or name == "time":
            return self.t
        if f"V({name})" in self.data:
            return self.data[f"V({name})"]
        if name.startswith("V(") and name.endswith(")") and "," in name:
            a, b = (s.strip() for s in name[2:-1].split(","))
            return self._node(a) - self._node(b)
        if name.startswith("VE(") and name.endswith(")"):
            return self.voltage_across(name[3:-1])
        raise KeyError(f"{name!r} not recorded. Available: {', '.join(self.names)}")

    def __contains__(self, name):
        try:
            self[name]
            return True
        except KeyError:
            return False

    @property
    def names(self):
        return list(self.data.keys())

    def _node(self, n):
        if n in ("0", "gnd", "GND", "ground"):
            return np.zeros_like(self.t)
        return self.data[f"V({n})"]

    def voltage_across(self, element):
        """Voltage across a two-terminal element (first node minus second)."""
        nodes = self.meta["element_nodes"][element]
        return self._node(nodes[0]) - self._node(nodes[1])

    # ------------------------------------------------------------------
    def window(self, t0=None, t1=None):
        """Return the results restricted to t0 <= t <= t1."""
        m = np.ones_like(self.t, dtype=bool)
        if t0 is not None:
            m &= self.t >= t0 - 1e-15
        if t1 is not None:
            m &= self.t <= t1 + 1e-15
        return Results(self.t[m], {k: v[m] for k, v in self.data.items()}, self.meta)

    def last(self, duration):
        """Window of the final ``duration`` seconds (e.g. N whole periods)."""
        return self.window(self.t[-1] - duration + 0.5 * self._dt(), None)

    def _dt(self):
        return float(self.t[1] - self.t[0]) if len(self.t) > 1 else self.meta.get("dt", 1.0)

    # ------------------------------------------------------------------
    def mean(self, name):
        return float(np.mean(self[name]))

    def rms(self, name):
        return float(np.sqrt(np.mean(np.square(self[name]))))

    def min(self, name):
        return float(np.min(self[name]))

    def max(self, name):
        return float(np.max(self[name]))

    def ripple(self, name):
        """Peak-to-peak value."""
        y = self[name]
        return float(np.max(y) - np.min(y))

    def final(self, name):
        return float(self[name][-1])

    def avg_power(self, v, i):
        """Average of v(t)*i(t) for two recorded quantities."""
        return float(np.mean(self[v] * self[i]))

    def power(self, element):
        """Average power absorbed by an element (negative = delivered)."""
        return float(np.mean(self.voltage_across(element) * self[f"I({element})"]))

    def efficiency(self, out_elements, in_elements):
        """Efficiency = sum(power absorbed by out_elements) /
        sum(power delivered by in_elements)."""
        pout = sum(self.power(e) for e in _list(out_elements))
        pin = -sum(self.power(e) for e in _list(in_elements))
        return pout / pin if pin else float("nan")

    def power_factor(self, v, i):
        """True power factor = P / (Vrms * Irms)."""
        return self.avg_power(v, i) / (self.rms(v) * self.rms(i))

    # ------------------------------------------------------------------
    def fft(self, name, f0=None, cycles=None):
        """Single-sided amplitude spectrum.

        If f0 and cycles are given only the last ``cycles`` periods of f0 are
        analysed (for an exact harmonic bin spacing of f0/cycles).
        Returns (freqs, amplitudes, phases_deg).
        """
        y, dt = self[name], self._dt()
        if f0 is not None:
            cycles = cycles or 1
            n = int(round(cycles / (f0 * dt)))
            if n > len(y):
                raise ValueError("not enough data for the requested number of cycles")
            y = y[-n:]
        n = len(y)
        spec = np.fft.rfft(y) / n
        amp = np.abs(spec)
        amp[1:] *= 2
        if n % 2 == 0:
            amp[-1] /= 2
        freqs = np.fft.rfftfreq(n, dt)
        return freqs, amp, np.degrees(np.angle(spec))

    def harmonic(self, name, f0, h=1, cycles=1):
        """Amplitude and phase (deg, sine reference) of harmonic h of f0."""
        y, dt = self[name], self._dt()
        n = int(round(cycles / (f0 * dt)))
        y = y[-n:]
        tt = self.t[-n:]
        w = 2 * math.pi * f0 * h
        a = 2 * np.mean(y * np.sin(w * tt))
        b = 2 * np.mean(y * np.cos(w * tt))
        return float(math.hypot(a, b)), float(math.degrees(math.atan2(b, a)))

    def thd(self, name, f0, cycles=1, n_harmonics=50):
        """Total harmonic distortion (ratio, not %) relative to the
        fundamental f0, using harmonics 2..n_harmonics."""
        freqs, amp, _ = self.fft(name, f0, cycles)
        df = freqs[1] - freqs[0]
        h1 = amp[int(round(f0 / df))]
        hs = [amp[int(round(k * f0 / df))] for k in range(2, n_harmonics + 1)
              if int(round(k * f0 / df)) < len(amp)]
        return float(math.sqrt(sum(a * a for a in hs)) / h1)

    # ------------------------------------------------------------------
    def summary(self, names=None, file=None):
        names = names or self.names
        lines = [f"{'signal':<22}{'mean':>12}{'rms':>12}{'min':>12}{'max':>12}{'p-p':>12}"]
        for n in names:
            lines.append(f"{n:<22}{self.mean(n):12.5g}{self.rms(n):12.5g}{self.min(n):12.5g}"
                         f"{self.max(n):12.5g}{self.ripple(n):12.5g}")
        text = "\n".join(lines)
        print(text, file=file)
        return text

    def to_csv(self, path, names=None):
        names = names or self.names
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time"] + names)
            cols = [self[n] for n in names]
            for k in range(len(self.t)):
                w.writerow([f"{self.t[k]:.9g}"] + [f"{c[k]:.9g}" for c in cols])

    def plot(self, *groups, title=None, save=None, show=False, xlim=None, figsize=None):
        """Plot waveforms, one subplot per group.

        Each group is a signal name or a list of names to overlay:
            res.plot("V(out)", ["I(L1)", "I(D1)"], save="buck.png")
        """
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        groups = groups or [[n] for n in self.names]
        groups = [[g] if isinstance(g, str) else list(g) for g in groups]
        fig, axes = plt.subplots(len(groups), 1, sharex=True,
                                 figsize=figsize or (9, 2.2 * len(groups) + 0.8), squeeze=False)
        xlabel = self.meta.get("xlabel")
        if xlabel:
            t, unit = self.t, None
        else:
            t, unit = _time_scale(self.t)
        for ax, grp in zip(axes[:, 0], groups):
            for n in grp:
                ax.plot(t, self[n], label=n, linewidth=1)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
        axes[-1, 0].set_xlabel(xlabel or f"time ({unit})")
        if xlim and not xlabel:
            s = {"s": 1, "ms": 1e3, "µs": 1e6}[unit]
            axes[-1, 0].set_xlim(xlim[0] * s, xlim[1] * s)
        fig.suptitle(title or self.meta.get("title", ""))
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=120)
        if show:
            plt.show()
        plt.close(fig)
        return fig

    def plot_spectrum(self, name, f0, cycles=1, fmax=None, save=None, show=False, title=None):
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f, a, _ = self.fft(name, f0, cycles)
        if fmax:
            m = f <= fmax
            f, a = f[m], a[m]
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.bar(f, a, width=(f[1] - f[0]) * 0.8)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("amplitude")
        ax.set_title(title or f"Spectrum of {name}  (THD = {100 * self.thd(name, f0, cycles):.2f} %)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=120)
        if show:
            plt.show()
        plt.close(fig)
        return fig


def _list(x):
    return [x] if isinstance(x, str) else list(x)


def _time_scale(t):
    span = float(t[-1]) if len(t) else 1.0
    if span < 2e-3:
        return t * 1e6, "µs"
    if span < 2:
        return t * 1e3, "ms"
    return t, "s"
