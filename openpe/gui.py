"""OpenPE Studio - desktop GUI (Tkinter + embedded matplotlib).

    python -m openpe gui [file.net]
    python gui.py                      (from the project folder; PyCharm: Run 'gui')

Features: netlist library and editor with syntax colouring, one-click
simulation in a background thread, interactive waveform / Bode / DC-sweep
plots with zoom and pan, an operating-point table, a runner for the
example scripts that shows the plots they produce, and a button that opens
the drag-and-drop schematic editor.
"""
import glob
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import webbrowser

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError as exc:  # pragma: no cover - depends on the Python build
    raise ImportError(
        "The desktop GUI needs Tkinter, which is missing from this Python installation. "
        "On Windows/macOS reinstall Python from python.org with 'tcl/tk' enabled; "
        "on Linux install the 'python3-tk' package. The browser editor still works: "
        "python -m openpe editor") from exc

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import EngFormatter  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CIRCUITS = os.path.join(PROJECT, "circuits")
EXAMPLES = os.path.join(PROJECT, "examples")
COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#65a30d"]

NEW_NETLIST = """* New circuit: RC low-pass filter
.title RC filter
V1 in 0 DC 0 AC 1 SIN(0 1 1k)
R1 in out 1k
C1 out 0 100n
.tran 2u 5m
.ac dec 20 10 1meg
.op
.plot V(in) V(out)
.end
"""


def _fmt(v, unit=""):
    return EngFormatter(unit=unit, places=4)(v) if v is not None else "-"


class Studio:
    def __init__(self, root, path=None):
        self.root = root
        self.path = None
        self.dirty = False
        self.busy = False
        self.result = None
        self.proc = None
        self.server = None
        self.q = queue.Queue()
        self.images = []
        from tkinter import font as tkfont
        self.mono = tkfont.nametofont("TkFixedFont").copy()
        self.mono.configure(size=10)
        self.mono_b = self.mono.copy()
        self.mono_b.configure(weight="bold")
        root.title("OpenPE Studio")
        root.geometry("1320x840")
        root.minsize(900, 560)
        root.protocol("WM_DELETE_WINDOW", self.quit)
        self._build_menu()
        self._build_toolbar()
        self._build_body()
        self.status = tk.StringVar(value="Ready - open a netlist from the library or press F5 to run")
        ttk.Label(root, textvariable=self.status, anchor="w", padding=(8, 3)).pack(side="bottom", fill="x")
        root.bind("<F5>", lambda e: self.run_netlist())
        root.bind("<Control-Return>", lambda e: self.run_netlist())
        root.bind("<Control-s>", lambda e: self.save())
        root.bind("<Control-o>", lambda e: self.open_dialog())
        root.bind("<Control-n>", lambda e: self.new())
        self._fill_library()
        if path:
            self.load(path)
        else:
            default = os.path.join(CIRCUITS, "buck_current_mode.net")
            self.load(default) if os.path.exists(default) else self.set_text(NEW_NETLIST)
        root.after(100, self._poll)

    # ------------------------------------------------------------------ layout
    def _build_menu(self):
        m = tk.Menu(self.root)
        f = tk.Menu(m, tearoff=False)
        f.add_command(label="New netlist", accelerator="Ctrl+N", command=self.new)
        f.add_command(label="Open netlist...", accelerator="Ctrl+O", command=self.open_dialog)
        f.add_command(label="Save", accelerator="Ctrl+S", command=self.save)
        f.add_command(label="Save as...", command=self.save_as)
        f.add_separator()
        f.add_command(label="Export waveforms (CSV)...", command=self.export_csv)
        f.add_separator()
        f.add_command(label="Exit", command=self.quit)
        m.add_cascade(label="File", menu=f)
        r = tk.Menu(m, tearoff=False)
        r.add_command(label="Run netlist", accelerator="F5", command=self.run_netlist)
        r.add_command(label="Run selected example script", command=self.run_selected_script)
        r.add_command(label="Stop script", command=self.stop_script)
        m.add_cascade(label="Run", menu=r)
        t = tk.Menu(m, tearoff=False)
        t.add_command(label="Open drag-and-drop schematic editor", command=self.open_schematic_editor)
        m.add_cascade(label="Tools", menu=t)
        h = tk.Menu(m, tearoff=False)
        h.add_command(label="Netlist syntax", command=self.show_syntax)
        h.add_command(label="About", command=self.about)
        m.add_cascade(label="Help", menu=h)
        self.root.config(menu=m)

    def _build_toolbar(self):
        bar = ttk.Frame(self.root, padding=(6, 4))
        bar.pack(side="top", fill="x")
        for text, cmd in (("New", self.new), ("Open...", self.open_dialog), ("Save", self.save)):
            ttk.Button(bar, text=text, command=cmd).pack(side="left", padx=2)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=6)
        self.run_btn = ttk.Button(bar, text="▶  Run (F5)", command=self.run_netlist)
        self.run_btn.pack(side="left", padx=2)
        ttk.Button(bar, text="Run example script", command=self.run_selected_script).pack(side="left", padx=2)
        ttk.Button(bar, text="Stop", command=self.stop_script).pack(side="left", padx=2)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(bar, text="Schematic editor (drag & drop)", command=self.open_schematic_editor).pack(side="left", padx=2)
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=120)
        self.progress.pack(side="right", padx=6)

    def _build_body(self):
        outer = ttk.PanedWindow(self.root, orient="horizontal")
        outer.pack(fill="both", expand=True)
        # library
        lib = ttk.Frame(outer, padding=4)
        ttk.Label(lib, text="Library", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        self.tree = ttk.Treeview(lib, show="tree", selectmode="browse")
        self.tree.column("#0", width=250, minwidth=150)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<Double-1>", self._tree_open)
        self.tree.bind("<Return>", self._tree_open)
        ttk.Label(lib, text="Double-click: open a netlist\nor run an example script", foreground="gray").pack(anchor="w", pady=4)
        outer.add(lib, weight=1)
        right = ttk.PanedWindow(outer, orient="vertical")
        outer.add(right, weight=5)
        # editor
        ed = ttk.Frame(right)
        self.file_label = tk.StringVar(value="untitled.net")
        ttk.Label(ed, textvariable=self.file_label, padding=(4, 2)).pack(anchor="w")
        tf = ttk.Frame(ed)
        tf.pack(fill="both", expand=True)
        self.text = tk.Text(tf, wrap="none", undo=True, font=self.mono, height=14)
        ys = ttk.Scrollbar(tf, orient="vertical", command=self.text.yview)
        xs = ttk.Scrollbar(tf, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        tf.rowconfigure(0, weight=1)
        tf.columnconfigure(0, weight=1)
        self.text.tag_configure("comment", foreground="#6b7280")
        self.text.tag_configure("directive", foreground="#2563eb", font=self.mono_b)
        self.text.tag_configure("block", foreground="#7c3aed")
        self.text.tag_configure("elem", foreground="#0f766e", font=self.mono_b)
        self.text.bind("<<Modified>>", self._on_modified)
        right.add(ed, weight=2)
        # results
        self.nb = ttk.Notebook(right)
        right.add(self.nb, weight=3)
        self._build_wave_tab()
        self.fig_ac, self.canvas_ac = self._figure_tab("AC analysis")
        self.fig_dc, self.canvas_dc = self._figure_tab("DC sweep")
        self._build_op_tab()
        self._build_image_tab()
        self._build_log_tab()

    def _figure_tab(self, title, parent=None):
        frame = parent or ttk.Frame(self.nb)
        if parent is None:
            self.nb.add(frame, text=title)
        fig = Figure(figsize=(8, 4), dpi=100, layout="constrained")
        canvas = FigureCanvasTkAgg(fig, master=frame)
        tb = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
        tb.update()
        tb.pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(fill="both", expand=True)
        fig.text(0.5, 0.5, "Run a simulation to see results here", ha="center", va="center", color="gray")
        canvas.draw_idle()
        return fig, canvas

    def _build_wave_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Waveforms")
        side = ttk.Frame(tab, padding=4)
        side.pack(side="left", fill="y")
        ttk.Label(side, text="Signals").pack(anchor="w")
        self.sig_list = tk.Listbox(side, selectmode="extended", exportselection=False, width=22, height=12)
        self.sig_list.pack(fill="y", expand=True)
        self.sig_list.bind("<<ListboxSelect>>", lambda e: self.plot_waves())
        self.overlay = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text="Overlay in one plot", variable=self.overlay, command=self.plot_waves).pack(anchor="w", pady=4)
        self.measure_var = tk.StringVar(value="")
        ttk.Label(side, textvariable=self.measure_var, foreground="#15803d", wraplength=170, justify="left").pack(anchor="w")
        plot = ttk.Frame(tab)
        plot.pack(side="left", fill="both", expand=True)
        self.fig_w, self.canvas_w = self._figure_tab(None, parent=plot)

    def _build_op_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Operating point")
        self.op_tree = ttk.Treeview(tab, columns=("value",), show="tree headings")
        self.op_tree.heading("#0", text="Quantity")
        self.op_tree.heading("value", text="Value")
        self.op_tree.column("value", width=160, anchor="e")
        self.op_tree.pack(fill="both", expand=True)

    def _build_image_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Script plots")
        side = ttk.Frame(tab, padding=4)
        side.pack(side="left", fill="y")
        ttk.Label(side, text="Images").pack(anchor="w")
        self.img_list = tk.Listbox(side, width=30, exportselection=False)
        self.img_list.pack(fill="y", expand=True)
        self.img_list.bind("<<ListboxSelect>>", lambda e: self._show_image())
        self.img_canvas = tk.Canvas(tab, background="white", highlightthickness=0)
        self.img_canvas.pack(side="left", fill="both", expand=True)
        self.img_canvas.bind("<Configure>", lambda e: self._show_image())

    def _build_log_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Log")
        self.log = tk.Text(tab, wrap="word", font=self.mono, state="disabled")
        ys = ttk.Scrollbar(tab, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=ys.set)
        ys.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True)
        self.log.tag_configure("err", foreground="#dc2626")
        self.log.tag_configure("ok", foreground="#15803d")

    def _fill_library(self):
        self.tree.delete(*self.tree.get_children())
        nets = sorted(glob.glob(os.path.join(CIRCUITS, "*.net")))
        scripts = sorted(glob.glob(os.path.join(EXAMPLES, "[0-9]*.py")))
        n = self.tree.insert("", "end", text="Netlists", open=True)
        for p in nets:
            self.tree.insert(n, "end", iid="net:" + p, text=os.path.basename(p))
        s = self.tree.insert("", "end", text="Example scripts", open=True)
        for p in scripts:
            self.tree.insert(s, "end", iid="py:" + p, text=os.path.basename(p))
        if not nets and not scripts:
            self.tree.insert("", "end", text="(project folder not found)")

    # ------------------------------------------------------------------ editor
    def set_text(self, text):
        self.text.delete("1.0", "end")
        self.text.insert("1.0", text)
        self.text.edit_reset()
        self.text.edit_modified(False)
        self.dirty = False
        self._highlight()

    def _on_modified(self, _e=None):
        if self.text.edit_modified():
            self.dirty = True
            self.text.edit_modified(False)
            self._title()
            self.root.after_idle(self._highlight)

    def _highlight(self):
        for t in ("comment", "directive", "block", "elem"):
            self.text.tag_remove(t, "1.0", "end")
        lines = self.text.get("1.0", "end").split("\n")
        for i, ln in enumerate(lines, 1):
            s = ln.lstrip()
            if not s:
                continue
            start = f"{i}.{len(ln) - len(s)}"
            if s[0] in "*#":
                self.text.tag_add("comment", start, f"{i}.end")
                continue
            word_end = f"{i}.{len(ln) - len(s) + len(s.split()[0])}"
            if s[0] == ".":
                self.text.tag_add("directive", start, word_end)
            elif s[0] in "Bb":
                self.text.tag_add("block", start, f"{i}.end")
            else:
                self.text.tag_add("elem", start, word_end)
            if ";" in ln:
                self.text.tag_add("comment", f"{i}.{ln.index(';')}", f"{i}.end")

    def _title(self):
        name = os.path.basename(self.path) if self.path else "untitled.net"
        self.file_label.set(("* " if self.dirty else "") + (self.path or name))
        self.root.title(f"OpenPE Studio - {name}{' *' if self.dirty else ''}")

    def _confirm_discard(self):
        return not self.dirty or messagebox.askyesno("Unsaved changes", "Discard changes to the current netlist?")

    def new(self):
        if self._confirm_discard():
            self.path = None
            self.set_text(NEW_NETLIST)
            self._title()

    def load(self, path):
        with open(path, encoding="utf-8") as f:
            self.set_text(f.read())
        self.path = os.path.abspath(path)
        self._title()
        self.status.set(f"Opened {path}")

    def open_dialog(self):
        if not self._confirm_discard():
            return
        p = filedialog.askopenfilename(title="Open netlist", initialdir=CIRCUITS if os.path.isdir(CIRCUITS) else None,
                                       filetypes=[("Netlists", "*.net *.cir *.sp *.txt"), ("All files", "*.*")])
        if p:
            self.load(p)

    def save(self):
        if not self.path:
            return self.save_as()
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(self.text.get("1.0", "end-1c"))
        self.dirty = False
        self._title()
        self.status.set(f"Saved {self.path}")

    def save_as(self):
        p = filedialog.asksaveasfilename(title="Save netlist", defaultextension=".net",
                                         filetypes=[("Netlists", "*.net"), ("All files", "*.*")])
        if p:
            self.path = p
            self.save()

    def _tree_open(self, _e=None):
        iid = self.tree.focus()
        if iid.startswith("net:"):
            if self._confirm_discard():
                self.load(iid[4:])
        elif iid.startswith("py:"):
            self.run_script(iid[3:])

    # ------------------------------------------------------------------ simulation
    def _set_busy(self, busy, msg=""):
        self.busy = busy
        self.run_btn.state(["disabled"] if busy else ["!disabled"])
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress["value"] = 0
        if msg:
            self.status.set(msg)

    def log_write(self, text, tag=None):
        self.log.configure(state="normal")
        self.log.insert("end", text, tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def run_netlist(self):
        if self.busy:
            return
        from .server import simulate_netlist
        text = self.text.get("1.0", "end-1c")
        self._set_busy(True, "Simulating...")
        t0 = time.perf_counter()

        def work():
            try:
                res = simulate_netlist(text, max_points=20000)
                self.q.put(("sim", res, time.perf_counter() - t0))
            except Exception as exc:  # report any parse / simulation error
                self.q.put(("simerr", f"{type(exc).__name__}: {exc}", traceback.format_exc()))

        threading.Thread(target=work, daemon=True).start()

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "sim":
                    self._show_results(msg[1], msg[2])
                elif kind == "simerr":
                    self._set_busy(False, "Simulation failed - see Log")
                    self.log_write(msg[1] + "\n", "err")
                    self.nb.select(self.nb.tabs()[-1])
                elif kind == "out":
                    self.log_write(msg[1])
                elif kind == "done":
                    self._script_done(*msg[1:])
        except queue.Empty:
            pass
        self.root.after(80, self._poll)

    def _show_results(self, res, dt):
        self.result = res
        self._set_busy(False, f"{res.get('title', '')}: finished in {dt:.2f} s")
        self.log_write(f"[{time.strftime('%H:%M:%S')}] {res.get('title', '')}: simulation finished in {dt:.2f} s\n", "ok")
        if "tran" in res:
            st = res["tran"]["stats"]
            self.log_write("  transient: " + ", ".join(f"{k}={v}" for k, v in st.items()) + "\n")
        for k, v in (res.get("measures") or {}).items():
            self.log_write(f"  {k} = {_fmt(v)}\n")
        self.measure_var.set("\n".join(f"{k} = {_fmt(v)}" for k, v in (res.get("measures") or {}).items()))
        # waveforms
        self.sig_list.delete(0, "end")
        if "tran" in res:
            names = list(res["tran"]["signals"])
            for n in names:
                self.sig_list.insert("end", n)
            if names:
                self.sig_list.selection_set(0, min(len(names), 4) - 1)
        self.plot_waves()
        self._plot_ac(res.get("ac"))
        self._plot_dc(res.get("dc"))
        self.op_tree.delete(*self.op_tree.get_children())
        if "op" in res:
            v = self.op_tree.insert("", "end", text="Node voltages", open=True)
            for k, val in res["op"]["voltages"].items():
                self.op_tree.insert(v, "end", text=f"V({k})", values=(_fmt(val, "V"),))
            p = self.op_tree.insert("", "end", text="Currents and device values", open=True)
            for k, val in res["op"]["probes"].items():
                unit = "A" if k.startswith(("I(", "Ib(")) else "V" if k.startswith(("Vbe", "Vce", "Vgs", "Vds", "Vd(")) else ""
                self.op_tree.insert(p, "end", text=k, values=(_fmt(val, unit),))
        tabs = self.nb.tabs()
        target = 0 if "tran" in res else 1 if "ac" in res else 2 if "dc" in res else 3
        self.nb.select(tabs[target])

    def plot_waves(self):
        fig = self.fig_w
        fig.clear()
        res = self.result
        if not res or "tran" not in res:
            fig.text(0.5, 0.5, "No transient results (add .tran and .plot to the netlist)", ha="center", va="center", color="gray")
            self.canvas_w.draw_idle()
            return
        t = res["tran"]["t"]
        names = [self.sig_list.get(i) for i in self.sig_list.curselection()]
        if not names:
            names = list(res["tran"]["signals"])[:1]
        sigs = res["tran"]["signals"]
        unit = lambda n: "A" if n.startswith("I") else "V" if n.startswith("V(") else ""  # noqa: E731
        if self.overlay.get():
            ax = fig.add_subplot(111)
            for i, n in enumerate(names):
                ax.plot(t, [float("nan") if v is None else v for v in sigs[n]], color=COLORS[i % 8], lw=1, label=n)
            ax.legend(loc="upper right", fontsize=8)
            axes = [ax]
        else:
            axes = []
            for i, n in enumerate(names):
                ax = fig.add_subplot(len(names), 1, i + 1, sharex=axes[0] if axes else None)
                ax.plot(t, [float("nan") if v is None else v for v in sigs[n]], color=COLORS[i % 8], lw=1, label=n)
                ax.legend(loc="upper right", fontsize=8)
                ax.yaxis.set_major_formatter(EngFormatter(unit=unit(n)))
                if i < len(names) - 1:
                    ax.tick_params(labelbottom=False)
                axes.append(ax)
        for ax in axes:
            ax.grid(True, alpha=0.3)
        axes[-1].xaxis.set_major_formatter(EngFormatter(unit="s"))
        axes[-1].set_xlabel("time")
        self.canvas_w.draw_idle()

    def _plot_ac(self, ac):
        fig = self.fig_ac
        fig.clear()
        if not ac:
            fig.text(0.5, 0.5, "No AC results (add .ac to the netlist)", ha="center", va="center", color="gray")
        else:
            a1, a2 = fig.add_subplot(211), fig.add_subplot(212)
            for i, (n, tr) in enumerate(ac["traces"].items()):
                c = COLORS[i % 8]
                a1.semilogx(ac["freq"], tr["mag_db"], color=c, label=n)
                a2.semilogx(ac["freq"], tr["phase_deg"], color=c, label=n)
                for f in (tr.get("f_low"), tr.get("f_high")):
                    if f:
                        a1.axvline(f, color=c, ls=":", lw=1)
            a1.set_ylabel("magnitude (dB)")
            a2.set_ylabel("phase (deg)")
            a2.xaxis.set_major_formatter(EngFormatter(unit="Hz"))
            a1.legend(fontsize=8)
            for a in (a1, a2):
                a.grid(True, which="both", alpha=0.3)
        self.canvas_ac.draw_idle()

    def _plot_dc(self, dc):
        fig = self.fig_dc
        fig.clear()
        if not dc:
            fig.text(0.5, 0.5, "No DC sweep (add .dc to the netlist)", ha="center", va="center", color="gray")
        else:
            ax = fig.add_subplot(111)
            for i, (n, d) in enumerate(dc["signals"].items()):
                ax.plot(dc["x"], [float("nan") if v is None else v for v in d], color=COLORS[i % 8], label=n)
            ax.set_xlabel(f"{dc['source']} value")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        self.canvas_dc.draw_idle()

    def export_csv(self):
        res = self.result
        if not res or "tran" not in res:
            messagebox.showinfo("Export", "Run a transient simulation first.")
            return
        p = filedialog.asksaveasfilename(title="Export waveforms", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not p:
            return
        names = list(res["tran"]["signals"])
        with open(p, "w", encoding="utf-8") as f:
            f.write(",".join(["time"] + names) + "\n")
            for k, tv in enumerate(res["tran"]["t"]):
                f.write(",".join([repr(tv)] + [repr(res["tran"]["signals"][n][k]) for n in names]) + "\n")
        self.status.set(f"Exported {len(names)} signals to {p}")

    # ------------------------------------------------------------------ example scripts
    def run_selected_script(self):
        iid = self.tree.focus()
        if iid.startswith("py:"):
            self.run_script(iid[3:])
        else:
            messagebox.showinfo("Run example", "Select an example script in the library first.")

    def run_script(self, path):
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("Busy", "An example script is already running (use Stop).")
            return
        name = os.path.basename(path)
        self.log_write(f"\n=== running {name} ===\n", "ok")
        self.nb.select(self.nb.tabs()[-1])
        self._set_busy(True, f"Running {name}...")
        start = time.time()
        env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        self.proc = subprocess.Popen([sys.executable, path], cwd=PROJECT, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")

        def reader(p=self.proc):
            for line in p.stdout:
                self.q.put(("out", line))
            self.q.put(("done", name, p.wait(), start))

        threading.Thread(target=reader, daemon=True).start()

    def stop_script(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.log_write("stopped\n", "err")

    def _script_done(self, name, code, start):
        self._set_busy(False, f"{name} {'finished' if code == 0 else f'failed (exit code {code})'}")
        self.log_write(f"--- {name}: {'OK' if code == 0 else f'exit code {code}'} ({time.time() - start:.1f} s)\n",
                       "ok" if code == 0 else "err")
        out_dir = os.path.join(EXAMPLES, "output")
        new = sorted((p for p in glob.glob(os.path.join(out_dir, "*.png")) if os.path.getmtime(p) >= start - 1),
                     key=os.path.getmtime)
        if new:
            self.images = new
            self.img_list.delete(0, "end")
            for p in new:
                self.img_list.insert("end", os.path.basename(p))
            self.img_list.selection_set(0)
            self.nb.select(self.nb.tabs()[4])
            self.root.after(50, self._show_image)

    def _show_image(self):
        sel = self.img_list.curselection()
        if not sel or not self.images:
            return
        path = self.images[sel[0]]
        try:
            img = tk.PhotoImage(file=path)
        except tk.TclError:
            return
        w, h = max(self.img_canvas.winfo_width(), 50), max(self.img_canvas.winfo_height(), 50)
        f = max(1, -(-img.width() // w), -(-img.height() // h))   # integer shrink factor to fit
        if f > 1:
            img = img.subsample(f, f)
        self._img_ref = img
        self.img_canvas.delete("all")
        self.img_canvas.create_image(w // 2, h // 2, image=img, anchor="center")

    # ------------------------------------------------------------------ misc
    def open_schematic_editor(self):
        if self.server is None:
            from .server import make_server
            try:
                self.server = make_server("127.0.0.1", 8765)
            except OSError:
                self.server = make_server("127.0.0.1", 0)       # port busy: any free port
            threading.Thread(target=self.server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{self.server.server_address[1]}/"
        webbrowser.open(url)
        self.status.set(f"Schematic editor running at {url} (opened in your browser)")
        self.log_write(f"schematic editor: {url}\n", "ok")

    def show_syntax(self):
        from . import netlist
        win = tk.Toplevel(self.root)
        win.title("Netlist syntax")
        t = tk.Text(win, wrap="none", font=self.mono, width=100, height=40)
        t.insert("1.0", netlist.__doc__)
        t.configure(state="disabled")
        t.pack(fill="both", expand=True)

    def about(self):
        from . import __version__
        messagebox.showinfo("About OpenPE Studio",
                            f"OpenPE {__version__}\nPower-electronics and circuit simulator.\n\n"
                            "Netlist editor + simulator (this window) and a drag-and-drop\n"
                            "schematic editor (Tools menu).")

    def quit(self):
        if not self._confirm_discard():
            return
        self.stop_script()
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        self.root.destroy()


def main(path=None):
    root = tk.Tk()
    Studio(root, path)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.path.insert(0, PROJECT)
    from openpe.gui import main as _main
    sys.exit(_main(sys.argv[1] if len(sys.argv) > 1 else None))
