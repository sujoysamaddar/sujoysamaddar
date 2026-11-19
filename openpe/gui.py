"""OpenPE Studio - desktop GUI (Tkinter + embedded matplotlib).

    python -m openpe gui [file.net]
    python gui.py                      (from the project folder; PyCharm: Run 'gui')

Features: netlist library and editor with syntax colouring, one-click
simulation in a background thread, interactive waveform / Bode / DC-sweep
plots with zoom and pan, an operating-point table, a runner for the
example scripts that shows the plots they produce, and a button that opens
the drag-and-drop schematic editor.
"""
import contextlib
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
    from tkinter import filedialog, messagebox, simpledialog, ttk
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
        self._closed = False
        self._jobs = set()
        self.sim_busy = False        # a simulation / sweep is running
        self.script_busy = False     # an example script is running
        self.pending_run = False     # re-run requested while a simulation was running
        self._autorun_job = self._param_job = None
        self.params, self.nominal = [], {}
        self.sweep_data = None
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
        root.geometry("1440x860")
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
        self._closed = False
        self._poll_job = self._after(100, self._poll)


    @contextlib.contextmanager
    def _undo_group(self):
        """Make everything done inside the block a single Ctrl+Z step in the
        netlist editor (a replacement is a delete plus an insert, which Tk
        would otherwise undo one half at a time)."""
        self.text.edit_separator()
        self.text.configure(autoseparators=False)
        try:
            yield
        finally:
            self.text.edit_separator()
            self.text.configure(autoseparators=True)

    # ------------------------------------------------------------------ timers
    def _after(self, ms, fn):
        """root.after() that is tracked, so closing the window cancels it."""
        if getattr(self, "_closed", False):
            return None
        jobs = self.__dict__.setdefault("_jobs", set())

        def run():
            jobs.discard(job)
            fn()

        job = self.root.after(ms, run)
        jobs.add(job)
        return job

    def _after_idle(self, fn):
        return self._after(1, fn)

    def _cancel(self, job):
        if job:
            self.__dict__.setdefault("_jobs", set()).discard(job)
            try:
                self.root.after_cancel(job)
            except tk.TclError:
                pass

    @property
    def busy(self):
        return self.sim_busy or self.script_busy

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
        r.add_command(label="Sweep selected parameter", command=self.run_sweep)
        r.add_command(label="Run selected example script", command=self.run_selected_script)
        r.add_command(label="Stop script", command=self.stop_script)
        m.add_cascade(label="Run", menu=r)
        t = tk.Menu(m, tearoff=False)
        t.add_command(label="Open drag-and-drop schematic editor", command=self.open_schematic_editor)
        t.add_separator()
        t.add_command(label="Snap R / C / L values to a standard series...", command=self.snap_dialog)
        t.add_command(label="Schedule a value change during the simulation...", command=self.schedule_change)
        t.add_command(label="Solve the selected parameter for a target value...", command=self.solve_dialog)
        t.add_command(label="Step-response metrics of the selected signal...", command=self.step_metrics)
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
        self.tree.column("#0", width=210, minwidth=150)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<Double-1>", self._tree_open)
        self.tree.bind("<Return>", self._tree_open)
        ttk.Label(lib, text="Double-click: open a netlist\nor run an example script", foreground="gray").pack(anchor="w", pady=4)
        outer.add(lib, weight=1)
        right = ttk.PanedWindow(outer, orient="vertical")
        outer.add(right, weight=5)
        self._build_param_panel(outer)
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
        self._build_loss_tab()
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
        ttk.Button(side, text="Step metrics...", command=self.step_metrics).pack(anchor="w", pady=(0, 4))
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

    def _build_loss_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Losses")
        cols = ("type", "conduction", "switching", "total")
        self.loss_tree = ttk.Treeview(tab, columns=cols, show="tree headings", height=8)
        self.loss_tree.heading("#0", text="Device")
        for c in cols:
            self.loss_tree.heading(c, text=c if c == "type" else f"{c} (W)")
            self.loss_tree.column(c, width=90 if c == "type" else 110, anchor="e")
        self.loss_tree.column("#0", width=160)
        self.loss_tree.pack(fill="both", expand=True)
        self.loss_summary = tk.StringVar(value="Add a  .losses  line to the netlist (e.g.  .losses in=Vin out=Rload) "
                                               "and give switches eon= / eoff= energies to get a loss report.")
        ttk.Label(tab, textvariable=self.loss_summary, padding=6, wraplength=700, justify="left").pack(fill="x")

    def _build_image_tab(self):
        tab = self.tab_images = ttk.Frame(self.nb)
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
        self.nominal = {}                    # a new design: remember its original values
        self.refresh_params()

    def _on_modified(self, _e=None):
        if self.text.edit_modified():
            self.dirty = True
            self.text.edit_modified(False)
            self._title()
            self._after_idle(self._highlight)
            if self._param_job:
                self._cancel(self._param_job)
            self._param_job = self._after(400, self.refresh_params)

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
    def _set_busy(self, busy, msg="", which="sim"):
        if which == "sim":
            self.sim_busy = busy
        else:
            self.script_busy = busy
        self.run_btn.state(["disabled"] if self.sim_busy else ["!disabled"])
        if self.busy:
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
        self._autorun_job = None
        if getattr(self, "_closed", False):
            return
        if self.sim_busy:
            self.pending_run = True          # run again with the latest values when done
            return
        self.pending_run = False
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
                    self._maybe_rerun()
                elif kind == "solve":
                    self._show_solution(msg[1])
                elif kind == "solveerr":
                    self._set_busy(False, "Solve failed - see Log")
                    self.log_write(msg[1] + "\n", "err")
                    self.nb.select(self.nb.tabs()[-1])
                    self._maybe_rerun()
                elif kind == "sweepprog":
                    self.status.set(f"Sweep: run {msg[1]} of {msg[2]} ({msg[3]})...")
                elif kind == "sweep":
                    self._show_sweep(msg[1])
                elif kind == "sweeperr":
                    self._set_busy(False, "Sweep failed - see Log")
                    self.log_write(msg[1] + "\n", "err")
                    self.nb.select(self.nb.tabs()[-1])
                elif kind == "out":
                    self.log_write(msg[1])
                elif kind == "done":
                    self._script_done(*msg[1:])
        except queue.Empty:
            pass
        if not self._closed:
            self._poll_job = self._after(80, self._poll)

    def _maybe_rerun(self):
        if self.pending_run:
            self.pending_run = False
            self._after(10, self.run_netlist)

    def _show_results(self, res, dt):
        self.result = res
        self.sweep_data = None
        self._set_busy(False, f"{res.get('title', '')}: finished in {dt:.2f} s")
        self.log_write(f"[{time.strftime('%H:%M:%S')}] {res.get('title', '')}: simulation finished in {dt:.2f} s\n", "ok")
        for w in res.get("warnings") or []:
            self.log_write(f"  warning: {w}\n", "err")
        if res.get("warnings"):
            self.status.set(self.status.get() + f"  -  {len(res['warnings'])} warning(s), see Log")
        self._update_signal_choices(res)
        self._maybe_rerun()
        if "tran" in res:
            st = res["tran"]["stats"]
            self.log_write("  transient: " + ", ".join(f"{k}={v}" for k, v in st.items()) + "\n")
        for k, v in (res.get("measures") or {}).items():
            if v is not None:
                self.log_write(f"  {k} = {_fmt(v)}\n")
        self.measure_var.set("\n".join(f"{k} = {_fmt(v) if v is not None else 'error'}"
                                        for k, v in (res.get("measures") or {}).items()))
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
        self._show_losses(res.get("losses"))
        tabs = self.nb.tabs()
        target = 0 if "tran" in res else 1 if "ac" in res else 2 if "dc" in res else 3
        self.nb.select(tabs[target])

    def _show_losses(self, rep):
        self.loss_tree.delete(*self.loss_tree.get_children())
        if not rep:
            self.loss_summary.set("Add a  .losses  line to the netlist (e.g.  .losses in=Vin out=Rload) "
                                  "and give switches eon= / eoff= energies to get a loss report.")
            return
        for r in rep["rows"]:
            self.loss_tree.insert("", "end", text=r["name"],
                                  values=(r["type"], f"{r['conduction']:.4g}", f"{r['switching']:.4g}", f"{r['total']:.4g}"))
        self.loss_tree.insert("", "end", text="all devices",
                              values=("", f"{rep['conduction_loss']:.4g}", f"{rep['switching_loss']:.4g}", f"{rep['device_loss']:.4g}"))
        txt = f"window {rep['window'][0] * 1e3:.4g} .. {rep['window'][1] * 1e3:.4g} ms.  "
        if rep.get("efficiency") is not None:
            txt += (f"Input {rep['p_in']:.4g} W (circuit) + {rep['switching_loss']:.4g} W switching loss, output {rep['p_out']:.4g} W, "
                    f"other losses {rep['other_loss']:.3g} W.  Efficiency {100 * rep['efficiency']:.2f} %")
        else:
            txt += "Add in=<source> out=<element> to the .losses line for efficiency."
        self.loss_summary.set(txt)

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
        for chg in res.get("changes") or []:                     # mark scheduled value changes
            if t and t[0] <= chg["t"] <= t[-1]:
                for ax in axes:
                    ax.axvline(chg["t"], color="#6b7280", ls="--", lw=0.9)
                axes[0].annotate(f"{chg['target']}\u2192{chg['value']:g}", (chg["t"], 1.0),
                                 xycoords=("data", "axes fraction"), fontsize=7, color="#6b7280",
                                 rotation=90, va="top", ha="right")
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


    # ------------------------------------------------------------------ design-time parameters
    def _build_param_panel(self, outer):
        pane = ttk.Frame(outer, padding=4)
        outer.add(pane, weight=2)
        ttk.Label(pane, text="Design parameters", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        self.autorun = tk.BooleanVar(value=True)
        ttk.Checkbutton(pane, text="Auto-run after each change", variable=self.autorun).pack(anchor="w")
        self.filter_var = tk.StringVar()
        frow = ttk.Frame(pane)
        frow.pack(fill="x", pady=(4, 2))
        ttk.Label(frow, text="Filter").pack(side="left")
        ttk.Entry(frow, textvariable=self.filter_var).pack(side="left", fill="x", expand=True, padx=(4, 0))
        self.filter_var.trace_add("write", lambda *a: self.refresh_params())
        tf = ttk.Frame(pane)
        tf.pack(fill="both", expand=True)
        self.ptree = ttk.Treeview(tf, columns=("value",), show="tree headings", selectmode="browse", height=12)
        self.ptree.heading("#0", text="Parameter")
        self.ptree.heading("value", text="Value")
        self.ptree.column("#0", width=150, minwidth=110)
        self.ptree.column("value", width=80, minwidth=60, anchor="e")
        ys = ttk.Scrollbar(tf, orient="vertical", command=self.ptree.yview)
        self.ptree.configure(yscrollcommand=ys.set)
        ys.pack(side="right", fill="y")                     # pack first so it is never squeezed out
        self.ptree.pack(side="left", fill="both", expand=True)
        self.ptree.bind("<<TreeviewSelect>>", lambda e: self._param_selected())
        self.ptree.bind("<Double-1>", lambda e: (self.p_entry.focus_set(), self.p_entry.select_range(0, "end")))
        self.ptree.tag_configure("changed", foreground="#d97706")
        # tune box
        tune = ttk.LabelFrame(pane, text="Change value", padding=6)
        tune.pack(fill="x", pady=4)
        self.p_name = tk.StringVar(value="select a parameter")
        ttk.Label(tune, textvariable=self.p_name, font=("TkDefaultFont", 9, "bold")).grid(row=0, column=0, columnspan=3, sticky="w")
        self.p_value = tk.StringVar()
        self.p_entry = ttk.Entry(tune, textvariable=self.p_value, width=12)
        self.p_entry.grid(row=1, column=0, sticky="ew", pady=2)
        self.p_entry.bind("<Return>", lambda e: self.apply_param())
        self.p_entry.bind("<Up>", lambda e: self.step_param(+1))
        self.p_entry.bind("<Down>", lambda e: self.step_param(-1))
        ttk.Button(tune, text="Apply", width=6, command=self.apply_param).grid(row=1, column=1, padx=2)
        ttk.Button(tune, text="Reset", width=6, command=self.reset_param).grid(row=1, column=2)
        self.p_scale = tk.Scale(tune, from_=-1.0, to=1.0, resolution=0.01, orient="horizontal", showvalue=False,
                                command=self._scale_moved, length=180)
        self.p_scale.grid(row=2, column=0, columnspan=3, sticky="ew")
        self.p_scale.bind("<ButtonRelease-1>", lambda e: self.apply_param())
        self.p_range = tk.StringVar(value="slider: nominal ÷10 … ×10")
        ttk.Label(tune, textvariable=self.p_range, foreground="gray").grid(row=3, column=0, columnspan=3, sticky="w")
        ttk.Label(tune, text="↑/↓ in the box: ±10 %", foreground="gray").grid(row=4, column=0, columnspan=3, sticky="w")
        ttk.Label(tune, text="Standard values (E-series)", foreground="gray").grid(row=5, column=0, columnspan=3, sticky="w", pady=(4, 0))
        std = ttk.Frame(tune)
        std.grid(row=6, column=0, columnspan=3, sticky="ew")
        self.series_var = tk.StringVar(value="E24")
        ttk.Combobox(std, textvariable=self.series_var, values=["E6", "E12", "E24", "E48", "E96"],
                     width=5, state="readonly").pack(side="left")
        ttk.Button(std, text="\u25c0", width=3, command=lambda: self.std_step(-1)).pack(side="left", padx=(4, 1))
        ttk.Button(std, text="\u25b6", width=3, command=lambda: self.std_step(+1)).pack(side="left", padx=1)
        ttk.Button(std, text="Snap", width=5, command=self.std_snap).pack(side="left", padx=(4, 0))
        tune.columnconfigure(0, weight=1)
        # sweep box
        sw = ttk.LabelFrame(pane, text="Parameter sweep", padding=6)
        sw.pack(fill="x", pady=4)
        self.sw_from, self.sw_to, self.sw_n = tk.StringVar(), tk.StringVar(), tk.StringVar(value="5")
        self.sw_log = tk.BooleanVar(value=False)
        for r, (lbl, var) in enumerate((("From", self.sw_from), ("To", self.sw_to), ("Steps", self.sw_n))):
            ttk.Label(sw, text=lbl).grid(row=r, column=0, sticky="w")
            ttk.Entry(sw, textvariable=var, width=10).grid(row=r, column=1, sticky="ew", pady=1)
        ttk.Checkbutton(sw, text="log", variable=self.sw_log).grid(row=2, column=2)
        ttk.Label(sw, text="Signal").grid(row=3, column=0, sticky="w")
        self.sw_signal = ttk.Combobox(sw, width=14, state="readonly")
        self.sw_signal.grid(row=3, column=1, columnspan=2, sticky="ew", pady=1)
        ttk.Button(sw, text="Run sweep", command=self.run_sweep).grid(row=4, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        sw.columnconfigure(1, weight=1)

    def refresh_params(self):
        from .designparams import extract_params
        self._param_job = None
        text = self.text.get("1.0", "end-1c")
        self.params = extract_params(text)
        for p in self.params:
            self.nominal.setdefault(p.label, p.value)
        sel = self._selected_label()
        flt = self.filter_var.get().strip().lower()
        self.ptree.delete(*self.ptree.get_children())
        groups = {}
        for p in self.params:
            if flt and flt not in p.label.lower():
                continue
            if p.group not in groups:
                groups[p.group] = self.ptree.insert("", "end", text=p.group, open=True)
            tags = ("changed",) if self.nominal.get(p.label) != p.value else ()
            self.ptree.insert(groups[p.group], "end", iid="p:" + p.label, text=p.name, values=(p.value,), tags=tags)
        if sel and self.ptree.exists("p:" + sel):
            self.ptree.selection_set("p:" + sel)
            self.ptree.see("p:" + sel)
        self._update_signal_choices()

    def _selected_label(self):
        sel = self.ptree.selection() if hasattr(self, "ptree") else ()
        return sel[0][2:] if sel and sel[0].startswith("p:") else None

    def _current_param(self):
        lbl = self._selected_label()
        return next((p for p in self.params if p.label == lbl), None) if lbl else None

    def select_param(self, label):
        """Select a design parameter by label, e.g. 'R1.value' or 'B4.kp'."""
        self.ptree.selection_set("p:" + label)
        self._param_selected()

    def _param_selected(self):
        p = self._current_param()
        if not p:
            return
        self.p_name.set(f"{p.group} → {p.name}   (original {self.nominal.get(p.label, p.value)})")
        self.p_value.set(p.value)
        self._scale_lock = True
        self.p_scale.set(0.0)
        self._scale_lock = False
        nom = self._nominal_number(p)
        if nom:
            from .designparams import format_value
            self.p_range.set(f"slider: {format_value(nom / 10)} … {format_value(nom * 10)}")
            if not self.sw_from.get() or self._sweep_for != p.label:
                self.sw_from.set(format_value(nom / 2))
                self.sw_to.set(format_value(nom * 2))
        else:
            self.p_range.set("slider: (value is zero - type a value)")
        self._sweep_for = p.label

    _sweep_for = None
    _scale_lock = False

    def _nominal_number(self, p):
        from .units import parse_value
        try:
            v = parse_value(self.nominal.get(p.label, p.value))
        except ValueError:
            return None
        return v if v else None

    def _scale_moved(self, x):
        if self._scale_lock:
            return
        p = self._current_param()
        nom = self._nominal_number(p) if p else None
        if nom:
            from .designparams import format_value
            self.p_value.set(format_value(nom * 10 ** float(x), 3))

    def std_step(self, direction):
        """Move the value to the next / previous standard (E-series) value."""
        from . import eseries
        from .designparams import format_value
        from .units import parse_value
        try:
            v = parse_value(self.p_value.get())
            self.p_value.set(format_value(eseries.step(v, direction, self.series_var.get()), 4))
        except ValueError:
            self.status.set("Standard values need a positive number")
            return
        self.apply_param()

    def std_snap(self):
        """Replace the value with the nearest standard (E-series) value."""
        from . import eseries
        from .designparams import format_value
        from .units import parse_value
        try:
            v = parse_value(self.p_value.get())
            n = eseries.nearest(v, self.series_var.get())
            self.p_value.set(format_value(n, 4))
            self.status.set(f"nearest {self.series_var.get()} value: {format_value(n, 4)} "
                            f"({eseries.error_percent(v, self.series_var.get()):+.2f} %)")
        except ValueError:
            self.status.set("Standard values need a positive number")
            return
        self.apply_param()

    def snap_dialog(self, series=None, kinds=None):
        """Snap every R / C / L value of the netlist to a standard series."""
        from .designparams import snap_to_series
        if series is None:
            ans = simpledialog.askstring(
                "Standard values", "Series (E6, E12, E24, E48, E96) and parts to snap (R, C, L), e.g.  E24 RC",
                initialvalue="E24 RC", parent=self.root)
            if not ans:
                return
            parts = ans.split()
            series = parts[0].upper()
            kinds = parts[1].upper() if len(parts) > 1 else "RC"
        try:
            new, changed = snap_to_series(self.text.get("1.0", "end-1c"), series, kinds or "RC")
        except ValueError as exc:
            messagebox.showerror("Standard values", str(exc))
            return
        if not changed:
            self.status.set(f"All {kinds or 'RC'} values are already {series} values")
            return
        with self._undo_group():                               # one Ctrl+Z restores every snapped value
            self.text.delete("1.0", "end")
            self.text.insert("1.0", new)
        self._highlight()
        self.refresh_params()
        self.log_write(f"=== snapped to {series} ===\n", "ok")
        for name, old, nv, err in changed:
            self.log_write(f"  {name}: {old} -> {nv}  ({err:+.2f} %)\n")
        self.status.set(f"{len(changed)} value(s) snapped to {series} (Ctrl+Z in the editor undoes it)")
        if self.autorun.get():
            self.run_netlist()

    def schedule_change(self, spec=None):
        """Insert a '.change' line: a value that changes at a time during the simulation."""
        from .netlist import parse
        if spec is None:
            spec = simpledialog.askstring(
                "Value change during the simulation",
                "time target value   (examples:  5m Rload 3     0.3 M1 tload=4     7m vref 9)",
                parent=self.root)
        if not spec or not spec.strip():
            return
        line = ".change " + spec.strip()
        lines = self.text.get("1.0", "end-1c").split("\n")
        at = next((i for i, ln in enumerate(lines) if ln.strip().lower().startswith((".tran", ".end"))), len(lines))
        trial = "\n".join(lines[:at] + [line] + lines[at:])
        try:
            from .simulator import Simulator
            Simulator(parse(trial).circuit, 1e-6, 1e-6)    # parses, then resolves targets and attributes
        except Exception as exc:
            messagebox.showerror("Value change", str(exc))
            return
        self.text.insert(f"{at + 1}.0", line + "\n")
        self._highlight()
        self.refresh_params()
        self.status.set(f"added: {line}")
        if self.autorun.get():
            self.run_netlist()

    def step_param(self, direction):
        from .designparams import format_value
        from .units import parse_value
        try:
            v = parse_value(self.p_value.get())
        except ValueError:
            return "break"
        self.p_value.set(format_value(v * (1.1 if direction > 0 else 1 / 1.1) if v else 0.1 * direction, 3))
        self.apply_param()
        return "break"

    def reset_param(self):
        p = self._current_param()
        if p:
            self.p_value.set(self.nominal.get(p.label, p.value))
            self.apply_param()

    def apply_param(self, value=None):
        """Write the edited value into the netlist (and re-run if Auto-run)."""
        from .units import parse_value
        p = self._current_param()
        if not p:
            return False
        new = (value if value is not None else self.p_value.get()).strip()
        try:
            parse_value(new)
        except ValueError:
            messagebox.showerror("Invalid value", f"'{new}' is not a number (examples: 4.7k, 100u, 2meg, 1e-3).")
            return False
        if new == p.value:
            return True
        line = p.line + 1
        with self._undo_group():
            self.text.delete(f"{line}.{p.start}", f"{line}.{p.end}")
            self.text.insert(f"{line}.{p.start}", new)
        self._highlight()
        self.refresh_params()
        self.status.set(f"{p.label} = {new}")
        if self.autorun.get():
            if self._autorun_job:
                self._cancel(self._autorun_job)
            self._autorun_job = self._after(250, self.run_netlist)
        return True

    def _update_signal_choices(self, res=None):
        names = []
        res = res or self.result
        if res:
            for k in ("tran", "dc"):
                if k in res:
                    names += [n for n in res[k]["signals"] if n not in names]
            if "ac" in res:
                names += [n for n in res["ac"]["traces"] if n not in names]
        if not names:
            try:
                from .netlist import parse
                names = [n for g in parse(self.text.get("1.0", "end-1c")).plots for n in g]
            except Exception:
                names = []
        self.sw_signal["values"] = names
        if names and self.sw_signal.get() not in names:
            self.sw_signal.set(names[0])

    def step_metrics(self, signal=None, t_step=None):
        """Rise time, overshoot and settling time of a waveform after a step."""
        import numpy as np
        from .results import Results
        from .units import parse_value
        res = self.result
        if not res or "tran" not in res:
            messagebox.showinfo("Step metrics", "Run a transient simulation first.")
            return None
        names = [self.sig_list.get(i) for i in self.sig_list.curselection()]
        name = signal or (names[0] if names else None)
        if not name:
            messagebox.showinfo("Step metrics", "Select a signal in the Waveforms list first.")
            return None
        if t_step is None:
            ans = simpledialog.askstring("Step metrics", f"Time of the step in {name} (seconds, e.g. 0 or 4m):",
                                         initialvalue="0", parent=self.root)
            if ans is None:
                return None
            try:
                t_step = parse_value(ans)
            except ValueError:
                messagebox.showerror("Step metrics", f"'{ans}' is not a time")
                return None
        data = np.array([float("nan") if v is None else v for v in res["tran"]["signals"][name]])
        try:
            info = Results(np.array(res["tran"]["t"]), {name: data}).step_info(name, t_step=t_step)
        except ValueError as exc:
            messagebox.showerror("Step metrics", str(exc))
            return None
        txt = (f"{name}: {_fmt(info['initial'])} -> {_fmt(info['final'])}   rise {_fmt(info['rise_time'], 's')}, "
               f"overshoot {info['overshoot_pct']:.2f} %, settling {_fmt(info['settling_time'], 's')}")
        self.log_write(f"[step metrics] {txt}\n", "ok")
        self.measure_var.set((self.measure_var.get() + "\n" if self.measure_var.get() else "") +
                             f"rise {_fmt(info['rise_time'], 's')}\novershoot {info['overshoot_pct']:.2f} %\n"
                             f"settling {_fmt(info['settling_time'], 's')}")
        self.status.set(txt)
        return info

    def solve_dialog(self, spec=None):
        """Find the value of the selected parameter that makes a measurement reach a target."""
        from .tuning import solve_for
        from .units import parse_value
        p = self._current_param()
        if not p:
            messagebox.showinfo("Solve", "Select a parameter in the Design parameters list first.")
            return
        if self.sim_busy:
            return
        if spec is None:
            spec = simpledialog.askstring(
                "Solve for a target value",
                f"Find the value of {p.label} for which\n   measurement = value\n"
                f"(examples:  overshoot V(b) = 20     max V(out) = 12     mean I(L1) = 3)\n"
                f"Searched between {self.sw_from.get()} and {self.sw_to.get()} (the Parameter sweep boxes).",
                parent=self.root)
        if not spec:
            return
        try:
            meas, val = spec.rsplit("=", 1)
            target = parse_value(val.strip())
            lo, hi = parse_value(self.sw_from.get()), parse_value(self.sw_to.get())
        except ValueError:
            messagebox.showerror("Solve", "Write it as  measurement = value,  e.g.  overshoot V(b) = 20")
            return
        base, log = self.text.get("1.0", "end-1c"), self.sw_log.get()
        self._set_busy(True, f"Solving {p.label} for {meas.strip()} = {target:g} ...")

        def work():
            try:
                self.q.put(("solve", solve_for(base, p.label, meas.strip(), target, lo, hi, log=log)))
            except Exception as exc:
                self.q.put(("solveerr", f"Solve failed: {type(exc).__name__}: {exc}"))

        threading.Thread(target=work, daemon=True).start()

    def _show_solution(self, sol):
        from .designparams import format_value
        self._set_busy(False, str(sol))
        self.log_write(f"\n=== solve {sol.label} for {sol.spec} = {sol.target:g} ===\n", "ok")
        for k, (v, m) in enumerate(sol.history, 1):
            self.log_write(f"  run {k:>2}: {sol.label} = {v:<14.8g} {sol.spec} = {m:.6g}\n")
        self.log_write(str(sol) + "\n", "ok" if sol.converged else "err")
        if sol.converged:
            self.select_param(sol.label)
            self.p_value.set(format_value(sol.value, 6))
            self.apply_param()                       # writes the value into the netlist (and re-runs with Auto-run)
        self._maybe_rerun()

    def run_sweep(self):
        from .designparams import format_value, set_param, sweep_values
        p = self._current_param()
        if not p:
            messagebox.showinfo("Parameter sweep", "Select a parameter in the Design parameters list first.")
            return
        if self.sim_busy:
            return
        try:
            values = sweep_values(self.sw_from.get(), self.sw_to.get(), self.sw_n.get(), self.sw_log.get())
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Parameter sweep", f"Invalid sweep range: {exc}")
            return
        signal = self.sw_signal.get()
        if not signal:
            messagebox.showinfo("Parameter sweep", "Add a .plot line (or run once) so there is a signal to compare.")
            return
        base = self.text.get("1.0", "end-1c")
        self._set_busy(True, f"Sweeping {p.label} ({len(values)} runs)...")

        def work():
            from .netlist import parse
            from .server import _decimate
            runs = []
            try:
                for k, v in enumerate(values, 1):
                    txt = set_param(base, p, format_value(v))
                    self.q.put(("sweepprog", k, len(values), f"{p.label}={format_value(v)}"))
                    nl = parse(txt)
                    run = {"value": v, "label": f"{p.label}={format_value(v)}"}
                    if nl.tran:
                        res = nl.run()
                        if signal not in res:
                            raise ValueError(f"signal {signal} not found")
                        win = res.last(nl.window) if nl.window else res.window(res.t[-1] * 0.5)
                        y = res[signal]
                        t, d = _decimate(res.t, {signal: y}, 3000)
                        run.update(kind="tran", x=t, y=d[signal], mean=win.mean(signal), min=win.min(signal),
                                   max=win.max(signal), final=float(y[-1]))
                    elif any(a[0] == "ac" for a in nl.analyses):
                        r = nl.ac()
                        lo, hi = r.corners(signal)
                        f_pk, g_pk = r.peak(signal)
                        run.update(kind="ac", x=r.freq, y=r.mag_db(signal), peak_db=g_pk, peak_hz=f_pk, f_high=hi)
                    elif any(a[0] == "dc" for a in nl.analyses):
                        r = nl.dc()
                        run.update(kind="dc", x=r.t, y=r[signal])
                    else:
                        op = nl.op()
                        run.update(kind="op", y=op[signal])
                    runs.append(run)
                self.q.put(("sweep", {"param": p.label, "signal": signal, "runs": runs}))
            except Exception as exc:
                self.q.put(("sweeperr", f"Sweep failed at {p.label}: {type(exc).__name__}: {exc}"))

        threading.Thread(target=work, daemon=True).start()

    def _show_sweep(self, data):
        self.sweep_data = data
        runs = data["runs"]
        self._set_busy(False, f"Sweep of {data['param']} finished ({len(runs)} runs)")
        sig = data["signal"]
        kind = runs[0]["kind"] if runs else "op"
        self.log_write(f"\n=== sweep of {data['param']}, signal {sig} ===\n", "ok")
        if kind == "tran":
            self.log_write(f"{'value':>12} {'mean':>12} {'min':>12} {'max':>12} {'final':>12}\n")
            for r in runs:
                self.log_write(f"{_fmt(r['value']):>12} {_fmt(r['mean']):>12} {_fmt(r['min']):>12} "
                               f"{_fmt(r['max']):>12} {_fmt(r['final']):>12}\n")
        elif kind == "ac":
            for r in runs:
                self.log_write(f"{_fmt(r['value']):>12}  peak {r['peak_db']:.2f} dB @ {_fmt(r['peak_hz'], 'Hz')}, "
                               f"-3 dB high {_fmt(r['f_high'], 'Hz')}\n")
        elif kind == "op":
            for r in runs:
                self.log_write(f"{_fmt(r['value']):>12}  {sig} = {_fmt(r['y'])}\n")
        # overlay plot on the Waveforms tab
        fig = self.fig_w
        fig.clear()
        ax = fig.add_subplot(111)
        import matplotlib.cm as cm
        cmap = cm.get_cmap("viridis") if hasattr(cm, "get_cmap") else matplotlib.colormaps["viridis"]
        for k, r in enumerate(runs):
            c = cmap(k / max(1, len(runs) - 1))
            if kind == "ac":
                ax.semilogx(r["x"], r["y"], color=c, label=r["label"])
            elif kind == "op":
                ax.plot([r["value"]], [r["y"]], "o", color=c)
            else:
                ax.plot(r["x"], r["y"], color=c, lw=1, label=r["label"])
        if kind == "op":
            ax.plot([r["value"] for r in runs], [r["y"] for r in runs], "-", color="gray")
            ax.set_xlabel(data["param"])
        elif kind == "tran":
            ax.xaxis.set_major_formatter(EngFormatter(unit="s"))
            ax.set_xlabel("time")
        elif kind == "ac":
            ax.xaxis.set_major_formatter(EngFormatter(unit="Hz"))
            ax.set_ylabel(f"|{sig}| (dB)")
        ax.set_title(f"Sweep of {data['param']}: {sig}")
        if kind != "op":
            ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        self.canvas_w.draw_idle()
        self.nb.select(self.nb.tabs()[0])

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
        self._set_busy(True, f"Running {name}...", which="script")
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
        self._set_busy(False, f"{name} {'finished' if code == 0 else f'failed (exit code {code})'}", which="script")
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
            self.nb.select(self.tab_images)
            self._after(50, self._show_image)

    def _show_image(self):
        if getattr(self, "_closed", False):
            return
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
        # flush pending redraws, then cancel every timer so nothing fires
        # into a destroyed window
        try:
            self.root.update_idletasks()
        except tk.TclError:
            pass
        self._closed = True
        for job in list(self.__dict__.get("_jobs", ())):
            self._cancel(job)
        for canvas in (self.canvas_w, self.canvas_ac, self.canvas_dc):
            idle = getattr(canvas, "_idle_draw_id", None)
            if idle:
                try:
                    self.root.after_cancel(idle)
                except tk.TclError:
                    pass
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
