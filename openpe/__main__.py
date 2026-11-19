"""Command line: python -m openpe run circuit.net [--plot out.png] [--csv out.csv]"""
import argparse
import sys

from .netlist import load


def _measure(res, spec):
    kind, sig = spec[0].lower(), spec[1]
    if kind == "mean":
        return res.mean(sig)
    if kind == "rms":
        return res.rms(sig)
    if kind == "max":
        return res.max(sig)
    if kind == "min":
        return res.min(sig)
    if kind == "pp":
        return res.ripple(sig)
    if kind == "thd":
        from .units import parse_value
        f0 = parse_value(spec[2])
        return res.thd(sig, f0, cycles=int(spec[3]) if len(spec) > 3 else 1)
    raise ValueError(f"unknown measure {kind}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="openpe", description="OpenPE power-electronics simulator")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="simulate a netlist")
    r.add_argument("netlist")
    r.add_argument("--plot", help="save waveform plot (PNG)")
    r.add_argument("--show", action="store_true", help="open an interactive plot window")
    r.add_argument("--csv", help="export waveforms to CSV")
    r.add_argument("--summary", action="store_true", help="print mean/rms/min/max of plotted signals")
    r.add_argument("--window", type=float, help="measure over the last N seconds only")
    sub.add_parser("blocks", help="list control block types")
    ed = sub.add_parser("editor", help="start the drag-and-drop schematic editor in a browser")
    ed.add_argument("--port", type=int, default=8765)
    ed.add_argument("--host", default="127.0.0.1")
    ed.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    args = ap.parse_args(argv)

    if args.cmd == "editor":
        from .server import serve
        serve(args.host, args.port, open_browser=not args.no_browser)
        return 0

    if args.cmd == "blocks":
        from .netlist import BLOCK_TYPES
        print("\n".join(sorted(BLOCK_TYPES)))
        return 0

    nl = load(args.netlist)
    if not nl.tran and not nl.analyses:
        print("netlist has no analysis directive (.tran, .op, .dc, .ac)", file=sys.stderr)
        return 2
    plot_names = [n for g in nl.plots for n in g]
    stem = args.plot[:-4] if args.plot and args.plot.lower().endswith(".png") else args.plot
    for a in nl.analyses:
        if a[0] == "op":
            print(nl.op().table())
        elif a[0] == "dc":
            r = nl.dc()
            print(f"DC sweep of {a[1]}: {len(r.t)} points")
            if args.plot or args.show:
                names = [n for n in plot_names if n in r] or [f"V({n})" for n in r.names[:1]]
                r.plot(*[[n] for n in names], save=f"{stem}_dc.png" if stem else None, show=args.show)
        elif a[0] == "ac":
            r = nl.ac()
            outs = [n for n in plot_names if n.startswith("V(")] or [f"V({list(r._node)[-1]})"]
            for o in outs:
                f_pk, g_pk = r.peak(o)
                lo, hi = r.corners(o)
                print(f"AC {o}: peak gain {g_pk:.2f} dB at {f_pk:.4g} Hz, "
                      f"-3 dB corners: low {lo:.4g} Hz, high {hi:.4g} Hz")
            if args.plot or args.show:
                r.plot(*outs, save=f"{stem}_ac.png" if stem else None, show=args.show)
    if not nl.tran:
        return 0
    res = nl.run()
    st = res.meta["stats"]
    print(f"{nl.circuit.title}: {st['steps']} steps, {st['factorizations']} matrix factorisations, "
          f"{st['cpu_time']:.2f} s CPU")
    window = args.window or nl.window
    mres = res.last(window) if window else res
    for m in nl.measures:
        print(f"  {' '.join(m):<30} = {_measure(mres, m):.6g}")
    if args.summary:
        mres.summary(plot_names or None)
    if args.csv:
        res.to_csv(args.csv, plot_names or None)
        print(f"  waveforms written to {args.csv}")
    if args.plot or args.show:
        res.plot(*(nl.plots or []), save=args.plot, show=args.show)
        if args.plot:
            print(f"  plot written to {args.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
