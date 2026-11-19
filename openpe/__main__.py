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
    args = ap.parse_args(argv)

    if args.cmd == "blocks":
        from .netlist import BLOCK_TYPES
        print("\n".join(sorted(BLOCK_TYPES)))
        return 0

    nl = load(args.netlist)
    if not nl.tran:
        print("netlist has no .tran directive", file=sys.stderr)
        return 2
    res = nl.run()
    st = res.meta["stats"]
    print(f"{nl.circuit.title}: {st['steps']} steps, {st['factorizations']} matrix factorisations, "
          f"{st['cpu_time']:.2f} s CPU")
    window = args.window or nl.window
    mres = res.last(window) if window else res
    for m in nl.measures:
        print(f"  {' '.join(m):<30} = {_measure(mres, m):.6g}")
    plot_names = [n for g in nl.plots for n in g]
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
