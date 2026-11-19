"""Run every script in examples/ and report which ones succeed.

In PyCharm: right-click this file > Run 'run_examples'.  Plots are written
to examples/output/.  Pass names to run only some, e.g. "01 09".
"""
import glob
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))


def main(filters):
    scripts = sorted(glob.glob(os.path.join(ROOT, "examples", "[0-9]*.py")))
    if filters:
        scripts = [s for s in scripts if any(f in os.path.basename(s) for f in filters)]
    failed = []
    for path in scripts:
        name = os.path.basename(path)
        print(f"=== {name}", flush=True)
        t0 = time.time()
        env = dict(os.environ, PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
        r = subprocess.run([sys.executable, path], cwd=ROOT, env=env)
        print(f"--- {name}: {'OK' if r.returncode == 0 else 'FAILED'} ({time.time() - t0:.1f} s)\n", flush=True)
        if r.returncode:
            failed.append(name)
    print(f"{len(scripts) - len(failed)}/{len(scripts)} examples succeeded"
          + (f"; failed: {', '.join(failed)}" if failed else "") + f". Plots: {os.path.join(ROOT, 'examples', 'output')}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
