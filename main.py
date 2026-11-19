"""Start here.

    python main.py                 # opens OpenPE Studio, the desktop GUI window
    python main.py editor          # drag-and-drop schematic editor (browser)
    python main.py run circuits/rlc_step.net --plot rlc.png

In PyCharm: right-click this file > Run 'main'.  Works without installing
the package; only the packages in requirements.txt are needed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openpe.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["gui"]))
