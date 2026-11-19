"""Open OpenPE Studio, the desktop GUI.  In PyCharm: right-click > Run 'gui'.

Works without installing the package (needs numpy + matplotlib and a Python
with Tkinter, which the python.org installers include).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openpe.gui import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
