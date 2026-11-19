import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)


def out(name):
    return os.path.join(OUT, name)
