"""Run the test suite.  In PyCharm: right-click this file > Run 'run_tests'."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    os.chdir(ROOT)
    sys.exit(pytest.main(["-q", "tests"] + sys.argv[1:]))
