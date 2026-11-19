"""OpenPE - an open-source power-electronics circuit simulator.

Inspired by the workflow of dedicated power-electronics simulators:
ideal switch models, fixed-step simulation with cached matrices, a separate
control circuit linked through sensors, motor-drive elements, waveform
post-processing (FFT/THD/RMS/power), parameter sweeps and AC sweeps --
plus SPICE-style analog analyses (DC operating point, DC sweep, AC
small-signal, Monte Carlo) with nonlinear semiconductor devices,
controlled sources and op-amps.
"""
import sys as _sys

if _sys.version_info < (3, 10):
    raise ImportError(f"OpenPE needs Python 3.10 or newer (this interpreter is {_sys.version.split()[0]}). "
                      "In PyCharm: Settings > Project > Python Interpreter.")
try:
    import numpy as _np  # noqa: F401
except ImportError as _exc:
    raise ImportError("OpenPE needs numpy. Install the requirements with:  pip install -r requirements.txt  "
                      "(PyCharm: open requirements.txt and click 'Install requirements').") from _exc

from . import control, devices, elements, waveforms
from .analysis import (MonteCarloResult, ac_sweep, format_table, monte_carlo, parameter_sweep, plot_bode,
                       tf_response)
from .circuit import Circuit
from .control import (PI, PID, PLL, PWM, ABCtoDQ, Abs, Comparator, Constant, CurrentSensor, DeadTime,
                      Divide, DQtoABC, Function, Gain, Integrator, Limiter, Logic, LowPass, MovingAverage,
                      Product, ProbeSensor, PythonBlock, Ramp, RateLimiter, RMSMeter, SampleHold, Sawtooth,
                      SineWave, SquareWave, StepSignal, Sum, TransferFunction, Triangle, UnitDelay,
                      VoltageSensor)
from .devices import BJT, CCCS, CCVS, MOSFET, VCCS, VCVS, OpAmp, PNDiode, Zener, thermal_voltage
from .elements import (Capacitor, CoupledInductors, CurrentSource, DCMotor, Diode, IdealTransformer,
                       Inductor, Resistor, Switch, ThreePhaseSource, Thyristor, Transformer, VoltageSource)
from .netlist import load as load_netlist
from .netlist import parse as parse_netlist
from .results import Results
from .simulator import ACResult, OPResult, Simulator
from .units import parse_value
from .waveforms import DC, PWL, Sine, Square, Step

__version__ = "0.2.0"
