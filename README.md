# OpenPE — open-source power-electronics and circuit simulator

OpenPE is a Python circuit simulator for **power electronics, motor drives,
their control loops, and analog electronics**.  It combines the features
that make dedicated power-electronics simulators such as PSIM pleasant to
use with the analyses of general-purpose SPICE-class simulators.

From the power-electronics side:

* ideal switch models and a fast fixed time step
* a power circuit kept separate from the control circuit, linked by sensors
* ready-made PWM, PI and dead-time blocks
* motor-drive elements
* a waveform viewer with FFT/THD
* parameter sweeps and AC sweeps on the switching circuit itself

From the SPICE side:

* DC operating point, DC sweep, AC small-signal and Monte Carlo analyses
* nonlinear diodes, Zeners, BJTs and MOSFETs solved by Newton-Raphson
* controlled sources and op-amps

It's pure Python: only `numpy` is required, and `matplotlib` is optional
for plots.

```python
import openpe as pe

c = pe.Circuit("Buck 48 V -> 12 V")
c.V("Vin", "in", "0", 48)
c.S("Q1", "in", "sw", gate="g")        # ideal switch driven by signal "g"
c.D("D1", "0", "sw")
c.L("L1", "sw", "out", 100e-6)
c.C("C1", "out", "0", 220e-6)
c.R("RL", "out", "0", 6)
c.add(pe.Constant("d", 0.25), pe.PWM("g", "d", freq=50e3))

res = c.simulate(dt=0.21e-6, t_end=10e-3)
print(res.last(1e-3).mean("V(out)"))   # -> 11.99
res.plot("V(out)", ["I(L1)", "I(D1)"], save="buck.png")
```

The same package handles analog electronics:

```python
amp = pe.Circuit("CE amplifier")
amp.V("Vcc", "vcc", "0", 12)
amp.V("Vin", "in", "0", pe.Sine(10e-3, 1e3), ac=1, dc=0)
amp.C("Cin", "in", "b", 10e-6)
amp.R("RB1", "vcc", "b", 47e3); amp.R("RB2", "b", "0", 10e3)
amp.R("RC", "vcc", "c", 2.2e3); amp.R("RE", "e", "0", 1e3); amp.C("CE", "e", "0", 100e-6)
amp.Q("Q1", "c", "b", "e", "npn", is_=1e-14, bf=150, vaf=100, cjc=5e-12)

print(amp.op())                                   # .op  bias point table
ac = amp.ac(np.logspace(0, 8, 400))               # .ac  small-signal response
print(ac.peak("V(c)"), ac.corners("V(c)"))        # midband gain, -3 dB corners
res = amp.simulate(2e-6, 5e-3, initial="op")      # .tran from the bias point
print(res.thd("V(c)", 1e3, cycles=2))             # harmonic distortion
```

---

## Feature overview

| PSIM-style feature | OpenPE implementation |
|---|---|
| **Ideal switch models** | `Switch` (MOSFET/IGBT, optional body diode), `Diode` (optional Vf), `Thyristor` (latching SCR). Piecewise-linear on/off resistances, with automatic diode/thyristor state iteration. |
| **Fast fixed-step solver** | MNA with trapezoidal (default) or backward-Euler companion models. Each switch-state combination's matrix is inverted **once and cached**, so a converter that cycles through a few topologies costs one matrix-vector product per step. |
| **Clean switching transients** | When the topology changes, that step is recomputed with backward Euler. This suppresses the trapezoidal rule's numerical ringing. Steps with no consistent switch state (zero crossing inside the step) are resolved automatically. |
| **Separate power and control circuits** | Control blocks run once per step after the power solve (one-step delay, as in PSIM). Algebraic loops are detected and reported. |
| **Sensors** | `VoltageSensor`, `CurrentSensor`, `ProbeSensor` (any quantity, e.g. `speed(M1)`). Signal-controlled sources (`V(..., signal="x")`) take control signals back into the power circuit. |
| **Control library** | Constant, sine, triangle/sawtooth carriers, square, step, ramp, gain, sum, product, divide, abs, limiter, rate limiter, comparator (with hysteresis), logic gates, **PWM**, **dead time**, integrator, **PI / PID** with anti-windup, **transfer function** (Tustin), low-pass, unit delay, moving average, RMS meter, abc↔dq transforms, PLL. |
| **C block** | `PythonBlock(outputs, inputs, fn, state)` runs arbitrary user code with persistent state, for example a complete digital controller. `Function` covers simple algebraic expressions. |
| **Digital control** | `SampleHold` (ZOH/ADC sampling, synchronous sampling), unit delay, computation-delay modelling inside a `PythonBlock`. |
| **Magnetics** | `IdealTransformer`, `Transformer` (magnetising + leakage inductance, winding resistance), `CoupledInductors` (any coupling k). |
| **Motor drive module** | `DCMotor`, fully electro-mechanical (current *and* speed are solver unknowns). Load torque can be a constant, a function `f(t, w)` or a control signal. Probes: `I`, `speed`, `rpm`, `Te`. |
| **Sources** | DC, sine, square, step, piecewise-linear, any `f(t)`, balanced `ThreePhaseSource`, current sources. |
| **SIMVIEW-like post-processing** | `mean`, `rms`, `min/max`, `ripple`, `fft`, `harmonic`, `thd`, `power` per element, `efficiency`, `power_factor`, differential voltages `V(a,b)`, time windows, multi-panel plots, spectrum plots, CSV export. |
| **Parameter sweep** | `parameter_sweep(build, {"L": [...], "fsw": [...]}, ...)`: Cartesian sweeps, optional multiprocessing, table output. |
| **AC sweep** | `ac_sweep(...)`: frequency response by **small-signal injection into the switching circuit**, so no averaged model is needed. `plot_bode`, `tf_response` for comparison. |
| **Netlist + CLI** | SPICE-like text netlists with control blocks, `.param`, `.tran`, `.plot`, `.measure`, `.window`. `python -m openpe run file.net --plot out.png --csv out.csv`. |
| **Subcircuits** | `Circuit.subcircuit(prefix, other, port_map)` reuses any circuit as a block. |

| SPICE-class feature | OpenPE implementation |
|---|---|
| **DC operating point (`.op`)** | `circuit.op()`: capacitors open, inductors short. Newton-Raphson with SPICE junction limiting, falling back to source stepping. Prints a bias table with node voltages, branch currents and device data (Vbe, Vce, gm, region...). |
| **DC sweep (`.dc`)** | `circuit.dc("V1", values)` for I-V curves, transfer curves and line regulation. Each point starts from the previous solution. |
| **AC small-signal (`.ac`)** | `circuit.ac(freqs)`: complex MNA linearised at the operating point. `ACResult` gives `mag_db`, `phase_deg`, `transfer(out, ref)`, `peak`, `bandwidth`, `corners` (−3 dB low/high) and Bode plots. Sources take `ac=` magnitude and `ac_phase=`. |
| **Transient from the bias point** | `simulate(..., initial="op")` starts from the DC solution (SPICE default). `initial="ic"` (the default here) starts from the element initial conditions. |
| **Nonlinear semiconductors** | `PNDiode` (Shockley: Is, n, series rs, junction cj, Zener breakdown bv/ibv), `Zener`, `BJT` (Ebers-Moll transport model, NPN/PNP, βF/βR, Early voltage, cje/cjc), `MOSFET` (level-1 square law, NMOS/PMOS, λ, symmetric source/drain, cgs/cgd). |
| **Controlled sources** | `VCVS` (E), `VCCS` (G), `CCCS` (F), `CCVS` (H). F and H are controlled by any branch current: voltage source, inductor, E or H source. |
| **Op-amp** | `OpAmp`: open-loop gain, single-pole gain-bandwidth product, output resistance, smooth rail saturation. |
| **Monte Carlo** | `monte_carlo(run, nominal, tolerances, n, distribution="uniform"|"gauss")` wraps any analysis. `MonteCarloResult` gives statistics, `yield_` against spec limits and histograms. |
| **Robust convergence** | Newton with junction limiting, FET step limiting, op-amp output limiting, round-off limit-cycle detection, automatic time-step subdivision when Newton fails, and source stepping for the operating point. |

---

## Installation

```bash
pip install -e ".[dev]"     # numpy + matplotlib + pytest
pytest -q                   # 74 tests, ~15 s
```

## Examples

Every script in `examples/` is self-contained and writes plots to
`examples/output/`.

| Script | What it demonstrates | Key results |
|---|---|---|
| `01_buck_closed_loop.py` | Buck with cascaded voltage/current PI loops, soft start, load step, synchronous current sampling | 12.00 V regulated through a 2→4 A load step |
| `02_inverter_spwm_thd.py` | Full-bridge inverter, unipolar SPWM, 1 µs dead time, LC filter, FFT/THD | Output THD 1.6 % (bridge voltage 57 %) |
| `03_thyristor_rectifier_sweep.py` | 6-pulse SCR rectifier with source inductance, firing-angle sweep | Vdc matches `1.35·VLL·cosα − 3ωLsId/π` within 0.1 % |
| `04_dc_motor_drive.py` | 4-quadrant H-bridge DC drive, speed + current loops, load step, speed reversal with regeneration | Speed tracks the reference, current limit holds at 20 A |
| `05_ac_sweep_buck.py` | Control-to-output Bode plot of a *switching* buck by injection | Within ~0.2 dB / 1° of the averaged model |
| `06_flyback_rcd_clamp.py` | Flyback with leakage inductance, RCD clamp, Coss | Drain peak 94 V with clamp, 202 V without |
| `07_digital_control_cblock.py` | Boost converter controlled by microcontroller-style code in a `PythonBlock` (ADC ZOH, 1-sample delay) | 24 V regulated through a load step |
| `08_boost_pfc.py` | Single-phase boost PFC, average-current mode, multiplier reference, duty feed-forward | PF 0.991, line THD 1.1 %, power balance 400 W in / 400 W out |
| `09_bjt_amplifier.py` | Common-emitter amplifier: `.op` bias table, `.ac` gain and bandwidth, transient from the bias point, THD vs input level | 36.1 dB midband gain (hand estimate 36.2 dB), 69 Hz – 459 kHz |
| `10_opamp_active_filters.py` | Sallen-Key Butterworth LPF and MFB band-pass vs analytic transfer functions, effect of op-amp GBW, rail clipping | fc within 0.01 %, band-pass Q = 5.00 |
| `11_semiconductor_curves.py` | Curve tracer: diode/Zener I-V, BJT and NMOS output characteristics, CMOS inverter VTC | Inverter threshold 2.496 V (ideal 2.5 V) |
| `12_monte_carlo_tolerance.py` | Monte Carlo of filter cut-off with 1 %/5 % parts, yield against a ±3 % spec | Yield 91.7 % (5 % caps) vs 100 % (1 % caps) |
| `13_linear_power_supply.py` | Transformer + bridge of real diodes + reservoir capacitor + Zener/BJT series regulator | 9.08 V at 0.5 A, 34 dB ripple rejection, 21 mV/V line regulation |

Netlist examples are in `circuits/`:

```bash
python -m openpe run circuits/buck_current_mode.net --plot buck.png --summary
python -m openpe run circuits/three_phase_rectifier.net
python -m openpe run circuits/ce_amplifier.net --plot ce.png    # .op + .ac + .tran
python -m openpe run circuits/opamp_sallen_key.net              # .ac
python -m openpe run circuits/diode_iv.net --plot iv.png        # .dc
python -m openpe blocks          # list control block types
```

<p align="center">
<img src="examples/output/01_buck_closed_loop.png" width="48%">
<img src="examples/output/02_inverter_spectrum.png" width="48%">
<img src="examples/output/04_dc_motor_drive.png" width="48%">
<img src="examples/output/05_ac_sweep_buck.png" width="48%">
<img src="examples/output/09_ce_amp_bode.png" width="48%">
<img src="examples/output/11_semiconductor_curves.png" width="48%">
</p>

---

## Netlist format

```text
* comment
.param vin=48 fsw=50k
Vin   in  0   DC {vin}            ; V/I: [DC v] [AC mag [ph]] [SIN(off amp f [ph]) | SQUARE v1 v2 f [duty] | STEP t v0 v1 | SIGNAL sig [gain]]
S1    in  sw  gate                ; switch: n1 n2 gate_signal [ron= roff= body_diode=1]
D1    0   sw                      ; diode: anode cathode [vf= ron= roff=]
Q1    a   k   g1                  ; thyristor
L1    sw  out 100u  ic=0
C1    out 0   220u  ic=0
T1    p1 p2 s1 s2 2 lm=200u llk=2u    ; transformer, n = Np/Ns
K1    a b c d 1m 4m 0.99              ; coupled inductors L1 L2 k
M1    a 0 ra=1 la=5m ke=0.5 j=0.01 tload=tl   ; DC motor
G1    a b c 0 400 50                  ; 3-phase source, VLL rms, f
* analog devices
D2    a k is=1e-14 n=1.8 rs=0.1 bv=5.6    ; Shockley diode (any of is/n/rs/bv/cj selects it)
Q2    c b e NPN is=1e-14 bf=150 vaf=100   ; BJT (NPN|PNP)
M2    d g s NMOS vto=1 kp=1m lambda=0.02  ; MOSFET (NMOS|PMOS)
E1    o 0 a b 10                          ; VCVS      G2 o 0 a b 1m   ; VCCS (5 fields)
F1    o 0 Vsense 2                        ; CCCS      H1 o 0 Vsense 1k ; CCVS
XU1   OPAMP inp inn out gain=200k gbw=1meg vmax=15 vmin=-15
* control blocks:  B<id> TYPE outputs... inputs... key=value
B1 VSENSOR  vo out 0
B2 ISENSOR  il L1
B3 SUM      err vref vo signs=+-
B4 PI       d err kp=0.05 ki=200 lo=0 hi=0.95
B5 PWM      gate d freq={fsw}
B6 TF       y u num=1 den=1e-3,1
.tran 0.2u 8m [method=trap|be] [start=t] [every=n] [init=ic|op]
.op                                ; DC operating point
.dc V1 0 5 0.01                    ; DC sweep: source start stop step
.ac dec 20 10 1meg                 ; AC analysis: dec|oct|lin points fstart fstop
.window 1m                         ; measurements over the last 1 ms
.plot V(out),vref I(L1)            ; comma = overlay in one panel
.measure mean V(out)               ; mean | rms | max | min | pp | thd sig f0 [cycles]
.end
```

## Conventions

* Ground is node `0` (aliases `gnd`, `GND`, `ground`).
* Element current `I(X)` flows from the element's first node to its second
  *through the element* (passive sign convention). `power(X) > 0` means
  absorbed, so a source delivering power reads negative.
* Recorded automatically: every node voltage `V(n)`, every element current,
  every control signal, plus element extras such as `speed(M1)` or `Is(T1)`.
  Differential voltages: `res["V(a,b)"]`. Voltage across an element:
  `res["VE(R1)"]`.

## How the solver works

1. Gated switches take their state from the control signals of the
   previous step.
2. The MNA system `A·x = b` is solved. `A` depends only on the switch
   states (and the integration method), so its inverse comes from a cache.
   Only `b` (sources and element history) is rebuilt every step.
3. Diodes and thyristors check their state against the solution and the
   step is repeated until the states are consistent. If the iteration
   cycles (a device current crosses zero inside the step), the
   least-violating state is accepted. Those steps are counted in
   `res.meta["stats"]["ambiguous_steps"]`.
   Nonlinear devices are solved inside this loop by Newton-Raphson. The
   linear part of the matrix stays cached and only device linearisations
   are re-stamped. Junction-voltage limiting, round-off limit-cycle
   detection and automatic step subdivision keep it converging.
4. If the topology changed, the step is redone with backward Euler (no
   trapezoidal ringing), and the first step always uses backward Euler.
   Linear solves use the cached inverse plus one step of iterative
   refinement, so even ill-conditioned circuits (for example a floating
   mains-side bridge) keep KCL and the power balance exact.
5. Element histories are updated, sensors read and the control circuit is
   evaluated. Carrier-type blocks (triangle, sawtooth, square, PWM) are
   sampled at the middle of the interval they control, which removes the
   systematic duty-cycle bias of naive sampling.

`res.meta["stats"]` reports steps, matrix factorisations, damped steps,
Newton iterations, subdivided steps and CPU time.

## Accuracy notes and limitations

* **PWM resolution is `dt · fsw`**, as in any fixed-step simulator. Use at
  least 100–200 steps per switching period. If an exact duty cycle such as
  0.25 matters, avoid a step size that puts the switching edge exactly on
  a sample.
* Backward-Euler damping at switching instants adds a small numerical loss
  in reactive elements. It is proportional to `dt` and only visible in
  precise efficiency studies; use a smaller `dt` for those.
* Switches are ideal (no switching-loss or thermal models). Conduction
  loss comes from `ron`/`vf`.
* Only DC machines are included. AC machines can be built from the
  provided dq transforms, sources and `PythonBlock`.
* Semiconductor models are textbook-level: Shockley, Ebers-Moll and
  MOSFET level 1, with linear junction capacitances. There is no
  Gummel-Poon, BSIM or temperature sweep, and no noise or pole-zero
  analysis.
* A DC operating point does not exist when a voltage source is shorted at
  DC by an inductor or a transformer winding. Start such circuits (mains
  transformers) from initial conditions instead (`initial="ic"`).

## Project layout

```
openpe/
  elements.py    power-circuit elements (R, L, C, sources, switches, magnetics, DC motor)
  devices.py     analog devices (Shockley/Zener diode, BJT, MOSFET, E/G/F/H sources, op-amp)
  control.py     control blocks (sources, math, PWM, PI/PID, TF, ZOH, transforms, PLL, ...)
  circuit.py     Circuit container, palette shortcuts, subcircuits
  simulator.py   MNA solver: transient (matrix caching, switch iteration, Newton), .op, .dc, .ac
  results.py     waveform analysis, plots, CSV
  analysis.py    parameter sweep, injection AC sweep, Monte Carlo, Bode helpers
  netlist.py     text netlist parser
  __main__.py    command-line interface
examples/        runnable showcase scripts (+ output/ plots)
circuits/        example netlists
tests/           pytest suite (analytic checks of every feature)
```

## License

MIT
