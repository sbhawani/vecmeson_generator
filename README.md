# vecmeson-generator

Amplitude-level event generator for exclusive vector-meson electroproduction

```
e p  ->  e' p' V ,     V -> h+ h-      (V = phi -> K+ K-   or   rho0 -> pi+ pi-)
```

The event distribution is driven entirely by the vector-meson **helicity amplitudes**
`T_{mu nu}` and `U_{mu nu}`, which you define at the top of `generate_events.py`. The
generator builds the spin-density matrix `rho = T T^dagger`, throws events following the full
angular intensity `W(Omega)` and a smooth production cross section, boosts the final state to the
lab, and writes kinematics plots and a Lund event file ready for a detector simulation.

## What it produces

- `Kin_plots/particle_kinematics_<V>.pdf` — lab momentum, polar angle theta, and azimuth phi of
  each final-state particle: rows `(e', p', h+, h-)` x columns `(momentum, theta, phi)`.
- `Kin_plots/dvep_kinematics_<V>.pdf` — Q^2, x_B, W, nu, -t, -t_min, t', epsilon, and the decay
  angles cos(theta), phi, Phi.
- `Kin_plots/amplitudes_<V>.pdf` — the real and imaginary parts of all nine T and U amplitude
  components versus |t| at a reference Q^2 (same layout as the extraction paper), plus sigma_T,
  sigma_L and R = sigma_L / sigma_T.
- `LUND_files/<V>_<E>gev_<i>.lund` — CLAS12/GEMC Lund files (header line + the four final-state
  particles e', p', h+, h- per event). The events are split into files of at most 5000 events
  each (GEMC's per-file limit), so `N` events give `ceil(N / 5000)` files numbered from 0.

## Requirements

Python 3.8+ with NumPy and Matplotlib:

```bash
pip install -r requirements.txt
```

## Running

```bash
python generate_events.py                          # phi, 10.6 GeV, 20k events
MESON=rho0  N=50000  E=7.546  python generate_events.py
MESON=phi   N=100000 E=6.535  POL=0.85 python generate_events.py
```

Options are passed as environment variables (or edit the USER SECTION directly):

| Variable | Meaning | Default |
|----------|---------|---------|
| `MESON`  | `phi` (K+K-) or `rho0` (pi+pi-) | `phi` |
| `E`      | Beam energy [GeV] | `10.6` |
| `N`      | Total number of events generated | `20000` |
| `CHUNK`  | Events per Lund file (GEMC limit); gives `N/CHUNK` files | `5000` |
| `POL`    | Beam polarization degree (0 = unpolarised, 1 = polarised). Each event's helicity **sign** (+1/-1, or 0 if unpolarised) is written to its Lund header. | `0` |
| `BW`     | Sample the meson mass from a relativistic Breit-Wigner line shape (1) or use the fixed pole mass (0). | `1` |

The meson invariant mass is drawn from a relativistic Breit-Wigner with a mass-dependent P-wave
width and a Blatt-Weisskopf L=1 barrier (pole mass and width set per meson in the `MESONS` table).
This gives the broad $\rho^0\to\pi\pi$ bump and the narrow, K$^+$K$^-$-threshold-skewed $\phi\to KK$
peak automatically; set `BW=0` for a fixed pole mass.

### Multi-energy (Rosenbluth) mode

Pass `--multi-energy` (or `MULTI=1`) to generate `N` events at **each** of the three beams
`6.535, 7.546, 10.6` GeV in one run:

```bash
python generate_events.py --multi-energy           # 3 x N events across all three beams
N=50000 MULTI=1 python generate_events.py
```

The events from the three energies are pooled, shuffled, and written to
`LUND_files/<V>_multiE_<i>.lund` (still 5000 events per file). Each event's Lund header carries
its **own** beam energy, so the three energies share the same files. This provides the epsilon
lever arm needed for a longitudinal/transverse (Rosenbluth) separation.

## Defining your amplitudes

Everything physical lives in one function at the top of `generate_events.py`:

```python
def user_amplitudes(Q2, t):
    Q = np.sqrt(Q2)
    T11  = (1.00 / Q) * np.exp(-0.65 * t)                       # real
    T00  = (1.73)     * np.exp(-0.55 * t) + 0j
    T01  = (0.45 / Q) * np.exp(-0.60 * t) * np.exp(+1j * 0.7)
    T10  = (0.15 / Q) * np.exp(-0.60 * t) + 0j
    T1m1 = (0.10)     * np.exp(-0.60 * t) * np.exp(-1j * 0.4)
    U11  = 0.0
    U01  = 0.0 + 0j
    U10  = 0.0 + 0j
    U1m1 = 0.0 + 0j
    return T11, T00, T01, T10, T1m1, U11, U01, U10, U1m1
```

`Q2` and `t = |t|` arrive as NumPy arrays, so any closed-form dependence works. Set the unnatural
amplitudes `U..` to zero for a natural-parity-dominated meson such as the phi.

## Conventions

- Meson helicity `mu` and photon helicity `nu` run over `{-1, 0, +1}`; the amplitudes are the
  nucleon-helicity non-flip set.
- Natural parity: `T_{-mu,-nu} = +(-1)^{mu-nu} T_{mu nu}`  ->  independent `T11, T00, T01, T10, T1m1`.
- Unnatural parity: `U_{-mu,-nu} = -(-1)^{mu-nu} U_{mu nu}`  ->  independent `U11, U01, U10, U1m1`,
  with `U00 = 0` by parity.
- `T11` and `U11` are taken **real** (global-phase / residual-phase reference); the other
  amplitudes may be complex.
- Cross sections: `sigma_T = 2|T11|^2 + |T01|^2`, `sigma_L = 2|T10|^2 + |T00|^2`, `R = sigma_L/sigma_T`.
- Decay angles are the vector-meson helicity-frame angles `(cos theta, phi)` of the `h+`, and `Phi`
  is the production-plane azimuth.

## Lund format

Header (10 fields): `nParticles  targetA  targetZ  targetPol  beamPol  beamPID  beamE  targetPID
processID  weight`.

Particle (14 fields): `index  lifetime  type  PID  parent  daughter  px  py  pz  E  mass  vx  vy  vz`
with `type = 1` (final state) and the vertex at the origin. Change the vertex or the second column
(here `lifetime = 0`) in `write_lund()` if your simulation expects a different layout.

## Files

```
generate_events.py   the generator + plotting + Lund writer (edit the USER SECTION)
kinematics.py        DVEP kinematics and Lorentz boosts to the lab
diehl_w.py           the angular intensity W(Omega) and SDMEs
amplitudes.py        amplitudes -> spin-density matrix (rho = T T^dagger)
```

The three physics modules are self-contained (NumPy only) and are shared with the amplitude
extraction analysis they were written for.
