# vecmeson-generator

Amplitude-level event generator for exclusive vector-meson electroproduction

```
e p  ->  e' p' V ,     V -> h+ h-      (V = phi -> K+ K-   or   rho0 -> pi+ pi-)
```

The event distribution is driven entirely by the vector-meson **helicity amplitudes**
`T_{mu nu}` and `U_{mu nu}`, which you define in one function at the top of `generate_events.py`.
The generator builds the spin-density matrix `rho = T T^dagger`, throws events following the full
angular intensity `W(Omega)` and the chosen cross-section weighting, boosts the final state to the
lab, and writes kinematics plots and CLAS12/GEMC-ready Lund files.

## What it produces

- `LUND_files/<V>_<E>gev_<i>.lund` — CLAS12/GEMC Lund files (header line + the four final-state
  particles `e', p', h+, h-` per event), split into files of `CHUNK` events each.
- `LUND_files/<base>_columns.txt` — documents the header columns for that run.
- `LUND_files/luminosity.txt` — (multi-energy) the per-beam relative luminosity and accepted count.
- `Kin_plots/*.pdf` — particle kinematics, DVEP kinematics, input amplitudes, and observables.

## Requirements

Python 3.8+ with NumPy and Matplotlib: `pip install -r requirements.txt`

## Running

```bash
python generate_events.py                           # phi, 10.6 GeV, 20k events
MESON=rho0 N=50000 E=7.546 python generate_events.py
MESON=rho0 MULTI=1 WEIGHT=flux N=200000 python generate_events.py   # realistic multi-beam
```

Options are environment variables:

| Variable | Meaning | Default |
|----------|---------|---------|
| `MESON`  | `phi` (K+K-) or `rho0` (pi+pi-) | `phi` |
| `E`      | Beam energy [GeV] (single-energy mode) | `10.6` |
| `N`      | Events generated (target for the highest-yield beam in multi-energy) | `20000` |
| `CHUNK`  | Events per Lund file (GEMC per-file limit) | `5000` |
| `MULTI`  | `1` (or `--multi-energy`): generate at 6.535 / 7.546 / 10.6 GeV in one run | `0` |
| `LUMI`   | Multi-energy relative beam luminosities, e.g. `LUMI=2,1,1` | equal |
| `POL`    | Beam polarization degree (0 = unpolarised); each event's helicity sign is written to the Lund header | `0` |
| `BW`     | Sample the meson mass from a relativistic Breit-Wigner (1) or use the fixed pole mass (0) | `1` |
| `WEIGHT` | Yield weighting **shape**: `amp` \| `flux` \| `toy` (see below) | `amp` |
| `WEIGHTED` | `0` = unweighted accept-reject events (like real data); `1` = keep all flat events, physics weight in Lund field 10 | `0` |
| `Q2MIN` `Q2MAX` | Q^2 sampling window [GeV^2] | `1.0` `6.0` |
| `XBMIN` `XBMAX` | x_B sampling window | `0.08` `0.5` |
| `TMAX`   | Flat t' upper bound [GeV^2] | `4.0` |
| `AMP_FILE` | Path to a file defining `user_amplitudes(Q2,t)`, overrides the built-in set | (none) |
| `LUND_KIN` | Append 8 kinematic columns to the Lund header (blind-safe) | `0` |
| `LUND_TRUTH` | Also append the 16 truth amplitude columns (reveals the truth) | `1` |

> The default Q^2/x_B/t windows match the extraction's trained forward model -- **keep the defaults
> for the blind test**; change them only for other studies.

## Generator weights (two distinct knobs)

**`WEIGHT` = the yield *shape*** (what goes into the per-event physics weight `wphys`):

- `amp`  : `wphys = W(Omega; amplitudes)` -- the Q^2/t/angle dependence comes purely from your
  amplitudes (`sigma_T + eps*sigma_L` and the full angular intensity), no extra falloff.
- `flux` : `wphys = Gamma(Q^2,x,y) * W` -- also folds in the physical virtual-photon flux. **Use this
  for realistic per-beam yields / a Rosenbluth L/T separation**, since the flux carries the Q^2 and
  epsilon dependence of the rate.
- `toy`  : legacy smooth cross-section shape x `W`.

**`WEIGHTED` = how events are *kept*:**

- `WEIGHTED=0` (default): **accept-reject** on `wphys` -> **unweighted** events distributed like the
  physics, each Lund weight (field 10) `= 1`. This is what real data looks like.
- `WEIGHTED=1`: keep **all** flat-in-kinematics events and write `wphys` into Lund field 10. A
  Monte-Carlo convenience (smooth yields, no accept-reject statistics); not a model of data.

Kinematics `(Q^2, x_B, t')` are always sampled **flat**; the physics lives in `wphys`. So for
`WEIGHTED=0` the *event density* carries the physics, and for `WEIGHTED=1` the *weight* does.

## Multi-energy (Rosenbluth) mode

`MULTI=1` generates at the three beams in one run with **realistic relative yields**: a common
accept-reject threshold across beams gives accepted counts `N(E) ~ L(E) * sigma(E)`, so the relative
multi-beam normalization (the epsilon lever arm for the longitudinal/transverse separation) is
physical -- **not** a fixed count per beam. `LUMI=L1,L2,L3` sets the relative luminosities (default
equal = equal running time). Each beam is written to `LUND_files/<E>GeV/`, and `luminosity.txt`
records `beam_energy  relative_luminosity  accepted_events` for the data-analysis normalization
`N_bin(E) / L_E`. Use `WEIGHT=flux` so the yields include the flux.

## Lund file structure

**Header** -- the 10 standard CLAS12/GEMC fields, optionally followed by extra columns:

| # | field | | # | field |
|---|-------|--|---|-------|
| 1 | nParticles (=4) | | 6 | beam PID (11 = e-) |
| 2 | target A (=1) | | 7 | **beam energy** [GeV] |
| 3 | target Z (=1) | | 8 | target PID (2212 = p) |
| 4 | target polarization (=0) | | 9 | process id (=0) |
| 5 | **beam helicity** (+-1, 0 if unpol.) | | 10 | **weight** (`wphys` if `WEIGHTED=1`, else 1) |

Optional appended columns (documented per run in `<base>_columns.txt`):
- `LUND_KIN=1` -> cols 11-18: `Q2  |t|  xB  W  cos(theta)_decay  phi_decay  Phi_Trento  eps`
  (blind-safe: all reconstructable from the 4-vectors).
- `LUND_TRUTH=1` (default) -> also cols 19-34: the 16 truth amplitude components
  `T11 ReT00 ImT00 ReT01 ImT01 ReT10 ImT10 ReT1m1 ImT1m1 U11 ReU01 ImU01 ReU10 ImU10 ReU1m1 ImU1m1`
  evaluated at the event's `(Q2,|t|)`. The truth travels sealed with the events; an analysis that
  reads only the 4-vectors (+ weight) stays blind by construction. Set `LUND_TRUTH=0` for a hard
  blind (clean 10-field header), e.g. when passing the file **through GEMC** (GEMC consumes the
  header and outputs reconstructed events, so header extras are not propagated anyway).

**Particle lines** (14 fields, standard GEMC):
`index  lifetime  type  PID  parent  daughter  px  py  pz  E  mass  vx  vy  vz`
with `type = 1` (final state) and vertex at the origin. Four lines per event: `e', p', h+, h-`.

## Kinematic variables and angles

Exclusive DVEP `e(k) p(P) -> e'(k') p'(P') V`, `V -> h+ h-`, with `q = k - k'` the virtual photon.

| symbol | definition |
|--------|------------|
| `Q2`   | photon virtuality, `Q^2 = -q^2` [GeV^2] |
| `xB`   | Bjorken x, `x_B = Q^2 / (2 P.q)` |
| `nu`   | energy transfer, `nu = P.q / M = (W^2 + Q^2 - M^2)/(2M)` |
| `y`    | inelasticity, `y = nu / E` |
| `W`    | gamma*-p invariant mass, `W = sqrt((q+P)^2)` [GeV] |
| `t`    | Mandelstam momentum transfer to the meson, `t = (q - p_V)^2 < 0`; the header stores `|t|` |
| `t_min`| minimum `|t|` at the event's `(Q^2, x_B, m_V)` |
| `t'`   | reduced momentum transfer, `t' = |t| - |t_min| >= 0` (sampled flat) |
| `eps`  | virtual-photon polarization, `eps = (1 - y - gamma^2 y^2/4) / (1 - y + y^2/2 + gamma^2 y^2/4)`, `gamma = 2 M x_B / Q` |
| `m_V`  | meson invariant mass (Breit-Wigner) |

**Decay angles** -- the `h+` direction in the vector-meson **helicity frame** (rest frame of `V`,
z-axis along the meson momentum in the gamma*-p CM):
- `cos(theta)` -- polar angle of `h+`,
- `phi` -- azimuth of `h+` measured from the production plane.

**Production angle** `Phi` (Trento) -- the azimuth of the meson production plane relative to the
**lepton (scattering) plane**, in the gamma*-p CM. It carries the `sigma_LT` / `sigma_TT`
(`cos Phi`, `cos 2Phi`) interference.

## Defining your amplitudes

Everything physical lives in `user_amplitudes(Q2, t)` (or an external `AMP_FILE`):

```python
def user_amplitudes(Q2, t):          # Q2, t=|t| arrive as NumPy arrays; any closed form works
    Q = np.sqrt(Q2)
    T11  = (1.00/Q) * np.exp(-0.65*t)                       # real, transverse (dominant)
    T00  = 1.73     * np.exp(-0.55*t) + 0j                  # longitudinal
    T01  = (0.45/Q) * np.exp(-0.60*t) * np.exp(+1j*0.7)     # single-flip (complex)
    T10  = (0.15/Q) * np.exp(-0.60*t) + 0j
    T1m1 = 0.10     * np.exp(-0.60*t) * np.exp(-1j*0.4)     # double-flip (complex)
    U11  = 0.0; U01 = 0.0+0j; U10 = 0.0+0j; U1m1 = 0.0+0j   # unnatural parity (0 for a phi-like meson)
    return T11, T00, T01, T10, T1m1, U11, U01, U10, U1m1
```

### Conventions

- Meson helicity `mu` and photon helicity `nu` run over `{-1,0,+1}`; nucleon-helicity non-flip set.
- Natural parity: `T_{-mu,-nu} = +(-1)^{mu-nu} T_{mu nu}` -> independent `T11,T00,T01,T10,T1m1`.
- Unnatural parity: `U_{-mu,-nu} = -(-1)^{mu-nu} U_{mu nu}` -> independent `U11,U01,U10,U1m1`, `U00=0`.
- `T11`, `U11` are **real** (global / residual-phase reference); the rest may be complex.
- `sigma_T = 2|T11|^2 + |T01|^2`, `sigma_L = |T00|^2 + 2|T10|^2`, `R = sigma_L / sigma_T`
  (natural parity; the U amplitudes add analogously).

## Files

```
generate_events.py   generator + weights + plots + Lund writer (edit user_amplitudes)
kinematics.py        DVEP kinematics and Lorentz boosts to the lab
diehl_w.py           the angular intensity W(Omega) and SDMEs
amplitudes.py        amplitudes -> spin-density matrix (rho = T T^dagger)
```

The three physics modules are self-contained (NumPy only) and shared with the amplitude-extraction
analysis they were written for.
