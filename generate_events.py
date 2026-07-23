#!/usr/bin/env python3
# =============================================================================
# INDEPENDENT amplitude-level event generator for exclusive vector-meson
# electroproduction:   e p -> e' p' V,   V -> h+ h-   (V = phi->K+K- or rho0->pi+pi-)
#
# Driven entirely by the helicity amplitudes T_{mu nu}, U_{mu nu} defined in the
# USER SECTION below (edit them freely).  It writes:
#   Kin_plots/particle_kinematics_<V>.pdf  rows(e,p,h+,h-) x cols(mom,theta,phi)
#   Kin_plots/dvep_kinematics_<V>.pdf      Q2,xB,W,nu,-t,-tmin,t',eps,cos,phi,Phi
#   Kin_plots/amplitudes_<V>.pdf           |T|,|U|, sigma_T, sigma_L, R  vs |t|
#   LUND_files/<V>_<E>gev.lund             Lund event file (e',p',h+,h-)
#
# Options are KEY=VALUE tokens, given EITHER on the command line OR as environment variables:
#   python generate_events.py                                   # phi, defaults
#   python generate_events.py MESON=rho0 N=50000                # <-- command-line form
#   MESON=rho0 N=50000 python generate_events.py                # <-- equivalent env form
#   python generate_events.py --help                            # list all options
# =============================================================================
import os, sys

# --- accept KEY=VALUE options on the command line (argv) as well as via the environment ---
# Any `KEY=VALUE` token on the command line is copied into os.environ BEFORE the settings below
# are read, so the two forms are equivalent and command-line tokens win over the environment.
# (This must run before the os.environ.get(...) reads in the USER SECTION.)
_OPTIONS = {  # name -> one-line help, for --help
    "MESON": "vector meson: rho0 | phi", "N": "number of events", "E": "beam energy [GeV]",
    "BEAM": "lepton beam: e | mu", "WEIGHT": "yield weight: flux | amp | vpk | hand | toy",
    "Q2MIN": "Q^2 lower [GeV^2]", "Q2MAX": "Q^2 upper [GeV^2]", "XBMIN": "x_B lower",
    "XBMAX": "x_B upper", "TMAX": "flat t' upper bound [GeV^2]", "POL": "beam helicity magnitude",
    "BW": "sample Breit-Wigner lineshape: 1 | 0", "CHUNK": "events per Lund file",
    "AMP_FILE": "path to a file defining user_amplitudes(Q2, xB, t)",
    "SEED": "random seed (int); omit for a fresh seed (different events) each run",
    "LUND_KIN": "append 8 kinematic columns (blind-safe): 1 | 0",
    "LUND_TRUTH": "append 16 truth-amplitude columns: 1 | 0",
    "MULTI": "multi-energy mode: 1 | 0 (or pass --multi-energy)",
    "LUMI": "relative beam luminosities for MULTI, e.g. 2,1,1",
    "WEIGHTED": "keep all flat events with physics weight in Lund field 10: 1 | 0",
}
if "--help" in sys.argv or "-h" in sys.argv:
    print("Usage: python generate_events.py [KEY=VALUE ...] [--multi-energy]\n\n"
          "Options (KEY=VALUE, on the command line or as environment variables):")
    for _k, _h in _OPTIONS.items():
        print(f"  {_k:<11} {_h}")
    print("\nExample:\n  python generate_events.py MESON=rho0 N=60000 Q2MIN=1.0 Q2MAX=9.0 "
          "XBMIN=0.09 XBMAX=0.68 TMAX=5.5")
    sys.exit(0)
for _tok in sys.argv[1:]:
    if "=" in _tok and not _tok.startswith("-"):
        _k, _v = _tok.split("=", 1)
        os.environ[_k] = _v
        if _k not in _OPTIONS:
            print(f"[warn] unknown option '{_k}' (ignored by the settings; run --help for the list)",
                  file=sys.stderr)

import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
plt.rcParams.update({"figure.dpi": 110, "font.size": 11, "axes.grid": True, "grid.alpha": 0.25})
from kinematics import _boost, _perp_axes, cross_section, M, ME
from diehl_w import UNAMES, W as W_ang, sigma_T, sigma_L
from amplitudes import amp_to_u28_batch
PI = np.pi

# ============================ USER SECTION ===================================
MESON       = os.environ.get("MESON", "phi")          # "phi" or "rho0"
BEAM_ENERGY = float(os.environ.get("E", "10.6"))       # GeV
# Lepton beam: "e" (electron, JLab/CLAS12) or "mu" (muon, COMPASS-like). At these energies the lepton
# mass is negligible; the meson production is via the virtual photon and is beam-flavor-independent.
BEAM = os.environ.get("BEAM", "e").lower()
M_LEP = ME if BEAM == "e" else 0.1056584               # beam lepton mass [GeV]
LEP_PID = 11 if BEAM == "e" else 13                    # PDG id of beam / scattered lepton
N_EVENTS    = int(os.environ.get("N", "20000"))        # total events generated
EVENTS_PER_FILE = int(os.environ.get("CHUNK", "5000")) # events per Lund file (GEMC limit); N/CHUNK files
BEAM_POL    = float(os.environ.get("POL", "0.0"))      # beam helicity magnitude (0 = unpolarised)
Q2_REF      = 2.6                                       # Q^2 [GeV^2] for the amplitude-vs-|t| plot
XB_REF      = 0.3                                       # x_B for the amplitude-vs-|t| plot (amps now x_B-dependent)
# Random seed: omit for a FRESH seed each run (different events every time); set SEED=<int> to
# reproduce a run.  The seed actually used is printed at startup so any run can be reproduced.
_seed_env   = os.environ.get("SEED", "")
SEED        = int(_seed_env) if _seed_env not in ("", "none", "None", "random") \
              else int.from_bytes(os.urandom(4), "little")
MULTI_ENERGIES = [6.535, 7.546, 10.6]                  # beams used in multi-energy mode
MULTI       = ("--multi-energy" in sys.argv) or (os.environ.get("MULTI", "0") not in ("0", "", "false", "False"))
E_LABEL     = ("multi-energy " + "/".join(f"{e:g}" for e in MULTI_ENERGIES) + " GeV") if MULTI else f"$E={BEAM_ENERGY}$ GeV"

MESONS = {  # vector-meson pole mass, width, decay-hadron mass, PDG ids, labels
    "phi":  dict(MV=1.019461, width=0.004249, MH=0.493677, pid_hp=+321, pid_hm=-321, hp="K+",  hm="K-",  htex=r"K^+K^-"),
    "rho0": dict(MV=0.775260, width=0.149100, MH=0.139570, pid_hp=+211, pid_hm=-211, hp="pi+", hm="pi-", htex=r"\pi^+\pi^-"),
}
BW_MASS = os.environ.get("BW", "1") not in ("0", "", "false", "False")   # sample the Breit-Wigner line shape
# How the Q^2/x/t event YIELD is weighted (kinematics are sampled FLAT, so this alone sets the shape).
# NOTE: the yield weight sets only the KINEMATIC density; it does NOT enter the decay-angle distribution
# (the SDMEs/amplitudes) -- the flux cancels there. So 'amp' is equally valid for extraction/acceptance MC;
# the default is 'flux' only so that a bare run yields a PHYSICAL Q^2 spectrum.
#   flux : x the Diehl virtual-photon flux (y^2/(1-eps))(1-xB)/xB/Q^2 -- the CORRECT flux for the
#          Diehl W(Omega) used here (arXiv:0704.1565)                                          (default)
#   amp  : W(Omega) only  -> Q^2/t dependence purely from your amplitudes, NO flux. Fine for extraction
#          (the flux cancels in the angular fit), but the Q^2 spectrum is much harder than physical
#          (<Q^2> ~ 2.9 vs ~ 2.0 GeV^2) -- do not use it for yields or Q^2 plots.
#   vpk  : x vpK's dsigma_3fold (= Diehl flux x (1+eps)) -- exact vpK cross-check ONLY, not physical:
#          vpK's placeholder dsigmaT=dsigmaL=1 makes the (1+eps) a stand-in for the T/L that our
#          W(Omega) already carries, so this DOUBLE-COUNTS T/L. Use it only to reproduce the vpK study.
#   hand : x the Hand flux ~ (1-y)K/(Q^2(1-eps)) -- DEPRECATED, inconsistent with the Diehl W
#   toy  : x a smooth legacy toy cross section
WEIGHT = os.environ.get("WEIGHT", "flux")
WEIGHTED = os.environ.get("WEIGHTED", "0") not in ("0", "", "false", "False")  # physical-yield (weighted) events
# Header columns beyond the 10 standard Lund fields:
#   LUND_KIN=1   -> +8 kinematics (Q2,|t|,xB,W,cos_theta,phi,Phi,eps), blind-safe (reconstructable).
#   LUND_TRUTH=1 -> +16 TRUTH amplitude values (T,U re/im).  DEFAULT ON: the truth travels sealed in
#                  the file for trivial unblinding; the extraction reads only the 4-vectors + weight
#                  (never cols 11+), so it stays blind by construction.  Set LUND_TRUTH=0 for a hard
#                  blind (amplitudes not written at all).
LUND_TRUTH = os.environ.get("LUND_TRUTH", "1") not in ("0", "", "false", "False")
LUND_KIN = LUND_TRUTH or os.environ.get("LUND_KIN", "0") not in ("0", "", "false", "False")
# Kinematic sampling windows (FLAT sampling; the physics shape comes from the weight). Defaults match
# the extraction's trained forward model -- for the blind test keep the defaults; override via env for
# other studies:  Q2MIN Q2MAX [GeV^2],  XBMIN XBMAX,  TMAX = flat t' upper bound [GeV^2].
Q2MIN = float(os.environ.get("Q2MIN", "1.0")); Q2MAX = float(os.environ.get("Q2MAX", "6.0"))
XBMIN = float(os.environ.get("XBMIN", "0.08")); XBMAX = float(os.environ.get("XBMAX", "0.5"))
TMAX = float(os.environ.get("TMAX", "4.0"))

# --- Helicity amplitudes as functions of (Q^2, |t|).  EDIT THESE. ------------
# Conventions (unpolarised nucleon-helicity non-flip):
#   T_{-mu,-nu} = +(-1)^{mu-nu} T_{mu nu}   (natural parity)
#   U_{-mu,-nu} = -(-1)^{mu-nu} U_{mu nu}   (unnatural parity),  U_{00} = 0
#   T11 and U11 are taken REAL (global-phase / residual-phase reference).
#   Everything else may be complex.  Return 9 amplitudes (T then U).
def user_amplitudes(Q2, xB, t):
    Q = np.sqrt(Q2)
    # --- x_B dependence ----------------------------------------------------------------
    # Each amplitude carries its OWN x_B factor, exactly as it already carries its own Q and
    # t dependence.  Edit each factor independently.  They are seeded with x_B^2 (1 - x_B) on
    # every amplitude, which reproduces a single COMMON factor -- change any one so T and U
    # differ in x_B.  NOTE: a common factor cancels out of R = sigma_L/sigma_T and out of the
    # angular / SDME distributions (it only reshapes the overall yield vs x_B); giving the
    # amplitudes DIFFERENT x_B forms is what makes R and the SDMEs genuinely x_B-dependent.
    fT00  = xB**2 * (1 - xB)
    fT11  = xB**2 * (1 - xB)
    fT01  = xB**2 * (1 - xB)
    fT10  = xB**2 * (1 - xB)
    fT1m1 = xB**2 * (1 - xB)
    fU11  = xB**2 * (1 - xB)
    fU01  = xB**2 * (1 - xB)
    fU10  = xB**2 * (1 - xB)
    fU1m1 = xB**2 * (1 - xB)
    # --- amplitudes (magnitude / Q / t as before, now x_B-weighted per amplitude) -------
    T11  = (1.00 / Q) * np.exp(-0.65 * t) * fT11                      # real, transverse (dominant)
    T00  = (1.73)     * np.exp(-0.55 * t) * fT00  + 0j                # longitudinal
    T01  = (0.45 / Q) * np.exp(-0.60 * t) * fT01 * np.exp(+1j * 0.7)  # single-flip (SCHC-violating)
    T10  = (0.15 / Q) * np.exp(-0.60 * t) * fT10  + 0j                # single-flip
    T1m1 = (0.10)     * np.exp(-0.60 * t) * fT1m1 * np.exp(-1j * 0.4) # double-flip
    U11  = 0.0 * fU11                                                 # real; unnatural parity
    U01  = 0.0 * fU01 + 0j
    U10  = 0.0 * fU10 + 0j
    U1m1 = 0.0 * fU1m1 + 0j
    return T11, T00, T01, T10, T1m1, U11, U01, U10, U1m1
# =============================================================================
# Optional external amplitudes: AMP_FILE=/path/to/amps.py defining user_amplitudes(Q2, xB, t)
# overrides the built-in set above (e.g. for independent blind-test samples), leaving the
# default untouched.  A LEGACY file defining user_amplitudes(Q2, t) (no x_B) is still accepted --
# it is called with (Q2, t) automatically, so old amp files keep working.
_AMP_FILE = os.environ.get("AMP_FILE", "")
if _AMP_FILE:
    import importlib.util as _ilu, inspect as _inspect
    _spec = _ilu.spec_from_file_location("user_amps", _AMP_FILE)
    _mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
    _ext = _mod.user_amplitudes
    _nargs = len(_inspect.signature(_ext).parameters)
    if _nargs >= 3:
        user_amplitudes = _ext                                    # new-style (Q2, xB, t)
    else:
        user_amplitudes = lambda Q2, xB, t, _f=_ext: _f(Q2, t)    # legacy (Q2, t) amp file
        print(f"[amplitudes] AMP_FILE user_amplitudes takes {_nargs} args -> called as (Q2, t), "
              f"x_B ignored", flush=True)
    print(f"[amplitudes] external AMP_FILE={_AMP_FILE}", flush=True)


def amps_to_params(Q2, xB, t):
    """User amplitudes -> the 16-real-parameter vector (T11,U11 real; U00=0)."""
    T11, T00, T01, T10, T1m1, U11, U01, U10, U1m1 = user_amplitudes(Q2, xB, t)
    z = np.zeros(np.broadcast(np.asarray(Q2), np.asarray(xB), np.asarray(t)).shape, float)
    re = lambda x: np.real(x) + z; im = lambda x: np.imag(x) + z
    return np.stack([re(T11), re(T00), im(T00), re(T01), im(T01), re(T10), im(T10),
                     re(T1m1), im(T1m1), re(U11), re(U01), im(U01),
                     re(U10), im(U10), re(U1m1), im(U1m1)], axis=-1)


# ISSUE(vpk-comp #8): Blatt-Weisskopf L=1 barrier lineshape; vpK uses a Jackson running-width BW (broader
# high-mass tail). Match the lineshape for the final comparison. See ISSUES_vpk_comparison.md.
def sample_meson_mass(rng, n, M0, Gamma, mh):
    """Draw n meson invariant masses from a relativistic Breit-Wigner with a mass-dependent
    P-wave width (V -> h+h-, L=1). Handles both the broad rho0 and the narrow, KK-threshold-
    skewed phi automatically. Gamma<=0 returns the fixed pole mass."""
    if Gamma <= 0 or not BW_MASS:
        return np.full(n, float(M0))
    m_lo = 2.0*mh + 1e-4; m_hi = M0 + 8.0*Gamma
    p0 = np.sqrt(M0**2/4 - mh**2); R = 5.0                       # meson radius ~1 fm (GeV^-1)
    def intensity(m):
        p = np.sqrt(np.clip(m**2/4 - mh**2, 0.0, None))
        bw = (1.0 + (p0*R)**2)/(1.0 + (p*R)**2)                  # Blatt-Weisskopf L=1 barrier factor
        Gm = Gamma*(M0/m)*(p/p0)**3*bw                           # mass-dependent P-wave width
        return m**2 * M0 * Gm / ((m**2 - M0**2)**2 + M0**2 * Gm**2)
    Imax = 1.05*intensity(np.linspace(m_lo, m_hi, 4000)).max()
    out = np.empty(n); filled = 0
    while filled < n:
        m = rng.uniform(m_lo, m_hi, 2*(n-filled)+16)
        m = m[rng.uniform(0.0, Imax, len(m)) < intensity(m)]
        take = min(len(m), n-filled); out[filled:filled+take] = m[:take]; filled += take
    return out


def throw(E, n_pool, rng, MV, GV, MH):
    """Full DVEP kinematics -> per-event production vars, decay angles, lab 4-vectors of
    e', p', h+, h-, and the physics weight wphys = sigma(Q2,xB,t') * W(Omega; amplitudes).
    MV is the pole mass, GV the width (per-event mass drawn from the Breit-Wigner)."""
    k = np.array([E, 0, 0, np.sqrt(E**2 - M_LEP**2)])
    mv = sample_meson_mass(rng, n_pool, MV, GV, MH)              # per-event invariant mass
    Q2 = rng.uniform(Q2MIN, Q2MAX, n_pool)                      # FLAT: shape set by the weight below
    xB = rng.uniform(XBMIN, XBMAX, n_pool); tprime = rng.uniform(0.0, TMAX, n_pool)   # FLAT in t'
    nu = Q2/(2*M*xB); y = nu/E; W2 = M*M + 2*M*nu - Q2; Ep = E - nu
    cose = 1 - Q2/(2*E*np.clip(Ep, 1e-6, None))
    # FIXED(vpk-comp #7a): honour TMAX (was a hardcoded 'tprime < 4' that silently overrode TMAX>4).
    ok = (y > 0) & (y < 0.99) & (W2 > (M+mv)**2) & (Ep > 0.3) & (np.abs(cose) < 1) & (tprime < TMAX)
    Q2, xB, tprime, nu, y, W2, Ep, cose, mv = (v[ok] for v in (Q2, xB, tprime, nu, y, W2, Ep, cose, mv))
    Wm = np.sqrt(W2); gam = 2*M*xB/np.sqrt(Q2)
    eps = (1 - y - 0.25*gam**2*y**2)/(1 - y + 0.5*y**2 + 0.25*gam**2*y**2)
    sine = np.sqrt(np.clip(1 - cose**2, 0, None)); phe = rng.uniform(-PI, PI, len(Q2))
    kp = np.stack([Ep, Ep*sine*np.cos(phe), Ep*sine*np.sin(phe), Ep*cose], 1)
    k4 = np.tile(k, (len(Q2), 1)); q = k4 - kp
    ptar = np.tile([M, 0, 0, 0], (len(Q2), 1)); Wsys = q + ptar
    beta = Wsys[:, 1:]/Wsys[:, 0:1]; qcm = _boost(q, -beta)
    pg = np.linalg.norm(qcm[:, 1:], axis=1); Eg = qcm[:, 0]
    Ephi = (W2 + mv**2 - M**2)/(2*Wm); pphi = np.sqrt(np.clip(Ephi**2 - mv**2, 0, None))
    tmin = -Q2 + mv**2 - 2*(Eg*Ephi - pg*pphi); t = tmin - tprime
    cosst = (t + Q2 - mv**2 + 2*Eg*Ephi)/(2*pg*pphi); good = np.abs(cosst) <= 1
    (Q2, xB, nu, y, Wm, eps, kp, q, Wsys, beta, qcm, pg, Eg, Ephi, pphi, tprime, t, tmin, cosst, mv) = (
        v[good] for v in (Q2, xB, nu, y, Wm, eps, kp, q, Wsys, beta, qcm, pg, Eg, Ephi, pphi,
                          tprime, t, tmin, cosst, mv))
    N = len(Q2); sinst = np.sqrt(1 - cosst**2); phipr = rng.uniform(-PI, PI, N)
    zc = qcm[:, 1:]/pg[:, None]
    # Production-azimuth reference = the LEPTON plane (scattered e- component perpendicular to the
    # gamma* in the CM), so phipr is the physical (Trento) angle between the lepton and hadron
    # planes.  Using an arbitrary lab axis here instead (e.g. _perp_axes) decouples the W(Phi)
    # sigma_LT/sigma_TT (cos Phi, cos 2Phi) modulation from the lepton plane and washes it out.
    kpcm = _boost(kp, -beta); lp = kpcm[:, 1:]
    e1 = lp - np.sum(lp*zc, axis=1, keepdims=True)*zc
    e1 /= np.clip(np.linalg.norm(e1, axis=1, keepdims=True), 1e-9, None)
    e2 = np.cross(zc, e1)
    dirphi = cosst[:, None]*zc + sinst[:, None]*(np.cos(phipr)[:, None]*e1 + np.sin(phipr)[:, None]*e2)
    phicm = np.concatenate([Ephi[:, None], pphi[:, None]*dirphi], 1); phi4 = _boost(phicm, beta)
    prot = Wsys - phi4
    zhel = dirphi; yhel = np.cross(zc, zhel)
    yhel /= np.clip(np.linalg.norm(yhel, axis=1, keepdims=True), 1e-9, None); xhel = np.cross(yhel, zhel)
    pK = np.sqrt(np.clip(mv**2/4 - MH**2, 0, None)); EK = np.sqrt(pK**2 + MH**2)   # per-event (mv)
    CosTh = rng.uniform(-1, 1, N); Phi = rng.uniform(-PI, PI, N); sinT = np.sqrt(1 - CosTh**2)
    kdir = CosTh[:, None]*zhel + sinT[:, None]*(np.cos(Phi)[:, None]*xhel + np.sin(Phi)[:, None]*yhel)
    Hp_rest = np.concatenate([EK[:, None],  pK[:, None]*kdir], 1)
    Hm_rest = np.concatenate([EK[:, None], -pK[:, None]*kdir], 1)
    bphicm = dirphi*(pphi/Ephi)[:, None]
    Hp = _boost(_boost(Hp_rest, bphicm), beta); Hm = _boost(_boost(Hm_rest, bphicm), beta)
    # Randomise the overall event azimuth about the beam: the unpolarised cross section is
    # invariant under rotations about z, so the lab azimuth is arbitrary. This enforces exact
    # rotational invariance (flat lab phi for every particle) and leaves all polar angles,
    # invariant masses, and the rest-frame decay/production angles unchanged.
    psi = rng.uniform(-PI, PI, N); cz, sz = np.cos(psi), np.sin(psi)
    def _rotz(p4):
        out = p4.copy()
        out[:, 1] = p4[:, 1]*cz - p4[:, 2]*sz; out[:, 2] = p4[:, 1]*sz + p4[:, 2]*cz
        return out
    kp, prot, Hp, Hm = _rotz(kp), _rotz(prot), _rotz(Hp), _rotz(Hm)
    hsign = rng.choice([-1.0, 1.0], N) if BEAM_POL > 0 else np.zeros(N)  # per-event beam helicity: +1/-1 (0 if unpol.)
    heli = BEAM_POL * hsign                                              # W uses (polarization degree) x (sign)
    u = amp_to_u28_batch(amps_to_params(Q2, xB, -t)); ud = {nm: u[:, i] for i, nm in enumerate(UNAMES)}
    Wp = np.nan_to_num(np.clip(W_ang(CosTh, Phi, phipr, eps, heli, ud), 0, None))   # |amplitude|^2 x W_SW(Omega)
    if WEIGHT == "toy":
        wt = cross_section(Q2, xB, tprime)                      # legacy smooth toy shape
    elif WEIGHT == "flux":
        # CORRECT flux for our Diehl W(Omega): the Diehl-Sapeta leptonic virtual-photon flux
        #   Gamma ~ (y^2/(1-eps)) * (1-xB)/xB * (1/Q^2)          [arXiv:0704.1565].
        # This is the flux the u-matrix / W(Omega) normalization is DEFINED against (NOT the Hand flux).
        # No extra (1+eps): our W already carries sigma_T + eps*sigma_L, so adding it would double-count T/L.
        wt = (y**2/np.clip(1.0-eps, 1e-3, None))*(1.0-xB)/np.clip(xB, 1e-3, None)/Q2
    elif WEIGHT == "vpk":
        # Exact reproduction of vpK's dsigma_3fold = Diehl flux x (dsigmaT+eps*dsigmaL) with placeholders
        # dsigmaT=dsigmaL=1 -> x(1+eps).  For a bit-for-bit vpK cross-check ONLY; the (1+eps) is redundant
        # with our W, so this is not the physical flux -- use WEIGHT=flux for physics.
        wt = (y**2/np.clip(1.0-eps, 1e-3, None))*(1.0-xB)/np.clip(xB, 1e-3, None)/Q2*(1.0+eps)
    elif WEIGHT == "hand":
        # (DEPRECATED) Hand equivalent-photon transverse flux ~ (1-y)*K/(Q^2(1-eps)), K=(W^2-M^2)/2M.
        # A different flux convention -- INCONSISTENT with the Diehl W(Omega) used here; kept for reference.
        Kf = (Wm**2 - M**2)/(2*M)
        wt = (1.0 - y)*Kf/(Q2*np.clip(1.0 - eps, 1e-3, None))
    else:                                                       # "amp": Q^2/t dependence from the amplitudes only
        wt = 1.0
    wphys = np.nan_to_num(wt*Wp)
    return dict(Q2=Q2, xB=xB, nu=nu, W=Wm, tprime=tprime, absT=-t, tmin=-tmin, eps=eps,
                CosTh=CosTh, phi=Phi, Phi=phipr, e=kp, p=prot, hp=Hp, hm=Hm, wphys=wphys,
                Ebeam=np.full(len(Q2), E), hsign=hsign, mV=mv)


def generate(E, n_target, MV, GV, MH, rng, wmax=None):
    """WEIGHTED=1: keep ALL flat-in-kinematics events, each carrying its physics weight wphys
    (written to the Lund weight field) -> physical yields with NO accept-reject clipping.
    Otherwise: accept-reject on wphys -> n_target unweighted events.  wmax overrides the accept-reject
    threshold with a COMMON value (so multi-energy relative yields are preserved -- see generate_multi)."""
    keep = {k: [] for k in ("Q2", "xB", "nu", "W", "tprime", "absT", "tmin", "eps",
                            "CosTh", "phi", "Phi", "e", "p", "hp", "hm", "Ebeam", "hsign", "mV", "wphys")}
    have = 0
    while have < n_target:
        d = throw(E, max(200000, 4*(n_target - have)), rng, MV, GV, MH)
        if WEIGHTED:
            acc = np.ones(len(d["wphys"]), bool)                # keep all (flat, weighted)
        else:
            Wm = wmax if wmax is not None else np.percentile(d["wphys"], 99.9)
            acc = rng.uniform(0, Wm, len(d["wphys"])) < d["wphys"]
        for k in keep: keep[k].append(d[k][acc])
        have += int(acc.sum())
        print(f"  ... {have}/{n_target}", flush=True)
    ev = {k: np.concatenate(v)[:n_target] for k, v in keep.items()}
    return ev


def generate_multi(energies, meta, rng, lumis, n_events):
    """REALISTIC unweighted multi-energy generation, matching how real data looks: a COMMON
    accept-reject threshold across all beams + accepted counts N(E) ~ L(E)*sigma(E), so the
    RELATIVE multi-energy yields (the L/T Rosenbluth lever) are physical.  n_events targets the
    highest-yield beam.  Returns (list of per-energy ev dicts, list of accepted counts)."""
    MV, GV, MH = meta["MV"], meta["width"], meta["MH"]
    if WEIGHT != "flux":
        print(f"[warn] WEIGHT={WEIGHT}: for realistic per-energy YIELDS use WEIGHT=flux "
              f"(virtual-photon flux x |amp|^2); the L/T lever needs the flux.", flush=True)
    cals = [throw(E, 400000, rng, MV, GV, MH) for E in energies]           # calibration pass
    wmax = 1.05 * max(np.percentile(c["wphys"], 99.9) for c in cals)       # COMMON threshold
    fracs = [float(np.mean(np.clip(c["wphys"] / wmax, 0, 1))) for c in cals]   # accept frac ~ sigma(E)
    yields = [L * f for L, f in zip(lumis, fracs)]; ymax = max(yields)
    evs, counts = [], []
    for E, y in zip(energies, yields):
        n_acc = max(1, int(round(n_events * y / ymax)))                   # N(E) ~ L(E)*sigma(E)
        print(f"  E={E} GeV: target {n_acc} accepted (relative yield {y/ymax:.3f}) ...", flush=True)
        ev = generate(E, n_acc, MV, GV, MH, rng, wmax=wmax)
        evs.append(ev); counts.append(len(ev["Q2"]))
    return evs, counts


# ------------------------------------------------------------------- Lund -----
# extra per-event header columns appended after the 10 standard Lund fields.
# Standard Lund parsers read the first 10 fields and ignore the rest.
LUND_KIN_COLS = ["Q2", "abs_t", "xB", "W", "CosTheta_decay", "phi_decay", "Phi_Trento", "eps"]
LUND_AMP_COLS = ["T11", "ReT00", "ImT00", "ReT01", "ImT01", "ReT10", "ImT10", "ReT1m1", "ImT1m1",
                 "U11", "ReU01", "ImU01", "ReU10", "ImU10", "ReU1m1", "ImU1m1"]


def write_lund(ev, meta, outdir, base):
    """CLAS12/GEMC Lund, split into files of at most EVENTS_PER_FILE events (GEMC limit),
    named <base>_0.lund, <base>_1.lund, ...  particle lines(14):
    idx lifetime type pid parent daughter px py pz E mass vx vy vz.
    HEADER = the 10 standard Lund fields
      [nparticles=4, A=1, Z=1, target_pol=0, beam_helicity, beam_type=11, E_beam, target_pid=2212,
       process_id=0, weight=wphys]
    DEFAULT header = the CLEAN 10 standard Lund fields (blind, standard).  Optional appended columns:
      LUND_KIN=1   -> cols 11..18: Q2, |t|, xB, W, cos(theta)_decay, phi_decay, Phi_Trento, eps
                      (reconstructable from the 4-vectors -- blind-safe).
      LUND_TRUTH=1 -> ALSO cols 19..34: the 16 TRUTH amplitude components at (Q2,|t|)
                      (reveals the hidden amplitudes -- YOUR validation only).
    A companion <base>_columns.txt documents the column order."""
    parts = [(LEP_PID, ev["e"], M_LEP), (2212, ev["p"], M),
             (meta["pid_hp"], ev["hp"], meta["MH"]), (meta["pid_hm"], ev["hm"], meta["MH"])]
    n = len(ev["Q2"]); nfiles = int(np.ceil(n / EVENTS_PER_FILE))
    Apar = amps_to_params(ev["Q2"], ev["xB"], ev["absT"]) if LUND_TRUTH else None    # truth amplitudes only if requested
    extra_cols = (LUND_KIN_COLS if LUND_KIN else []) + (LUND_AMP_COLS if LUND_TRUTH else [])
    with open(os.path.join(outdir, f"{base}_columns.txt"), "w") as fc:
        std = ["nparticles", "A", "Z", "target_pol", "beam_helicity", "beam_type", "E_beam",
               "target_pid", "process_id", "weight_wphys"]
        tag = "  [+truth amplitudes]" if LUND_TRUTH else ("  [+kinematics, blind]" if LUND_KIN else "  [clean standard Lund]")
        fc.write(f"Lund header columns (1-indexed){tag}:\n")
        for c, name in enumerate(std + extra_cols, 1):
            fc.write(f"  {c:2d}  {name}\n")
    for fi in range(nfiles):
        lo, hi = fi*EVENTS_PER_FILE, min((fi+1)*EVENTS_PER_FILE, n)
        with open(os.path.join(outdir, f"{base}_{fi}.lund"), "w") as f:
            # Lund field 10: WEIGHTED mode -> physics weight wphys; unweighted (accept-reject) -> 1.0
            wt = ev["wphys"] if (WEIGHTED and "wphys" in ev) else np.ones(n)
            for i in range(lo, hi):
                extra = ""
                if LUND_KIN:
                    extra = (f" {ev['Q2'][i]:.5g} {ev['absT'][i]:.5g} {ev['xB'][i]:.5g} {ev['W'][i]:.5g} "
                             f"{ev['CosTh'][i]:.5g} {ev['phi'][i]:.5g} {ev['Phi'][i]:.5g} {ev['eps'][i]:.5g}")
                if LUND_TRUTH:
                    extra += " " + " ".join(f"{a:.5g}" for a in Apar[i])
                f.write(f"4 1 1 0 {int(ev['hsign'][i])} {LEP_PID} {ev['Ebeam'][i]:.4f} 2212 0 {wt[i]:.6g}{extra}\n")
                for j, (pid, p4, mass) in enumerate(parts, start=1):
                    f.write(f"{j} 0 1 {pid} 0 0 {p4[i,1]:.6f} {p4[i,2]:.6f} {p4[i,3]:.6f} "
                            f"{p4[i,0]:.6f} {mass:.6f} 0 0 0\n")
    mode = "clean standard Lund" if not extra_cols else (f"10 std + {len(extra_cols)} extra"
           + (", incl. TRUTH amplitudes" if LUND_TRUTH else ", blind kinematics"))
    print(f"[lund] wrote {n} events in {nfiles} file(s) of <= {EVENTS_PER_FILE} "
          f"-> {os.path.join(outdir, base)}_[0..{nfiles-1}].lund  ({mode}; see {base}_columns.txt)", flush=True)


# ------------------------------------------------------------------ plots -----
def _labphi(p4):
    p = np.linalg.norm(p4[:, 1:], axis=1)
    th = np.degrees(np.arccos(np.clip(p4[:, 3]/np.clip(p, 1e-9, None), -1, 1)))
    ph = np.degrees(np.arctan2(p4[:, 2], p4[:, 1]))
    return p, th, ph


def plot_particle_kinematics(ev, meta, path):
    rows = [("$e'$", ev["e"]), ("$p'$", ev["p"]), (f"${meta['hp']}$", ev["hp"]), (f"${meta['hm']}$", ev["hm"])]
    cols = ["momentum [GeV]", r"$\theta$ [deg]", r"$\phi$ [deg]"]
    fig, axs = plt.subplots(4, 3, figsize=(13, 13))
    for r, (lab, p4) in enumerate(rows):
        p, th, ph = _labphi(p4)
        for c, (dat, rng_) in enumerate([(p, None), (th, (0, 140)), (ph, (-180, 180))]):
            ax = axs[r, c]
            ax.hist(dat, bins=60, range=rng_, color="#2166ac", alpha=0.85)
            if c == 0: ax.set_ylabel(lab + "   counts")
            if r == 3: ax.set_xlabel(cols[c])
            if r == 0: ax.set_title(cols[c])
    fig.suptitle(f"Particle kinematics ({MESON}, {E_LABEL})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] {path}", flush=True)


def plot_dvep(ev, path):
    PAN = [("Q2", r"$Q^2$ [GeV$^2$]"), ("xB", r"$x_B$"), ("W", r"$W$ [GeV]"), ("nu", r"$\nu$ [GeV]"),
           ("absT", r"$-t$ [GeV$^2$]"), ("tmin", r"$-t_{\min}$ [GeV$^2$]"), ("tprime", r"$t'$ [GeV$^2$]"),
           ("eps", r"$\varepsilon$"), ("CosTh", r"$\cos\theta$"), ("phi", r"$\varphi$ (decay)"),
           ("Phi", r"$\Phi$ (prod.)"),
           ("mV", r"$m_{%s}$ [GeV] (Breit-Wigner)" % MESONS[MESON]["htex"])]
    fig, axs = plt.subplots(3, 4, figsize=(17, 11))
    for ax, (key, lab) in zip(axs.flat, PAN):
        ax.hist(ev[key], bins=60, color="#1b7837", alpha=0.85); ax.set_xlabel(lab); ax.set_ylabel("counts")
    fig.suptitle(f"DVEP kinematics ({MESON}, {E_LABEL})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] {path}", flush=True)


# the 16 real components in the paper's order: |T11|, Re/Im of the rest; U11 real; U00=0
AMP_LABELS = [r"$|T_{11}|$",
              r"$\mathrm{Re}\,T_{00}$", r"$\mathrm{Im}\,T_{00}$",
              r"$\mathrm{Re}\,T_{01}$", r"$\mathrm{Im}\,T_{01}$",
              r"$\mathrm{Re}\,T_{10}$", r"$\mathrm{Im}\,T_{10}$",
              r"$\mathrm{Re}\,T_{1-1}$", r"$\mathrm{Im}\,T_{1-1}$",
              r"$U_{11}$",
              r"$\mathrm{Re}\,U_{01}$", r"$\mathrm{Im}\,U_{01}$",
              r"$\mathrm{Re}\,U_{10}$", r"$\mathrm{Im}\,U_{10}$",
              r"$\mathrm{Re}\,U_{1-1}$", r"$\mathrm{Im}\,U_{1-1}$"]


def plot_amplitudes(meta, path):
    """Paper-style: every real and imaginary amplitude component + sigma_T, sigma_L, R vs |t|."""
    t = np.linspace(0.05, 3.5, 80); Q2 = np.full_like(t, Q2_REF); xB = np.full_like(t, XB_REF)
    A = amps_to_params(Q2, xB, t)                   # (80,16) real components, at Q2_REF, XB_REF
    u = amp_to_u28_batch(A); ud = {nm: u[:, i] for i, nm in enumerate(UNAMES)}
    sT = sigma_T(ud); sL = sigma_L(ud); R = sL/np.clip(sT, 1e-12, None)
    fig, axs = plt.subplots(4, 5, figsize=(19, 13))
    for k in range(16):
        ax = axs.flat[k]
        ax.plot(t, A[:, k], color=("#1f4e79" if k < 9 else "#b03020"), lw=2)   # T blue, U red
        ax.set_ylabel(AMP_LABELS[k]); ax.set_xlabel(r"$|t|$ [GeV$^2$]"); ax.grid(alpha=0.3)
    for ax, lab, y in zip(axs.flat[16:19],
                          [r"$\sigma_T$", r"$\sigma_L$", r"$R=\sigma_L/\sigma_T$"], [sT, sL, R]):
        ax.plot(t, y, color="#1b7837", lw=2); ax.set_ylabel(lab); ax.set_xlabel(r"$|t|$ [GeV$^2$]"); ax.grid(alpha=0.3)
    axs.flat[19].axis("off")
    fig.suptitle(f"Input helicity amplitudes (real and imaginary parts) and cross sections "
                 f"at $Q^2={Q2_REF}$ GeV$^2$ ({MESON})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98]); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] {path}", flush=True)


def plot_observables(ev, meta, path):
    """Physics observables: the beam-spin asymmetry A_LU (helicity-separated) vs the production
    plane Phi and the decay phi, the helicity-split yield N(Phi), and the decay-angle
    distributions W(cos theta, phi, Phi) that carry the SDMEs."""
    h = ev["hsign"]; pol = bool(np.any(h != 0))
    fig, axs = plt.subplots(2, 3, figsize=(17, 9))

    def alu(angle, ax, xlabel, nb=16):
        e = np.linspace(-PI, PI, nb + 1); c = 0.5 * (e[:-1] + e[1:]); A = np.zeros(nb); Er = np.zeros(nb)
        for i in range(nb):
            m = (angle >= e[i]) & (angle < e[i + 1]); Np = int((h[m] > 0).sum()); Nm = int((h[m] < 0).sum())
            Nt = max(Np + Nm, 1); A[i] = (Np - Nm) / Nt; Er[i] = np.sqrt(max(1 - A[i] ** 2, 0) / Nt)
        ax.errorbar(c, A, yerr=Er, fmt="o", ms=4, color="#b03020")
        ax.axhline(0, color="0.6", lw=0.8); ax.set_xlabel(xlabel); ax.set_ylabel(r"$A_{LU}$")
        ax.set_ylim(-0.4, 0.4); ax.grid(alpha=0.3)
        if pol:
            a = 2.0 * np.mean(A * np.sin(c))
            ax.plot(c, a * np.sin(c), "-", color="#1f4e79", lw=1.6, label=r"$%+.3f\,\sin$" % a)
            ax.legend(fontsize=9, loc="upper right")
        else:
            ax.text(0.5, 0.88, "POL=0 (set POL=1 for the beam SSA)", transform=ax.transAxes,
                    ha="center", fontsize=9, color="0.4")

    alu(ev["Phi"], axs[0, 0], r"$\Phi$ (production plane) [rad]"); axs[0, 0].set_title(r"beam SSA  $A_{LU}(\Phi)$")
    alu(ev["phi"], axs[0, 1], r"$\varphi$ (decay) [rad]"); axs[0, 1].set_title(r"$A_{LU}(\varphi_{\rm decay})$")
    ax = axs[0, 2]; b = np.linspace(-PI, PI, 31)
    if pol:
        ax.hist(ev["Phi"][h > 0], bins=b, histtype="step", lw=1.6, color="#2166ac", label=r"$h=+1$")
        ax.hist(ev["Phi"][h < 0], bins=b, histtype="step", lw=1.6, color="#b03020", label=r"$h=-1$")
        ax.legend(fontsize=9)
    else:
        ax.hist(ev["Phi"], bins=b, color="0.7")
    ax.set_title(r"yield $N(\Phi)$ by helicity"); ax.set_xlabel(r"$\Phi$ [rad]"); ax.set_ylabel("counts")

    for ax, (key, lab, rg) in zip(axs[1], [("CosTh", r"$\cos\theta$ (helicity frame)", (-1, 1)),
                                           ("phi", r"$\varphi$ (decay) [rad]", (-PI, PI)),
                                           ("Phi", r"$\Phi$ (production) [rad]", (-PI, PI))]):
        ax.hist(ev[key], bins=40, range=rg, color="#1b7837", alpha=0.85)
        ax.set_xlabel(lab); ax.set_ylabel("counts"); ax.grid(alpha=0.3)
    axs[1, 0].set_ylabel("decay-angle $W(\\Omega)$   counts")

    fig.suptitle(f"Observables ({MESON}, {E_LABEL}, POL={BEAM_POL}) -- beam SSA needs POL=1 "
                 f"and helicity separation", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] {path}", flush=True)


def main():
    if MESON not in MESONS:
        sys.exit(f"MESON must be one of {list(MESONS)}")
    meta = MESONS[MESON]; rng = np.random.default_rng(SEED)
    print(f"[seed] SEED={SEED}"
          + ("  (random; pass SEED={} to reproduce this run)".format(SEED) if not _seed_env else ""),
          flush=True)
    kd = os.path.join(HERE, "Kin_plots"); ld = os.path.join(HERE, "LUND_files")
    os.makedirs(kd, exist_ok=True); os.makedirs(ld, exist_ok=True)
    if MULTI:
        # relative beam luminosities (default equal = equal running time per beam)
        lumis = [float(x) for x in os.environ.get("LUMI", ",".join(["1"] * len(MULTI_ENERGIES))).split(",")]
        assert len(lumis) == len(MULTI_ENERGIES), "LUMI must list one value per beam energy"
        print(f"[multi-energy] REALISTIC yields at {MULTI_ENERGIES} GeV, relative luminosities {lumis}; "
              f"N(E) ~ L(E)*sigma(E), n_events={N_EVENTS} targets the highest-yield beam ...", flush=True)
        evs, counts = generate_multi(MULTI_ENERGIES, meta, rng, lumis, N_EVENTS)
        for E, evE in zip(MULTI_ENERGIES, evs):
            sub = os.path.join(ld, f"{E:g}GeV"); os.makedirs(sub, exist_ok=True)   # e.g. LUND_files/6.535GeV/
            write_lund(evE, meta, sub, f"{MESON}_{E:g}gev")
        with open(os.path.join(ld, "luminosity.txt"), "w") as f:                    # data-analysis normalization
            f.write("# beam_energy_GeV   relative_luminosity   accepted_events\n")
            for E, L, nE in zip(MULTI_ENERGIES, lumis, counts):
                f.write(f"{E:g}  {L:g}  {nE}\n")
        print(f"[multi-energy] accepted per beam: {dict(zip(MULTI_ENERGIES, counts))}; "
              f"wrote {ld}/luminosity.txt", flush=True)
        ev = {k: np.concatenate([e[k] for e in evs]) for k in evs[0]}   # pooled -> combined plots only
    else:
        print(f"generating {N_EVENTS} {MESON} events at E={BEAM_ENERGY} GeV ...", flush=True)
        ev = generate(BEAM_ENERGY, N_EVENTS, meta["MV"], meta["width"], meta["MH"], rng)
        write_lund(ev, meta, ld, f"{MESON}_{BEAM_ENERGY:.1f}gev")
    plot_particle_kinematics(ev, meta, os.path.join(kd, f"particle_kinematics_{MESON}.pdf"))
    plot_dvep(ev, os.path.join(kd, f"dvep_kinematics_{MESON}.pdf"))
    plot_amplitudes(meta, os.path.join(kd, f"amplitudes_{MESON}.pdf"))
    plot_observables(ev, meta, os.path.join(kd, f"observables_{MESON}.pdf"))
    print("done.", flush=True)


if __name__ == "__main__":
    main()
