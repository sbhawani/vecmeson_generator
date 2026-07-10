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
#   python generate_events.py                 # phi, defaults
#   MESON=rho0 N=50000 python generate_events.py
# =============================================================================
import os, sys
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
N_EVENTS    = int(os.environ.get("N", "20000"))        # total events generated
EVENTS_PER_FILE = int(os.environ.get("CHUNK", "5000")) # events per Lund file (GEMC limit); N/CHUNK files
BEAM_POL    = float(os.environ.get("POL", "0.0"))      # beam helicity magnitude (0 = unpolarised)
Q2_REF      = 2.6                                       # Q^2 [GeV^2] for the amplitude-vs-|t| plot
SEED        = 1
MULTI_ENERGIES = [6.535, 7.546, 10.6]                  # beams used in multi-energy mode
MULTI       = ("--multi-energy" in sys.argv) or (os.environ.get("MULTI", "0") not in ("0", "", "false", "False"))
E_LABEL     = ("multi-energy " + "/".join(f"{e:g}" for e in MULTI_ENERGIES) + " GeV") if MULTI else f"$E={BEAM_ENERGY}$ GeV"

MESONS = {  # vector-meson pole mass, width, decay-hadron mass, PDG ids, labels
    "phi":  dict(MV=1.019461, width=0.004249, MH=0.493677, pid_hp=+321, pid_hm=-321, hp="K+",  hm="K-",  htex=r"K^+K^-"),
    "rho0": dict(MV=0.775260, width=0.149100, MH=0.139570, pid_hp=+211, pid_hm=-211, hp="pi+", hm="pi-", htex=r"\pi^+\pi^-"),
}
BW_MASS = os.environ.get("BW", "1") not in ("0", "", "false", "False")   # sample the Breit-Wigner line shape
# How the Q^2/x/t event yield is weighted (kinematics are sampled FLAT, so this alone sets the shape):
#   amp  : weight = |amplitude|^2 only -> the Q^2/t dependence comes PURELY from your amplitudes (default)
#   flux : weight = virtual-photon flux x |amplitude|^2  (adds the physical Gamma(Q^2,x,y))
#   toy  : weight = smooth toy cross section x |amplitude|^2  (legacy paper-study shape)
WEIGHT = os.environ.get("WEIGHT", "amp")
TMAX   = 4.0                                                              # flat t' sampling range [GeV^2]

# --- Helicity amplitudes as functions of (Q^2, |t|).  EDIT THESE. ------------
# Conventions (unpolarised nucleon-helicity non-flip):
#   T_{-mu,-nu} = +(-1)^{mu-nu} T_{mu nu}   (natural parity)
#   U_{-mu,-nu} = -(-1)^{mu-nu} U_{mu nu}   (unnatural parity),  U_{00} = 0
#   T11 and U11 are taken REAL (global-phase / residual-phase reference).
#   Everything else may be complex.  Return 9 amplitudes (T then U).
def user_amplitudes(Q2, t):
    Q = np.sqrt(Q2)
    T11  = (1.00 / Q)         * np.exp(-0.65 * t)                     # real, transverse (dominant)
    T00  = (1.73)             * np.exp(-0.55 * t) + 0j                # longitudinal
    T01  = (0.45 / Q)         * np.exp(-0.60 * t) * np.exp(+1j * 0.7) # single-flip (SCHC-violating)
    T10  = (0.15 / Q)         * np.exp(-0.60 * t) + 0j                # single-flip
    T1m1 = (0.10)             * np.exp(-0.60 * t) * np.exp(-1j * 0.4) # double-flip
    U11  = 0.0                                                        # real; unnatural parity
    U01  = 0.0 + 0j
    U10  = 0.0 + 0j
    U1m1 = 0.0 + 0j
    return T11, T00, T01, T10, T1m1, U11, U01, U10, U1m1
# =============================================================================


def amps_to_params(Q2, t):
    """User amplitudes -> the 16-real-parameter vector (T11,U11 real; U00=0)."""
    T11, T00, T01, T10, T1m1, U11, U01, U10, U1m1 = user_amplitudes(Q2, t)
    z = np.zeros(np.broadcast(np.asarray(Q2), np.asarray(t)).shape, float)
    re = lambda x: np.real(x) + z; im = lambda x: np.imag(x) + z
    return np.stack([re(T11), re(T00), im(T00), re(T01), im(T01), re(T10), im(T10),
                     re(T1m1), im(T1m1), re(U11), re(U01), im(U01),
                     re(U10), im(U10), re(U1m1), im(U1m1)], axis=-1)


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
    k = np.array([E, 0, 0, np.sqrt(E**2 - ME**2)])
    mv = sample_meson_mass(rng, n_pool, MV, GV, MH)              # per-event invariant mass
    Q2 = rng.uniform(1.0, 6.0, n_pool)                          # FLAT: shape set by the weight below
    xB = rng.uniform(0.08, 0.5, n_pool); tprime = rng.uniform(0.0, TMAX, n_pool)   # FLAT in t'
    nu = Q2/(2*M*xB); y = nu/E; W2 = M*M + 2*M*nu - Q2; Ep = E - nu
    cose = 1 - Q2/(2*E*np.clip(Ep, 1e-6, None))
    ok = (y > 0) & (y < 0.99) & (W2 > (M+mv)**2) & (Ep > 0.3) & (np.abs(cose) < 1) & (tprime < 4)
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
    zc = qcm[:, 1:]/pg[:, None]; e1, e2 = _perp_axes(zc)
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
    u = amp_to_u28_batch(amps_to_params(Q2, -t)); ud = {nm: u[:, i] for i, nm in enumerate(UNAMES)}
    Wp = np.nan_to_num(np.clip(W_ang(CosTh, Phi, phipr, eps, heli, ud), 0, None))   # |amplitude|^2 x W_SW(Omega)
    if WEIGHT == "toy":
        wt = cross_section(Q2, xB, tprime)                      # legacy smooth toy shape
    elif WEIGHT == "flux":
        Kf = (Wm**2 - M**2)/(2*M)                               # Hand equivalent photon energy
        wt = (1.0 - y)*Kf/(Q2*np.clip(1.0 - eps, 1e-3, None))   # transverse virtual-photon flux (relative)
    else:                                                       # "amp": Q^2/t dependence from the amplitudes only
        wt = 1.0
    wphys = np.nan_to_num(wt*Wp)
    return dict(Q2=Q2, xB=xB, nu=nu, W=Wm, tprime=tprime, absT=-t, tmin=-tmin, eps=eps,
                CosTh=CosTh, phi=Phi, Phi=phipr, e=kp, p=prot, hp=Hp, hm=Hm, wphys=wphys,
                Ebeam=np.full(len(Q2), E), hsign=hsign, mV=mv)


def generate(E, n_target, MV, GV, MH, rng):
    """Accept-reject on wphys -> n_target events."""
    keep = {k: [] for k in ("Q2", "xB", "nu", "W", "tprime", "absT", "tmin", "eps",
                            "CosTh", "phi", "Phi", "e", "p", "hp", "hm", "Ebeam", "hsign", "mV")}
    have = 0
    while have < n_target:
        d = throw(E, max(200000, 4*(n_target - have)), rng, MV, GV, MH)
        Wmax = np.percentile(d["wphys"], 99.9)
        acc = rng.uniform(0, Wmax, len(d["wphys"])) < d["wphys"]
        for k in keep: keep[k].append(d[k][acc])
        have += int(acc.sum())
        print(f"  ... {have}/{n_target}", flush=True)
    ev = {k: np.concatenate(v)[:n_target] for k, v in keep.items()}
    return ev


# ------------------------------------------------------------------- Lund -----
def write_lund(ev, meta, outdir, base):
    """CLAS12/GEMC Lund, split into files of at most EVENTS_PER_FILE events (GEMC limit),
    named <base>_0.lund, <base>_1.lund, ...  header(10) + one line per final-state particle(14):
    idx lifetime type pid parent daughter px py pz E mass vx vy vz."""
    parts = [(11, ev["e"], ME), (2212, ev["p"], M),
             (meta["pid_hp"], ev["hp"], meta["MH"]), (meta["pid_hm"], ev["hm"], meta["MH"])]
    n = len(ev["Q2"]); nfiles = int(np.ceil(n / EVENTS_PER_FILE))
    for fi in range(nfiles):
        lo, hi = fi*EVENTS_PER_FILE, min((fi+1)*EVENTS_PER_FILE, n)
        with open(os.path.join(outdir, f"{base}_{fi}.lund"), "w") as f:
            for i in range(lo, hi):
                f.write(f"4 1 1 0 {int(ev['hsign'][i])} 11 {ev['Ebeam'][i]:.4f} 2212 0 1.0\n")
                for j, (pid, p4, mass) in enumerate(parts, start=1):
                    f.write(f"{j} 0 1 {pid} 0 0 {p4[i,1]:.6f} {p4[i,2]:.6f} {p4[i,3]:.6f} "
                            f"{p4[i,0]:.6f} {mass:.6f} 0 0 0\n")
    print(f"[lund] wrote {n} events in {nfiles} file(s) of <= {EVENTS_PER_FILE} "
          f"-> {os.path.join(outdir, base)}_[0..{nfiles-1}].lund", flush=True)


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
    t = np.linspace(0.05, 3.5, 80); Q2 = np.full_like(t, Q2_REF)
    A = amps_to_params(Q2, t)                       # (80,16) real components
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
    kd = os.path.join(HERE, "Kin_plots"); ld = os.path.join(HERE, "LUND_files")
    os.makedirs(kd, exist_ok=True); os.makedirs(ld, exist_ok=True)
    if MULTI:
        print(f"[multi-energy] {N_EVENTS} {MESON} events at each of {MULTI_ENERGIES} GeV; "
              f"Lund files per energy in their own subdir ...", flush=True)
        evs = []
        for E in MULTI_ENERGIES:
            print(f"  E={E} GeV:", flush=True)
            evE = generate(E, N_EVENTS, meta["MV"], meta["width"], meta["MH"], rng)
            sub = os.path.join(ld, f"{E:g}GeV"); os.makedirs(sub, exist_ok=True)   # e.g. LUND_files/6.535GeV/
            write_lund(evE, meta, sub, f"{MESON}_{E:g}gev")
            evs.append(evE)
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
