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
N_EVENTS    = int(os.environ.get("N", "20000"))        # events written to the Lund file
BEAM_POL    = float(os.environ.get("POL", "0.0"))      # beam helicity magnitude (0 = unpolarised)
Q2_REF      = 2.6                                       # Q^2 [GeV^2] for the amplitude-vs-|t| plot
SEED        = 1

MESONS = {  # vector-meson mass, decay-hadron mass, PDG ids, labels
    "phi":  dict(MV=1.019461, MH=0.493677, pid_hp=+321, pid_hm=-321, hp="K+",  hm="K-"),
    "rho0": dict(MV=0.775260, MH=0.139570, pid_hp=+211, pid_hm=-211, hp="pi+", hm="pi-"),
}

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


def throw(E, n_pool, rng, MV, MH):
    """Full DVEP kinematics -> per-event production vars, decay angles, lab 4-vectors of
    e', p', h+, h-, and the physics weight wphys = sigma(Q2,xB,t') * W(Omega; amplitudes)."""
    k = np.array([E, 0, 0, np.sqrt(E**2 - ME**2)])
    Q2 = np.exp(rng.uniform(np.log(1.0), np.log(6.0), n_pool))
    xB = rng.uniform(0.08, 0.5, n_pool); tprime = rng.exponential(0.8, n_pool)
    nu = Q2/(2*M*xB); y = nu/E; W2 = M*M + 2*M*nu - Q2; Ep = E - nu
    cose = 1 - Q2/(2*E*np.clip(Ep, 1e-6, None))
    ok = (y > 0) & (y < 0.99) & (W2 > (M+MV)**2) & (Ep > 0.3) & (np.abs(cose) < 1) & (tprime < 4)
    Q2, xB, tprime, nu, y, W2, Ep, cose = (v[ok] for v in (Q2, xB, tprime, nu, y, W2, Ep, cose))
    Wm = np.sqrt(W2); gam = 2*M*xB/np.sqrt(Q2)
    eps = (1 - y - 0.25*gam**2*y**2)/(1 - y + 0.5*y**2 + 0.25*gam**2*y**2)
    sine = np.sqrt(np.clip(1 - cose**2, 0, None)); phe = rng.uniform(-PI, PI, len(Q2))
    kp = np.stack([Ep, Ep*sine*np.cos(phe), Ep*sine*np.sin(phe), Ep*cose], 1)
    k4 = np.tile(k, (len(Q2), 1)); q = k4 - kp
    ptar = np.tile([M, 0, 0, 0], (len(Q2), 1)); Wsys = q + ptar
    beta = Wsys[:, 1:]/Wsys[:, 0:1]; qcm = _boost(q, -beta)
    pg = np.linalg.norm(qcm[:, 1:], axis=1); Eg = qcm[:, 0]
    Ephi = (W2 + MV**2 - M**2)/(2*Wm); pphi = np.sqrt(np.clip(Ephi**2 - MV**2, 0, None))
    tmin = -Q2 + MV**2 - 2*(Eg*Ephi - pg*pphi); t = tmin - tprime
    cosst = (t + Q2 - MV**2 + 2*Eg*Ephi)/(2*pg*pphi); good = np.abs(cosst) <= 1
    (Q2, xB, nu, y, Wm, eps, kp, q, Wsys, beta, qcm, pg, Eg, Ephi, pphi, tprime, t, tmin, cosst) = (
        v[good] for v in (Q2, xB, nu, y, Wm, eps, kp, q, Wsys, beta, qcm, pg, Eg, Ephi, pphi,
                          tprime, t, tmin, cosst))
    N = len(Q2); sinst = np.sqrt(1 - cosst**2); phipr = rng.uniform(-PI, PI, N)
    zc = qcm[:, 1:]/pg[:, None]; e1, e2 = _perp_axes(zc)
    dirphi = cosst[:, None]*zc + sinst[:, None]*(np.cos(phipr)[:, None]*e1 + np.sin(phipr)[:, None]*e2)
    phicm = np.concatenate([Ephi[:, None], pphi[:, None]*dirphi], 1); phi4 = _boost(phicm, beta)
    prot = Wsys - phi4
    zhel = dirphi; yhel = np.cross(zc, zhel)
    yhel /= np.clip(np.linalg.norm(yhel, axis=1, keepdims=True), 1e-9, None); xhel = np.cross(yhel, zhel)
    pK = np.sqrt(MV**2/4 - MH**2); EK = np.sqrt(pK**2 + MH**2)
    CosTh = rng.uniform(-1, 1, N); Phi = rng.uniform(-PI, PI, N); sinT = np.sqrt(1 - CosTh**2)
    kdir = CosTh[:, None]*zhel + sinT[:, None]*(np.cos(Phi)[:, None]*xhel + np.sin(Phi)[:, None]*yhel)
    Hp_rest = np.concatenate([np.full((N, 1), EK),  pK*kdir], 1)
    Hm_rest = np.concatenate([np.full((N, 1), EK), -pK*kdir], 1)
    bphicm = dirphi*(pphi/Ephi)[:, None]
    Hp = _boost(_boost(Hp_rest, bphicm), beta); Hm = _boost(_boost(Hm_rest, bphicm), beta)
    heli = BEAM_POL*rng.choice([-1.0, 1.0], N)
    u = amp_to_u28_batch(amps_to_params(Q2, -t)); ud = {nm: u[:, i] for i, nm in enumerate(UNAMES)}
    Wp = np.nan_to_num(np.clip(W_ang(CosTh, Phi, phipr, eps, heli, ud), 0, None))
    wphys = np.nan_to_num(cross_section(Q2, xB, tprime)*Wp)
    return dict(Q2=Q2, xB=xB, nu=nu, W=Wm, tprime=tprime, absT=-t, tmin=-tmin, eps=eps,
                CosTh=CosTh, phi=Phi, Phi=phipr, e=kp, p=prot, hp=Hp, hm=Hm, wphys=wphys)


def generate(E, n_target, MV, MH, rng):
    """Accept-reject on wphys -> n_target events."""
    keep = {k: [] for k in ("Q2", "xB", "nu", "W", "tprime", "absT", "tmin", "eps",
                            "CosTh", "phi", "Phi", "e", "p", "hp", "hm")}
    have = 0
    while have < n_target:
        d = throw(E, max(200000, 4*(n_target - have)), rng, MV, MH)
        Wmax = np.percentile(d["wphys"], 99.9)
        acc = rng.uniform(0, Wmax, len(d["wphys"])) < d["wphys"]
        for k in keep: keep[k].append(d[k][acc])
        have += int(acc.sum())
        print(f"  ... {have}/{n_target}", flush=True)
    ev = {k: np.concatenate(v)[:n_target] for k, v in keep.items()}
    return ev


# ------------------------------------------------------------------- Lund -----
def write_lund(ev, meta, path):
    """CLAS12/GEMC Lund: header(10) + one line per final-state particle(14).
    Particle cols: idx lifetime type pid parent daughter px py pz E mass vx vy vz."""
    pid_e, pid_p = 11, 2212
    parts = [(pid_e, ev["e"], ME), (pid_p, ev["p"], M),
             (meta["pid_hp"], ev["hp"], meta["MH"]), (meta["pid_hm"], ev["hm"], meta["MH"])]
    n = len(ev["Q2"])
    with open(path, "w") as f:
        for i in range(n):
            f.write(f"4 1 1 0 {BEAM_POL:.3f} 11 {BEAM_ENERGY:.4f} 2212 0 1.0\n")
            for j, (pid, p4, mass) in enumerate(parts, start=1):
                px, py, pz, En = p4[i, 1], p4[i, 2], p4[i, 3], p4[i, 0]
                f.write(f"{j} 0 1 {pid} 0 0 "
                        f"{px:.6f} {py:.6f} {pz:.6f} {En:.6f} {mass:.6f} 0 0 0\n")
    print(f"[lund] wrote {n} events -> {path}", flush=True)


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
    fig.suptitle(f"Particle kinematics ({MESON}, $E={BEAM_ENERGY}$ GeV)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] {path}", flush=True)


def plot_dvep(ev, path):
    PAN = [("Q2", r"$Q^2$ [GeV$^2$]"), ("xB", r"$x_B$"), ("W", r"$W$ [GeV]"), ("nu", r"$\nu$ [GeV]"),
           ("absT", r"$-t$ [GeV$^2$]"), ("tmin", r"$-t_{\min}$ [GeV$^2$]"), ("tprime", r"$t'$ [GeV$^2$]"),
           ("eps", r"$\varepsilon$"), ("CosTh", r"$\cos\theta$"), ("phi", r"$\varphi$ (decay)"),
           ("Phi", r"$\Phi$ (prod.)")]
    fig, axs = plt.subplots(3, 4, figsize=(17, 11))
    for ax, (key, lab) in zip(axs.flat, PAN):
        ax.hist(ev[key], bins=60, color="#1b7837", alpha=0.85); ax.set_xlabel(lab); ax.set_ylabel("counts")
    axs.flat[-1].axis("off")
    fig.suptitle(f"DVEP kinematics ({MESON}, $E={BEAM_ENERGY}$ GeV)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] {path}", flush=True)


def plot_amplitudes(meta, path):
    t = np.linspace(0.05, 3.5, 60); Q2 = np.full_like(t, Q2_REF)
    T11, T00, T01, T10, T1m1, U11, U01, U10, U1m1 = user_amplitudes(Q2, t)
    A = amps_to_params(Q2, t); u = amp_to_u28_batch(A); ud = {nm: u[:, i] for i, nm in enumerate(UNAMES)}
    sT = sigma_T(ud); sL = sigma_L(ud); R = sL/np.clip(sT, 1e-12, None)
    amps = [("|T_{11}|", T11), ("|T_{00}|", T00), ("|T_{01}|", T01), ("|T_{10}|", T10),
            ("|T_{1-1}|", T1m1), ("|U_{11}|", U11), ("|U_{01}|", U01), ("|U_{10}|", U10), ("|U_{1-1}|", U1m1)]
    fig, axs = plt.subplots(3, 4, figsize=(17, 11))
    for ax, (lab, a) in zip(axs.flat, amps):
        ax.plot(t, np.abs(np.asarray(a) + 0j) + 0*t, color="#b03020", lw=2)
        ax.set_xlabel(r"$|t|$ [GeV$^2$]"); ax.set_ylabel(f"${lab}$"); ax.grid(alpha=0.3)
    for ax, (lab, y) in zip(axs.flat[9:], [(r"\sigma_T", sT), (r"\sigma_L", sL), ("R=\\sigma_L/\\sigma_T", R)]):
        ax.plot(t, y, color="#1f4e79", lw=2); ax.set_xlabel(r"$|t|$ [GeV$^2$]"); ax.set_ylabel(f"${lab}$"); ax.grid(alpha=0.3)
    fig.suptitle(f"Input amplitudes and cross sections at $Q^2={Q2_REF}$ GeV$^2$ ({MESON})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] {path}", flush=True)


def main():
    if MESON not in MESONS:
        sys.exit(f"MESON must be one of {list(MESONS)}")
    meta = MESONS[MESON]; rng = np.random.default_rng(SEED)
    kd = os.path.join(HERE, "Kin_plots"); ld = os.path.join(HERE, "LUND_files")
    os.makedirs(kd, exist_ok=True); os.makedirs(ld, exist_ok=True)
    print(f"generating {N_EVENTS} {MESON} events at E={BEAM_ENERGY} GeV ...", flush=True)
    ev = generate(BEAM_ENERGY, N_EVENTS, meta["MV"], meta["MH"], rng)
    write_lund(ev, meta, os.path.join(ld, f"{MESON}_{BEAM_ENERGY:.1f}gev.lund"))
    plot_particle_kinematics(ev, meta, os.path.join(kd, f"particle_kinematics_{MESON}.pdf"))
    plot_dvep(ev, os.path.join(kd, f"dvep_kinematics_{MESON}.pdf"))
    plot_amplitudes(meta, os.path.join(kd, f"amplitudes_{MESON}.pdf"))
    print("done.", flush=True)


if __name__ == "__main__":
    main()
