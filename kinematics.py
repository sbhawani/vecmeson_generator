#!/usr/bin/env python3
# =============================================================================
# Toy DVMP event-kinematics generator for the extraction paper:
#   e p -> e' p' phi,  phi -> K+ K-   at a given beam energy.
# Produces full LAB 4-vectors (e', p', K+, K-) + DVEP variables (Q2, xB, W, t')
# + eps, with realistic Poisson YIELDS proportional to a smooth cross section.
# A simple lab fiducial (CLAS12-like angular/momentum coverage) gives the
# "accepted" sample for the particle-kinematics plot.
#
# NOTE: this is for the kinematics FIGURES and the realistic yields.  The SDME
# decay-angle analysis uses generate_toy.py / acc_forward.py (decay-angle accept.).
# =============================================================================
import numpy as np

M = 0.93827        # proton mass
MK = 0.49368       # kaon mass
MPHI = 1.01946     # phi mass
ME = 0.000511

BEAM = {"E65": 6.535, "E75": 7.546, "E106": 10.6}


def _mass2(p):                       # p (...,4) = (E,px,py,pz)
    return p[..., 0] ** 2 - np.sum(p[..., 1:] ** 2, -1)


def _boost(p, beta):
    """Boost 4-vectors p (N,4) by velocity beta (N,3) (active: rest-frame -> moving frame)."""
    b2 = np.sum(beta ** 2, 1)
    g = 1.0 / np.sqrt(np.clip(1 - b2, 1e-12, None))
    bp = np.sum(p[:, 1:] * beta, 1)
    E = p[:, 0]
    out = np.empty_like(p)
    out[:, 0] = g * (E + bp)
    fac = np.where(b2 > 0, (g - 1) * bp / np.where(b2 > 0, b2, 1) + g * E, 0.0)
    out[:, 1:] = p[:, 1:] + fac[:, None] * beta
    return out


def _perp_axes(zhat):
    """Two unit vectors perpendicular to zhat (N,3)."""
    a = np.tile(np.array([0.0, 0.0, 1.0]), (len(zhat), 1))
    flip = np.abs(zhat[:, 2]) > 0.9
    a[flip] = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(a, zhat); e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(zhat, e1)
    return e1, e2


def cross_section(Q2, xB, tprime):
    """Smooth toy cross-section weight (sets the realistic yield distribution)."""
    return (1.0 / Q2 ** 2.0) * np.exp(-1.3 * tprime) * np.exp(-((xB - 0.22) / 0.12) ** 2)


def generate(E, n_target, rng, lumi=None):
    """Generate ~n_target accepted (lab-fiducial) events at beam energy E.
    If lumi given, yields are Poisson(lumi * <sigma>); else fixed n_target."""
    k = np.array([E, 0, 0, np.sqrt(E ** 2 - ME ** 2)])
    out = {key: [] for key in ("Q2", "xB", "nu", "W", "tprime", "t", "tmin", "eps",
                               "CosTh", "Phi", "PolPhi",
                               "e_p", "e_th", "e_ph", "p_p", "p_th", "p_ph",
                               "Kp_p", "Kp_th", "Kp_ph", "Km_p", "Km_th", "Km_ph",
                               "gen_Q2", "gen_xB", "gen_tprime")}
    have = 0
    while have < n_target:
        m = int((n_target - have) * 8) + 5000
        Q2 = np.exp(rng.uniform(np.log(1.0), np.log(6.0), m))
        xB = rng.uniform(0.08, 0.5, m)
        tprime = rng.exponential(0.8, m)
        nu = Q2 / (2 * M * xB); y = nu / E
        W2 = M * M + 2 * M * nu - Q2
        Ep = E - nu
        cose = 1 - Q2 / (2 * E * np.clip(Ep, 1e-6, None))
        ok = (y > 0) & (y < 0.99) & (W2 > (M + MPHI) ** 2) & (Ep > 0.3) & (np.abs(cose) < 1) & (tprime < 4)
        out["gen_Q2"].append(Q2[ok]); out["gen_xB"].append(xB[ok]); out["gen_tprime"].append(tprime[ok])
        Q2, xB, tprime, nu, y, W2, Ep, cose = (v[ok] for v in (Q2, xB, tprime, nu, y, W2, Ep, cose))
        N = len(Q2)
        if N == 0:
            continue
        W = np.sqrt(W2)
        gam = 2 * M * xB / np.sqrt(Q2)
        eps = (1 - y - 0.25 * gam ** 2 * y ** 2) / (1 - y + 0.5 * y ** 2 + 0.25 * gam ** 2 * y ** 2)
        # scattered electron (lab)
        sine = np.sqrt(np.clip(1 - cose ** 2, 0, None)); phe = rng.uniform(-np.pi, np.pi, N)
        kp = np.stack([Ep, Ep * sine * np.cos(phe), Ep * sine * np.sin(phe), Ep * cose], 1)
        k4 = np.tile(k, (N, 1))
        q = k4 - kp
        ptar = np.tile([M, 0, 0, 0], (N, 1))
        Wsys = q + ptar
        beta = Wsys[:, 1:] / Wsys[:, 0:1]
        qcm = _boost(q, -beta)
        pg = np.linalg.norm(qcm[:, 1:], axis=1); Eg = qcm[:, 0]
        Ephi = (W ** 2 + MPHI ** 2 - M ** 2) / (2 * W)
        pphi = np.sqrt(np.clip(Ephi ** 2 - MPHI ** 2, 0, None))
        tmin = -Q2 + MPHI ** 2 - 2 * (Eg * Ephi - pg * pphi)
        t = tmin - tprime
        cosst = (t + Q2 - MPHI ** 2 + 2 * Eg * Ephi) / (2 * pg * pphi)
        good = np.abs(cosst) <= 1
        if good.sum() == 0:
            continue
        Q2, xB, nu, y, W, eps, kp, q, Wsys, beta, qcm, pg, Ephi, pphi, tprime, t, tmin, cosst = (
            v[good] for v in (Q2, xB, nu, y, W, eps, kp, q, Wsys, beta, qcm, pg, Ephi, pphi,
                              tprime, t, tmin, cosst))
        N = len(Q2)
        sinst = np.sqrt(1 - cosst ** 2); phipr = rng.uniform(-np.pi, np.pi, N)   # PolPhi (production)
        zc = qcm[:, 1:] / pg[:, None]
        e1, e2 = _perp_axes(zc)
        dirphi = (cosst[:, None] * zc + sinst[:, None] * (np.cos(phipr)[:, None] * e1 + np.sin(phipr)[:, None] * e2))
        phicm = np.concatenate([Ephi[:, None], pphi[:, None] * dirphi], 1)
        phi4 = _boost(phicm, beta)
        prot = Wsys - phi4
        # phi -> K+K- decay in the DIEHL HELICITY frame (z = phi dir in CM,
        # y = production-plane normal); CosTh, Phi are the Diehl decay angles.
        zhel = dirphi
        yhel = np.cross(zc, zhel); yhel /= np.clip(np.linalg.norm(yhel, axis=1, keepdims=True), 1e-9, None)
        xhel = np.cross(yhel, zhel)
        pK = np.sqrt(MPHI ** 2 / 4 - MK ** 2); EK = np.sqrt(pK ** 2 + MK ** 2)
        CosTh = rng.uniform(-1, 1, N); Phi = rng.uniform(-np.pi, np.pi, N)
        sinT = np.sqrt(1 - CosTh ** 2)
        kdir = (CosTh[:, None] * zhel + sinT[:, None] * (np.cos(Phi)[:, None] * xhel + np.sin(Phi)[:, None] * yhel))
        Kp_rest = np.concatenate([np.full((N, 1), EK), pK * kdir], 1)        # CM-axis coords
        Km_rest = np.concatenate([np.full((N, 1), EK), -pK * kdir], 1)
        bphicm = dirphi * (pphi / Ephi)[:, None]                            # phi velocity in CM
        Kp = _boost(_boost(Kp_rest, bphicm), beta)                          # phi-rest -> CM -> lab
        Km = _boost(_boost(Km_rest, bphicm), beta)

        def labvars(p4):
            p = np.linalg.norm(p4[:, 1:], axis=1)
            th = np.degrees(np.arccos(np.clip(p4[:, 3] / np.clip(p, 1e-9, None), -1, 1)))
            ph = np.degrees(np.arctan2(p4[:, 2], p4[:, 1]))
            return p, th, ph
        ep_p, ep_th, ep_ph = labvars(kp)
        pr_p, pr_th, pr_ph = labvars(prot)
        kp_p, kp_th, kp_ph = labvars(Kp)
        km_p, km_th, km_ph = labvars(Km)
        # simple lab fiducial (CLAS12-like): electron FD, hadrons FD+CD
        fid = ((ep_th > 5) & (ep_th < 35) & (ep_p > 1.0) &
               (pr_th > 5) & (pr_th < 125) & (pr_p > 0.2) &
               (kp_th > 5) & (kp_th < 125) & (kp_p > 0.3) &
               (km_th > 5) & (km_th < 125) & (km_p > 0.3))
        # realistic yield weighting (cross-section accept-reject on top of fiducial)
        w = cross_section(Q2, xB, tprime); w /= w.max()
        keep = fid & (rng.uniform(0, 1, N) < w)
        idx = np.where(keep)[0]
        for nm, v in (("Q2", Q2), ("xB", xB), ("nu", nu), ("W", W), ("tprime", tprime),
                      ("t", t), ("tmin", tmin), ("eps", eps),
                      ("CosTh", CosTh), ("Phi", Phi), ("PolPhi", phipr),
                      ("e_p", ep_p), ("e_th", ep_th), ("e_ph", ep_ph),
                      ("p_p", pr_p), ("p_th", pr_th), ("p_ph", pr_ph),
                      ("Kp_p", kp_p), ("Kp_th", kp_th), ("Kp_ph", kp_ph),
                      ("Km_p", km_p), ("Km_th", km_th), ("Km_ph", km_ph)):
            out[nm].append(v[idx])
        have += len(idx)
    res = {k: np.concatenate(v)[:n_target] if k not in ("gen_Q2", "gen_xB", "gen_tprime")
           else np.concatenate(v) for k, v in out.items()}
    return res


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    d = generate(10.6, 30000, rng)
    print("toy DVMP kinematics sanity (10.6 GeV, accepted):")
    for k in ("Q2", "xB", "W", "tprime", "eps", "e_p", "e_th", "p_th", "Kp_p", "Kp_th", "Km_th"):
        v = d[k]; print(f"  {k:7s} mean={v.mean():7.3f}  range=[{v.min():7.3f},{v.max():7.3f}]")
    print(f"  accepted fraction shape: {len(d['Q2'])} events")
