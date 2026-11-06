#!/usr/bin/env python3
# =============================================================================
# S + P wave decay angular distribution at the HELICITY-AMPLITUDE level (2026-11-02).
#
# WHY. diehl_w.W is the J = 1 (vector meson) term of the general gamma* p -> (h+ h-) p
# intensity. The non-resonant h+h- under the phi / rho0 peak is a J = 0 (S-wave) pair; it has
# its own production helicity amplitudes S_{0 mu} (meson helicity 0 only, photon helicity
# mu = +1, 0, -1), an isotropic decay function Y_0^0, and it INTERFERES with the P-wave:
#     M_mu(theta, phi) = sum_nu T_{nu mu} Y_1^nu(theta, phi) + S_{0 mu} Y_0^0
#     I = sum_{mu mu'} rho^gamma_{mu mu'}(eps, Phi, h) [ M^N_mu M^N*_mu' + M^U_mu M^U*_mu' ]
# with the natural (N) and unnatural (U) parity sectors incoherent, as in amplitudes.py
# [Diehl JHEP09(2007)064 eq. (5.3)]. The interference adds the ODD moments (cos theta,
# sin theta cos/sin phi x Phi harmonics) that a pure vector meson cannot produce; |S|^2 adds a
# flat-in-(theta, phi) term with its own Phi dependence (the S-wave's L/T and T/T interference).
#
# CONVENTIONS -- not chosen, MEASURED: with S = 0 this I equals diehl_w.W to 1e-15 for random
# amplitudes, angles, eps and beam helicity (self-test below). That fixed every sign:
#   Y_1^0 = sqrt(3/4pi) cos theta,  Y_1^{+-1} = -+ sqrt(3/8pi) sin theta e^{+-i phi}  (Condon-Shortley)
#   rho_{++} = (1 + h sqrt(1-eps^2))/2,  rho_{--} = (1 - h sqrt(1-eps^2))/2,  rho_{00} = eps
#   rho_{+-} = -(eps/2) e^{-2i Phi}
#   rho_{+0} = e^{-i Phi} [ -1/2 sqrt(eps(1+eps)) - h/2 sqrt(eps(1-eps)) ]
#   rho_{-0} = e^{+i Phi} [ +1/2 sqrt(eps(1+eps)) - h/2 sqrt(eps(1-eps)) ]        (hermitian)
# i.e. Schilling-Wolf's virtual-photon density matrix with the lepton plane at azimuth Phi.
#
# S-WAVE PARAMETERS (6 real, appended to the 16 of amplitudes.py):
#   S00_re, S00_im : natural   S_{0 0}   (longitudinal photon -> S-wave pair)
#   S01_re, S01_im : natural   S_{0 +1}  (transverse photon);   S_{0 -1} = -S_{0 +1}
#   V01_re, V01_im : unnatural S_{0 +1};                        S_{0 -1} = +S_{0 +1},  S_{0 0} = 0
# The parity relations are the SAME as for the vector's N / U sectors (amplitudes.py): the
# Jacob-Wick phase carries eta_meson (-1)^{s_meson}, which is (+1) for both 0^+ and 1^-. The
# phase of S is relative to T11 (real by convention), so all 6 are identifiable in principle;
# with both sectors present the model spans every S_{0 +-1} pair whatever the labelling.
#
# MASS DEPENDENCE is NOT in here: the P-wave carries the Breit-Wigner phase, the S-wave is
# smooth, so across the peak their relative phase rotates by pi. This module is the intensity
# at ONE m(hh) (or integrated over a window with S understood as the window-averaged effective
# amplitude). A mass-dependent fit multiplies T by BW(m)/|BW(m0)| before calling W_sp.
# =============================================================================
import numpy as np

S_NAMES = ["S00_re", "S00_im", "S01_re", "S01_im", "V01_re", "V01_im"]
NS = len(S_NAMES)
_Y10 = np.sqrt(3.0 / (4 * np.pi)); _Y11 = np.sqrt(3.0 / (8 * np.pi)); _Y00 = 1.0 / np.sqrt(4 * np.pi)


def rho_gamma(eps, Phi, heli):
    """Virtual-photon density matrix {(mu, mu'): complex array}, conventions as in the header."""
    eps = np.asarray(eps, float); Phi = np.asarray(Phi, float); h = np.asarray(heli, float)
    e2 = np.sqrt(np.maximum(0.0, 1.0 - eps * eps)); P = np.exp(-1j * Phi)
    a = 0.5 * np.sqrt(eps * (1.0 + eps)); b = 0.5 * np.sqrt(eps * (1.0 - eps))
    r = {(1, 1): 0.5 * (1.0 + h * e2) + 0j, (-1, -1): 0.5 * (1.0 - h * e2) + 0j, (0, 0): eps + 0j,
         (1, -1): -0.5 * eps * P * P, (1, 0): P * (-a - h * b), (-1, 0): np.conj(P) * (a - h * b)}
    r[(-1, 1)] = np.conj(r[(1, -1)]); r[(0, 1)] = np.conj(r[(1, 0)]); r[(0, -1)] = np.conj(r[(-1, 0)])
    return r


def y1(c, phi):
    """{nu: Y_1^nu(theta, phi)} for c = cos theta."""
    st = np.sqrt(np.maximum(0.0, 1.0 - c * c)); e = np.exp(1j * phi)
    return {0: _Y10 * c + 0j, 1: -_Y11 * st * e, -1: _Y11 * st * np.conj(e)}


def vector_NU_batch(A):
    """(n, 16) vector-meson params (amplitudes.py order) -> (N, U) maps {(nu, mu): complex} under
    the parity relations N_{-nu,-mu} = (-1)^{nu-mu} N_{nu,mu}, U_{-nu,-mu} = -(-1)^{nu-mu} U_{nu,mu},
    U_{00} = 0. Same as amplitudes.build_NU_batch, kept here so this module has no dependency on
    which `amplitudes` module is first on sys.path (the extraction tree ships its own)."""
    A = np.atleast_2d(np.asarray(A, float))
    T11 = A[:, 0] + 0j; T00 = A[:, 1] + 1j * A[:, 2]; T01 = A[:, 3] + 1j * A[:, 4]
    T10 = A[:, 5] + 1j * A[:, 6]; T1m1 = A[:, 7] + 1j * A[:, 8]
    U11 = A[:, 9] + 0j; U01 = A[:, 10] + 1j * A[:, 11]; U10 = A[:, 12] + 1j * A[:, 13]; U1m1 = A[:, 14] + 1j * A[:, 15]
    z = np.zeros_like(T11)
    N = {(1, 1): T11, (0, 0): T00, (0, 1): T01, (1, 0): T10, (1, -1): T1m1,
         (-1, -1): T11, (0, -1): -T01, (-1, 0): -T10, (-1, 1): T1m1}
    U = {(1, 1): U11, (0, 0): z, (0, 1): U01, (1, 0): U10, (1, -1): U1m1,
         (-1, -1): -U11, (0, -1): U01, (-1, 0): U10, (-1, 1): -U1m1}
    return N, U


def swave_NU_batch(S):
    """(n, 6) S-wave params -> (N, U) maps {(0, mu): complex} under the parity relations."""
    S = np.atleast_2d(np.asarray(S, float))
    S00 = S[:, 0] + 1j * S[:, 1]; S01 = S[:, 2] + 1j * S[:, 3]; V01 = S[:, 4] + 1j * S[:, 5]
    z = np.zeros_like(S00)
    return ({(0, 0): S00, (0, 1): S01, (0, -1): -S01}, {(0, 0): z, (0, 1): V01, (0, -1): V01})


def W_sp(c, phi, Phi, eps, heli, A16, S6=None):
    """S + P intensity per event. A16 (n, 16) vector-meson params (amplitudes.py order), S6 (n, 6)
    S-wave params or None (then identical to diehl_w.W). c = cos theta, phi = decay azimuth,
    Phi = production-plane azimuth, heli = beam helicity (signed polarization)."""
    Nm, Um = vector_NU_batch(A16)
    Y = y1(np.asarray(c, float), np.asarray(phi, float))
    sects = [(Nm, None), (Um, None)]
    if S6 is not None:
        SN, SU = swave_NU_batch(S6); sects = [(Nm, SN), (Um, SU)]
    r = rho_gamma(eps, Phi, heli)
    I = 0.0
    for X, Sx in sects:
        M = {}
        for mu in (1, 0, -1):
            m = sum(X[(nu, mu)] * Y[nu] for nu in (1, 0, -1))
            if Sx is not None:
                m = m + Sx[(0, mu)] * _Y00
            M[mu] = m
        for mu in (1, 0, -1):
            for mup in (1, 0, -1):
                I = I + (r[(mu, mup)] * M[mu] * np.conj(M[mup])).real
    return I


# ---------------------------------------------------------------- linear (design) form -----
# I is a real quadratic form in the 12 channel amplitudes X_p, p = (a, mu) with a in
# {+1, 0, -1, S} (meson helicity / S-wave) and mu in {+1, 0, -1} (photon helicity):
#     I = sum_p |X_p|^2 H_pp + 2 sum_{p<p'} Re(H_{pp'} X_p X*_p'),   H_{pp'} = rho_{mu mu'} Y_a Y*_b
# so I = F(Omega) . G(params) with F the 144 real angular functions below and G the bilinears,
# summed over the incoherent N and U sectors. This is what a design-matrix UML needs.
CHANNELS = [(a, mu) for a in (1, 0, -1, "S") for mu in (1, 0, -1)]
_PAIRS = [(i, j) for i in range(len(CHANNELS)) for j in range(i + 1, len(CHANNELS))]
NDESIGN = len(CHANNELS) + 2 * len(_PAIRS)      # 12 + 2 * 66 = 144


def design_sp(c, phi, Phi, eps, heli):
    """(n, 144) angular design matrix F with W_sp = F @ coef_sp(A16, S6)."""
    c = np.asarray(c, float); Y = y1(c, np.asarray(phi, float)); Y["S"] = np.full(c.shape, _Y00, complex)
    r = rho_gamma(eps, Phi, heli)
    H = lambda p, q: r[(CHANNELS[p][1], CHANNELS[q][1])] * Y[CHANNELS[p][0]] * np.conj(Y[CHANNELS[q][0]])
    cols = [H(p, p).real for p in range(len(CHANNELS))]
    for i, j in _PAIRS:
        h = H(i, j); cols += [2.0 * h.real, -2.0 * h.imag]
    return np.stack(cols, 1)


def coef_sp(A16, S6=None):
    """(144,) bilinear coefficient vector G for ONE parameter set (16 vector + 6 S-wave reals)."""
    A16 = np.atleast_2d(np.asarray(A16, float)); Nm, Um = vector_NU_batch(A16)
    if S6 is not None and np.size(S6) == 0:
        S6 = None
    SN, SU = swave_NU_batch(np.zeros((A16.shape[0], NS)) if S6 is None else S6)
    G = 0.0
    for X, Sx in ((Nm, SN), (Um, SU)):
        Xp = [X[(a, mu)] if a != "S" else Sx[(0, mu)] for a, mu in CHANNELS]
        g = [np.abs(x) ** 2 for x in Xp]
        for i, j in _PAIRS:
            z = Xp[i] * np.conj(Xp[j]); g += [z.real, z.imag]
        G = G + np.stack(g, -1)
    return G[0] if G.shape[0] == 1 else G


def odd_moments(c, phi, Phi):
    """The leading S-P interference observables of a sample: <cos theta>, <sin theta cos phi>,
    <sin theta sin phi>, and their cos Phi / sin Phi projections. All vanish for a pure vector."""
    st = np.sqrt(np.maximum(0.0, 1.0 - c * c))
    f = {"cos": c, "sin_cphi": st * np.cos(phi), "sin_sphi": st * np.sin(phi)}
    out = {}
    for k, v in f.items():
        out[k] = float(np.mean(v)); out[k + "_cPhi"] = float(np.mean(v * np.cos(Phi))); out[k + "_sPhi"] = float(np.mean(v * np.sin(Phi)))
    return out


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from diehl_w import W, UNAMES
    from amplitudes import amp_to_u28_batch, NAMP
    rng = np.random.default_rng(11); n = 2000
    A = rng.uniform(-1, 1, (n, NAMP)); A[:, 0] = np.abs(A[:, 0])
    c = rng.uniform(-1, 1, n); ph = rng.uniform(-np.pi, np.pi, n); Ph = rng.uniform(-np.pi, np.pi, n)
    eps = rng.uniform(0.1, 0.99, n); h = rng.choice([-0.85, 0.85, 0.0], n)
    u = amp_to_u28_batch(A); ud = {nm: u[:, i] for i, nm in enumerate(UNAMES)}
    d = np.max(np.abs(W_sp(c, ph, Ph, eps, h, A) - W(c, ph, Ph, eps, h, ud)))
    print(f"(1) S = 0 reproduces diehl_w.W: max |dI| = {d:.2e}"); assert d < 1e-12
    S = rng.uniform(-1, 1, (n, NS))
    I = W_sp(c, ph, Ph, eps, h, A, S)
    print(f"(2) positivity with random S-wave: min I = {I.min():.3e}"); assert I.min() >= -1e-12
    # (3) S-only: flat in the decay angles at fixed (eps, Phi, h)
    A0 = np.zeros((n, NAMP)); A0[:, 0] = 1e-300
    S1 = np.tile(np.array([[0.4, 0.1, 0.3, -0.2, 0.1, 0.05]]), (n, 1))
    I0 = W_sp(c, ph, np.full(n, 0.3), np.full(n, 0.6), np.full(n, 0.85), A0, S1)
    print(f"(3) S-only intensity flat in (theta, phi): spread {np.std(I0)/np.mean(I0):.2e}"); assert np.std(I0) / np.mean(I0) < 1e-12
    # (4) odd moments: zero for P only, non-zero with S, by weighted means over a flat sample
    nn = 400000; cc = rng.uniform(-1, 1, nn); pp = rng.uniform(-np.pi, np.pi, nn); PP = rng.uniform(-np.pi, np.pi, nn)
    Ap = np.tile(np.array([[0.9, 1.3, 0.2, 0.1, 0.05, 0.15, -0.1, 0.05, 0.02, 0, 0, 0, 0, 0, 0, 0]]), (nn, 1))
    Sp = np.tile(np.array([[0.5, 0.3, 0.2, 0.1, 0.0, 0.0]]), (nn, 1))
    wP = W_sp(cc, pp, PP, np.full(nn, 0.7), np.full(nn, 0.0), Ap)
    wS = W_sp(cc, pp, PP, np.full(nn, 0.7), np.full(nn, 0.0), Ap, Sp)
    mP = np.sum(wP * cc) / np.sum(wP); mS = np.sum(wS * cc) / np.sum(wS)
    print(f"(4) <cos theta>: P only {mP:+.4f}, S + P {mS:+.4f}; S-wave fraction of the rate {1 - np.sum(wP)/np.sum(wS):.3f}")
    assert abs(mP) < 5e-3 and abs(mS) > 0.05
    # (5) design form: F @ G == W_sp for S = 0 and S != 0
    F = design_sp(c, ph, Ph, eps, h)
    d5 = max(np.max(np.abs(np.einsum("ik,ik->i", F, coef_sp(A, S)) - W_sp(c, ph, Ph, eps, h, A, S))),
             np.max(np.abs(np.einsum("ik,ik->i", F, coef_sp(A)) - W_sp(c, ph, Ph, eps, h, A))))
    print(f"(5) design form F . G == W_sp (per-event params): max |d| = {d5:.2e}"); assert d5 < 1e-12
    print("done.")
