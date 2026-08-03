#!/usr/bin/env python3
# =============================================================================
# PHASE 2 -- FULL polarized angular distribution, Diehl (4.5)/(4.8)/(4.9):
#
#   W = W_UU + Pb W_LU + S_L [W_UL + Pb W_LL] + S_T [W_UT + Pb W_LT]
#
# built from FOUR verbatim tables (Diehl 4.10 / 4.12 / 4.13 / 4.14) plus the exact
# substitution structure of eq (4.2):
#     sum_tau rho = u  +  S_L l  +  S_T cos(Phi-phiS) s  -  S_T sin(Phi-phiS) i n
# so   W_UT = sin(Phi-phiS) UU[i n] + cos(Phi-phiS) UL[s]
#      W_LT = sin(Phi-phiS) LU[i n] + cos(Phi-phiS) LL[s]
# with the u-pattern slots reading  Re(i n) = -Im n -> re-slot <- +Im n after the
# overall minus of (4.2), i.e. re-slot <- Im n, im-slot <- -Re n   [= rule (4.16)].
# s substitutes l literally (same symmetry class, Diehl p.13).
# Each sub-block carries its theta weight from (4.8)/(4.9):
#     LL: cos^2(th)    LT: sqrt(2) cos(th) sin(th)    TT: sin^2(th)
# Angle map: Diehl phi = production-plane azimuth = our 'polphi' (Phi here);
#            Diehl varphi = decay azimuth = our 'phi' (v here).
# =============================================================================
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from amplitudes import FIELD_RECIPE
from diehl_w import UNAMES
from amplitudes_full import _blocks

# ---- complex position-fields (l/s/n have Im parts where u's symmetry forced real) ----
def _basename(n):
    return n[3:] if n.startswith(("Re_", "Im_")) else n

_POS = []
_seen = set()
for _j, _n in enumerate(UNAMES):
    _b = _basename(_n)
    if _b not in _seen:
        _seen.add(_b); _POS.append((_b, FIELD_RECIPE[_j][1]))

def complex_fields(X, Y):
    """basename -> complex scalar/array, X Y'* contracted per the recipe sigma-sums."""
    out = {}
    for b, terms in _POS:
        v = 0j
        for s, mu, nu, mp, np_ in terms:
            v = v + s * (X[(mu, nu)] * np.conj(Y[(mp, np_)]))
        out[b] = v
    return out


def ulsn_fields(A):
    """A (34,) -> complex field dicts u, l, s, n (Diehl 5.1-5.3 structure functions)."""
    T0, U0, T1, U1 = _blocks(np.atleast_2d(np.asarray(A, dtype=float)))
    cf = complex_fields
    def lin(*pairs):
        ds = [cf(X, Y) for _, X, Y in pairs]
        return {k: sum(sg * d[k] for (sg, _, _), d in zip(pairs, ds)) for k in ds[0]}
    u = lin((1, T0, T0), (1, U0, U0), (1, T1, T1), (1, U1, U1))
    l = lin((1, T0, U0), (1, U0, T0), (1, T1, U1), (1, U1, T1))
    s = lin((1, T0, U1), (-1, U0, T1), (-1, T1, U0), (1, U1, T0))
    n = lin((-1, T0, T1), (1, U0, U1), (1, T1, T0), (-1, U1, U0))
    return u, l, s, n


# ---- the four verbatim tables: (block, coef(eps), harm(Phi, v), base, re|im) ---------
_EPP = lambda e: np.sqrt(e * (1 + e)); _EPM = lambda e: np.sqrt(e * (1 - e))
_E2 = lambda e: np.sqrt(1 - e * e); _C = lambda P, v: 1.0 + 0.0 * P

W_UU_TABLE = [   # Diehl (4.10), unpolarized beam & target
    ("LL", lambda e: 1.0,       _C,                            "u00_pp", "re"),
    ("LL", lambda e: e,         _C,                            "u00_00", "re"),
    ("LL", lambda e: -2*_EPP(e), lambda P, v: np.cos(P),       "u00_0p", "re"),
    ("LL", lambda e: -e,        lambda P, v: np.cos(2*P),      "u00_mp", "re"),
    ("LT", lambda e: _EPP(e),   lambda P, v: np.cos(P + v),    "u0p_0p_minus_um0_0p", "re"),
    ("LT", lambda e: -1.0,      lambda P, v: np.cos(v),        "u0p_pp_minus_um0_pp", "re"),
    ("LT", lambda e: -2*e,      lambda P, v: np.cos(v),        "u0p_00", "re"),
    ("LT", lambda e: e,         lambda P, v: np.cos(2*P + v),  "u0p_mp", "re"),
    ("LT", lambda e: -_EPP(e),  lambda P, v: np.cos(P - v),    "u0m_0p_minus_up0_0p", "re"),
    ("LT", lambda e: e,         lambda P, v: np.cos(2*P - v),  "up0_mp", "re"),
    ("TT", lambda e: 0.5,       _C,                            "u11_pp", "re"),
    ("TT", lambda e: 0.5,       _C,                            "umm_pp", "re"),
    ("TT", lambda e: e,         _C,                            "u11_00", "re"),
    ("TT", lambda e: 0.5*e,     lambda P, v: np.cos(2*P + 2*v), "ump_mp", "re"),
    ("TT", lambda e: -_EPP(e),  lambda P, v: np.cos(P),        "upp_0p_plus_umm_0p", "re"),
    ("TT", lambda e: _EPP(e),   lambda P, v: np.cos(P + 2*v),  "ump_0p", "re"),
    ("TT", lambda e: -1.0,      lambda P, v: np.cos(2*v),      "ump_pp", "re"),
    ("TT", lambda e: -e,        lambda P, v: np.cos(2*v),      "ump_00", "re"),
    ("TT", lambda e: -e,        lambda P, v: np.cos(2*P),      "upp_mp", "re"),
    ("TT", lambda e: _EPP(e),   lambda P, v: np.cos(P - 2*v),  "upm_0p", "re"),
    ("TT", lambda e: 0.5*e,     lambda P, v: np.cos(2*P - 2*v), "upm_mp", "re"),
]

W_LU_TABLE = [   # Diehl (4.12), polarized beam, unpolarized target
    ("LL", lambda e: -2*_EPM(e), lambda P, v: np.sin(P),       "u00_0p", "im"),
    ("LT", lambda e: _EPM(e),   lambda P, v: np.sin(P + v),    "u0p_0p_minus_um0_0p", "im"),
    ("LT", lambda e: -_E2(e),   lambda P, v: np.sin(v),        "u0p_pp_minus_um0_pp", "im"),
    ("LT", lambda e: -_EPM(e),  lambda P, v: np.sin(P - v),    "u0m_0p_minus_up0_0p", "im"),
    ("TT", lambda e: -_EPM(e),  lambda P, v: np.sin(P),        "upp_0p_plus_umm_0p", "im"),
    ("TT", lambda e: _EPM(e),   lambda P, v: np.sin(P + 2*v),  "ump_0p", "im"),
    ("TT", lambda e: -_E2(e),   lambda P, v: np.sin(2*v),      "ump_pp", "im"),
    ("TT", lambda e: _EPM(e),   lambda P, v: np.sin(P - 2*v),  "upm_0p", "im"),
]

W_UL_TABLE = [   # Diehl (4.13), longitudinal target, unpolarized beam
    ("LL", lambda e: -2*_EPP(e), lambda P, v: np.sin(P),       "u00_0p", "im"),
    ("LL", lambda e: -e,        lambda P, v: np.sin(2*P),      "u00_mp", "im"),
    ("LT", lambda e: _EPP(e),   lambda P, v: np.sin(P + v),    "u0p_0p_minus_um0_0p", "im"),
    ("LT", lambda e: -1.0,      lambda P, v: np.sin(v),        "u0p_pp_minus_um0_pp", "im"),
    ("LT", lambda e: -2*e,      lambda P, v: np.sin(v),        "u0p_00", "im"),
    ("LT", lambda e: e,         lambda P, v: np.sin(2*P + v),  "u0p_mp", "im"),
    ("LT", lambda e: -_EPP(e),  lambda P, v: np.sin(P - v),    "u0m_0p_minus_up0_0p", "im"),
    ("LT", lambda e: e,         lambda P, v: np.sin(2*P - v),  "up0_mp", "im"),
    ("TT", lambda e: 0.5*e,     lambda P, v: np.sin(2*P + 2*v), "ump_mp", "im"),
    ("TT", lambda e: -_EPP(e),  lambda P, v: np.sin(P),        "upp_0p_plus_umm_0p", "im"),
    ("TT", lambda e: _EPP(e),   lambda P, v: np.sin(P + 2*v),  "ump_0p", "im"),
    ("TT", lambda e: -1.0,      lambda P, v: np.sin(2*v),      "ump_pp", "im"),
    ("TT", lambda e: -e,        lambda P, v: np.sin(2*v),      "ump_00", "im"),
    ("TT", lambda e: -e,        lambda P, v: np.sin(2*P),      "upp_mp", "im"),
    ("TT", lambda e: _EPP(e),   lambda P, v: np.sin(P - 2*v),  "upm_0p", "im"),
    ("TT", lambda e: 0.5*e,     lambda P, v: np.sin(2*P - 2*v), "upm_mp", "im"),
]

W_LL_TABLE = [   # Diehl (4.14), polarized beam AND longitudinal target
    ("LL", lambda e: -2*_EPM(e), lambda P, v: np.cos(P),       "u00_0p", "re"),
    ("LL", lambda e: _E2(e),    _C,                            "u00_pp", "re"),
    ("LT", lambda e: _EPM(e),   lambda P, v: np.cos(P + v),    "u0p_0p_minus_um0_0p", "re"),
    ("LT", lambda e: -_E2(e),   lambda P, v: np.cos(v),        "u0p_pp_minus_um0_pp", "re"),
    ("LT", lambda e: -_EPM(e),  lambda P, v: np.cos(P - v),    "u0m_0p_minus_up0_0p", "re"),
    ("TT", lambda e: 0.5*_E2(e), _C,                           "u11_pp", "re"),
    ("TT", lambda e: 0.5*_E2(e), _C,                           "umm_pp", "re"),
    ("TT", lambda e: -_EPM(e),  lambda P, v: np.cos(P),        "upp_0p_plus_umm_0p", "re"),
    ("TT", lambda e: _EPM(e),   lambda P, v: np.cos(P + 2*v),  "ump_0p", "re"),
    ("TT", lambda e: -_E2(e),   lambda P, v: np.cos(2*v),      "ump_pp", "re"),
    ("TT", lambda e: _EPM(e),   lambda P, v: np.cos(P - 2*v),  "upm_0p", "re"),
]

_TH = {"LL": lambda c: c * c,
       "LT": lambda c: np.sqrt(2.0) * c * np.sqrt(np.clip(1 - c * c, 0.0, None)),
       "TT": lambda c: 1 - c * c}


def _slot(F, base, sel, kind):
    v = F[base]
    if kind == "n":                      # u-pattern slot reading  -i n  (eq 4.2 / 4.16)
        return v.imag if sel == "re" else -v.real
    return v.real if sel == "re" else v.imag


def eval_table(table, F, kind, c, v, Phi, eps):
    tot = 0.0
    for blk, coef, harm, base, sel in table:
        tot = tot + _TH[blk](c) * coef(eps) * harm(Phi, v) * _slot(F, base, sel, kind)
    return (3.0 / (4 * np.pi)) * tot


def W_full(c, phi, polphi, eps, heli, A, S_L=0.0, S_T=0.0, phiS=0.0):
    """Full Diehl (4.5) angular weight for one 34-parameter amplitude row A, in
    PRODUCTION angle/helicity conventions (same signature order as diehl_w.W):
    c=cos(theta), phi=decay azimuth, polphi=production-plane azimuth, heli=beam.
    Measured convention map to the Diehl tables (per-slot audit vs diehl_w.W):
        varphi_Diehl = -phi_production ,   Pb_Diehl = -heli_production
    (a pure convention pair -- SW-style decay azimuth and beam-sign; using the map
    consistently for ALL blocks keeps generator/extractor self-consistent).
    Mode A: S_L=S_T=0 reproduces production W bit-exactly. phiS is measured from the
    production-plane azimuth like Diehl's phi_S (transverse target direction)."""
    vph, Phi, Pb = -np.asarray(phi), np.asarray(polphi), -heli
    u, l, s, n = ulsn_fields(A)
    w = eval_table(W_UU_TABLE, u, "d", c, vph, Phi, eps) \
        + Pb * eval_table(W_LU_TABLE, u, "d", c, vph, Phi, eps)
    if np.any(S_L != 0.0):
        w = w + S_L * (eval_table(W_UL_TABLE, l, "d", c, vph, Phi, eps)
                       + Pb * eval_table(W_LL_TABLE, l, "d", c, vph, Phi, eps))
    if np.any(S_T != 0.0):
        sn, cs = np.sin(Phi - phiS), np.cos(Phi - phiS)
        w = w + S_T * (sn * (eval_table(W_UU_TABLE, n, "n", c, vph, Phi, eps)
                             + Pb * eval_table(W_LU_TABLE, n, "n", c, vph, Phi, eps))
                       + cs * (eval_table(W_UL_TABLE, s, "d", c, vph, Phi, eps)
                               + Pb * eval_table(W_LL_TABLE, s, "d", c, vph, Phi, eps)))
    return w


def sample_events_full(A, eps, heli, n, rng, S_L=0.0, S_T=0.0, phiS=0.0,
                       oversample=4):
    """Accept-reject n decay events (c, phi, polphi) from W_full at a FIXED
    polarization state (S_L, S_T, phiS). Balanced-target datasets are built by
    calling this once per spin state with S -> -S."""
    out = ([], [], []); n_acc = 0
    cs = rng.uniform(-1, 1, 8000); ph = rng.uniform(-np.pi, np.pi, 8000)
    po = rng.uniform(-np.pi, np.pi, 8000)
    Wmax = 1.3 * np.max(W_full(cs, ph, po, eps, heli, A, S_L, S_T, phiS))
    while n_acc < n:
        m = int(max(n - n_acc, 1) * oversample)
        c = rng.uniform(-1, 1, m); ph = rng.uniform(-np.pi, np.pi, m)
        po = rng.uniform(-np.pi, np.pi, m)
        w = W_full(c, ph, po, eps, heli, A, S_L, S_T, phiS)
        acc = rng.uniform(0, Wmax, m) < w
        if np.any(w > Wmax):
            Wmax = 1.3 * float(w.max()); out = ([], [], []); n_acc = 0; continue
        out[0].append(c[acc]); out[1].append(ph[acc]); out[2].append(po[acc])
        n_acc += int(acc.sum())
    return tuple(np.concatenate(o)[:n] for o in out)


# =============================================================================
# AUDITS
# =============================================================================
if __name__ == "__main__":
    from amplitudes import amp_to_u28_batch
    from amplitudes_full import NAMP_FULL
    from diehl_w import W as W_modeA
    rng = np.random.default_rng(3)
    ok = lambda d, tol=1e-12: "PASS" if d < tol else "FAIL"

    A16 = rng.uniform(-0.4, 0.4, 16); A16[0] = abs(A16[0]) + 0.5
    A0 = np.concatenate([A16, np.zeros(18)])                  # Mode A embedded in 34
    A = np.concatenate([A16, rng.uniform(-0.3, 0.3, 18)])     # full, with flip block
    c = rng.uniform(-1, 1, 5000); vph = rng.uniform(-np.pi, np.pi, 5000)
    Phi = rng.uniform(-np.pi, np.pi, 5000); phiS = rng.uniform(-np.pi, np.pi, 5000)
    eps = 0.62

    # 0. complex u-fields == audited amp_to_u28_batch on every UNAMES slot
    uF = ulsn_fields(A0)[0]
    u28 = amp_to_u28_batch(A16[None, :])[0]
    mine = np.array([uF[_basename(nm)].imag if nm.startswith("Im_")
                     else uF[_basename(nm)].real for nm in UNAMES]).ravel()
    d0 = np.abs(mine - u28).max()
    print(f"0. complex u-fields vs amp_to_u28_batch (30 slots): {d0:.2e}  {ok(d0)}")

    # 1. table-built UU + heli*LU  ==  production diehl_w.W  (Mode-A bit-exact reduction)
    for heli in (+1, -1):
        wt = W_full(c, vph, Phi, eps, heli, A0)
        wp = W_modeA(c, vph, Phi, eps, heli, dict(zip(UNAMES, u28)))
        d1 = np.abs(wt - wp).max() / np.abs(wp).max()
        print(f"1. u-sector vs production W (heli={heli:+d}): rel {d1:.2e}  {ok(d1, 1e-10)}")

    # 2. longitudinal invariants: UL odd / LL even under (Phi,vph) -> (-Phi,-vph)
    _, lF, sF, nF = ulsn_fields(A)
    ul = eval_table(W_UL_TABLE, lF, "d", c, vph, Phi, eps)
    ulr = eval_table(W_UL_TABLE, lF, "d", c, -vph, -Phi, eps)
    ll = eval_table(W_LL_TABLE, lF, "d", c, vph, Phi, eps)
    llr = eval_table(W_LL_TABLE, lF, "d", c, -vph, -Phi, eps)
    d2a = np.abs(ul + ulr).max(); d2b = np.abs(ll - llr).max()
    print(f"2. W_UL odd {d2a:.2e} {ok(d2a)} | W_LL even {d2b:.2e} {ok(d2b)}")

    # 3. SIGNED U: longitudinal part flips under U0 -> -U0 (flip block zero)
    Af = A0.copy(); Af[9:16] *= -1
    wa = W_full(c, vph, Phi, eps, 0.85, A0, S_L=1.0) - W_full(c, vph, Phi, eps, 0.85, A0)
    wb = W_full(c, vph, Phi, eps, 0.85, Af, S_L=1.0) - W_full(c, vph, Phi, eps, 0.85, Af)
    d3 = np.abs(wa + wb).max()
    print(f"3. signed-U: S_L part odd under U0->-U0 at f1=0: {d3:.2e} (scale "
          f"{np.abs(wa).max():.3f})  {ok(d3)}")

    # 4. transverse sector == 0 when the flip block vanishes (n = s = 0)
    w4 = W_full(c, vph, Phi, eps, 0.85, A0, S_T=1.0, phiS=phiS) \
        - W_full(c, vph, Phi, eps, 0.85, A0)
    d4 = np.abs(w4).max()
    print(f"4. S_T sector vanishes at zero flip amplitudes: {d4:.2e}  {ok(d4)}")

    # 5. reflection parity incl. phiS: UT part odd, LT part even under
    #    (Phi, vph, phiS) -> (-Phi, -vph, -phiS)
    ut = W_full(c, vph, Phi, eps, 0.0, A, S_T=1.0, phiS=phiS) \
        - W_full(c, vph, Phi, eps, 0.0, A)
    utr = W_full(c, -vph, -Phi, eps, 0.0, A, S_T=1.0, phiS=-phiS) \
        - W_full(c, -vph, -Phi, eps, 0.0, A)
    lt = (W_full(c, vph, Phi, eps, 1.0, A, S_T=1.0, phiS=phiS)
          - W_full(c, vph, Phi, eps, 0.0, A, S_T=1.0, phiS=phiS)
          - W_full(c, vph, Phi, eps, 1.0, A) + W_full(c, vph, Phi, eps, 0.0, A))
    ltr = (W_full(c, -vph, -Phi, eps, 1.0, A, S_T=1.0, phiS=-phiS)
           - W_full(c, -vph, -Phi, eps, 0.0, A, S_T=1.0, phiS=-phiS)
           - W_full(c, -vph, -Phi, eps, 1.0, A) + W_full(c, -vph, -Phi, eps, 0.0, A))
    d5a = np.abs(ut + utr).max(); d5b = np.abs(lt - ltr).max()
    print(f"5. W_UT odd {d5a:.2e} {ok(d5a)} | W_LT even {d5b:.2e} {ok(d5b)}")

    # 6. POSITIVITY: physical density for pure polarization states, random amplitudes
    wmin = np.inf
    for _ in range(40):
        At = np.concatenate([rng.uniform(-0.5, 0.5, 16), rng.uniform(-0.4, 0.4, 18)])
        At[0] = abs(At[0]) + 0.05
        th = rng.uniform(0, np.pi); ps = rng.uniform(-np.pi, np.pi)
        w = W_full(c, vph, Phi, eps, rng.choice([-1.0, 1.0]), At,
                   S_L=np.cos(th), S_T=np.sin(th), phiS=ps)
        wmin = min(wmin, w.min())
    print(f"6. positivity over 40 random pure-state configs: min W = {wmin:.3e}  "
          f"{'PASS' if wmin > -1e-10 else 'FAIL'}")
