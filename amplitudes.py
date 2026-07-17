#!/usr/bin/env python3
# =============================================================================
# FULL unpolarised (nucleon-helicity NON-FLIP) helicity-amplitude set for phi
# electroproduction, extending amplitudes.py from natural-parity-only (5 complex,
# 9 real) to natural + UNNATURAL parity (9 complex, 17 real before identifiability
# check).  Requested by D. Glazier (single-energy L/T / full unpolarised set).
#
#   Natural   (T):  T_{-mu,-nu} = +(-1)^{mu-nu} T_{mu,nu}   -> T11,T00,T01,T10,T1m1
#   Unnatural (U):  U_{-mu,-nu} = -(-1)^{mu-nu} U_{mu,nu}   -> U11,U01,U10,U1m1 (U00=0)
#   Full amplitude   F_{mu,nu} = T_{mu,nu} + U_{mu,nu}.
#
# Everything downstream (u = F F^dagger, sigma_T/L, SDMEs) flows through build_T,
# so no other forward-model change is needed once build_T returns F.
# =============================================================================
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)
from diehl_w import UNAMES, sigma_T, sigma_L

# 16 real params: T11 real fixes the global phase; U11 real fixes the residual
# natural/unnatural continuous ambiguity (the 17th direction is unidentifiable
# from unpolarised data -- see identifiability test); U00 = 0 by parity.
AMP_NAMES = ["T11",
             "T00_re", "T00_im", "T01_re", "T01_im",
             "T10_re", "T10_im", "T1m1_re", "T1m1_im",
             "U11", "U01_re", "U01_im",
             "U10_re", "U10_im", "U1m1_re", "U1m1_im"]
NAMP = len(AMP_NAMES)
HMAP = {"p": 1, "m": -1, "0": 0, "1": 1}

# indices of the 7 unnatural params (for the "unnatural ~ 0" studies)
UNNATURAL_IDX = [AMP_NAMES.index(n) for n in
                 ["U11", "U01_re", "U01_im",
                  "U10_re", "U10_im", "U1m1_re", "U1m1_im"]]


def _pack(a):
    if not isinstance(a, dict):
        a = dict(zip(AMP_NAMES, a))
    T11 = complex(a["T11"], 0.0)
    T00 = complex(a["T00_re"], a["T00_im"]);  T01 = complex(a["T01_re"], a["T01_im"])
    T10 = complex(a["T10_re"], a["T10_im"]);  T1m1 = complex(a["T1m1_re"], a["T1m1_im"])
    U11 = complex(a["U11"], 0.0);             U01 = complex(a["U01_re"], a["U01_im"])
    U10 = complex(a["U10_re"], a["U10_im"]);  U1m1 = complex(a["U1m1_re"], a["U1m1_im"])
    return T11, T00, T01, T10, T1m1, U11, U01, U10, U1m1


def build_T(a):
    """params -> {(mu,nu): complex} for the FULL amplitude F = T + U."""
    T11, T00, T01, T10, T1m1, U11, U01, U10, U1m1 = _pack(a)
    return {
        (1, 1):  T11 + U11,   (0, 0):  T00,          (0, 1):  T01 + U01,
        (1, 0):  T10 + U10,   (1, -1): T1m1 + U1m1,
        (-1, -1): T11 - U11,  (0, -1): -T01 + U01,   (-1, 0): -T10 + U10,
        (-1, 1):  T1m1 - U1m1,
    }


def _term(term, T):
    mes, pho = term[1:].split("_")
    mu, mup = HMAP[mes[0]], HMAP[mes[1]]
    nu, nup = HMAP[pho[0]], HMAP[pho[1]]
    return T[(mu, nu)] * np.conj(T[(mup, nup)])


def field_value(name, T):
    part = "re"; s = name
    if s.startswith("Re_"): s = s[3:]
    elif s.startswith("Im_"): s = s[3:]; part = "im"
    if "_minus_" in s:
        a, b = s.split("_minus_"); val = _term(a, T) - _term(b, T)
    elif "_plus_" in s:
        a, b = s.split("_plus_"); val = _term(a, T) + _term(b, T)
    else:
        val = _term(s, T)
    return val.real if part == "re" else val.imag


def amp_to_u28(a):
    T = build_T(a)
    return np.array([field_value(n, T) for n in UNAMES], dtype="float64")


# ---- vectorised ------------------------------------------------------------
def _parse_field(name):
    part = "re"; s = name
    if s.startswith("Re_"): s = s[3:]
    elif s.startswith("Im_"): s = s[3:]; part = "im"
    if "_minus_" in s:
        a, b = s.split("_minus_"); terms = [(+1.0, a), (-1.0, b)]
    elif "_plus_" in s:
        a, b = s.split("_plus_"); terms = [(+1.0, a), (+1.0, b)]
    else:
        terms = [(+1.0, s)]
    out = []
    for sgn, t in terms:
        mes, pho = t[1:].split("_")
        out.append((sgn, HMAP[mes[0]], HMAP[pho[0]], HMAP[mes[1]], HMAP[pho[1]]))
    return part, out

FIELD_RECIPE = [_parse_field(n) for n in UNAMES]


def build_T_batch(A):
    A = np.atleast_2d(A)
    T11  = A[:, 0] + 0j
    T00  = A[:, 1] + 1j * A[:, 2];   T01  = A[:, 3] + 1j * A[:, 4]
    T10  = A[:, 5] + 1j * A[:, 6];   T1m1 = A[:, 7] + 1j * A[:, 8]
    U11  = A[:, 9] + 0j
    U01  = A[:, 10] + 1j * A[:, 11]; U10  = A[:, 12] + 1j * A[:, 13]
    U1m1 = A[:, 14] + 1j * A[:, 15]
    return {
        (1, 1):  T11 + U11,   (0, 0):  T00,          (0, 1):  T01 + U01,
        (1, 0):  T10 + U10,   (1, -1): T1m1 + U1m1,
        (-1, -1): T11 - U11,  (0, -1): -T01 + U01,   (-1, 0): -T10 + U10,
        (-1, 1):  T1m1 - U1m1,
    }


def amp_to_u28_batch(A):
    T = build_T_batch(A); N = next(iter(T.values())).shape[0]
    out = np.zeros((N, len(UNAMES)))
    for j, (part, terms) in enumerate(FIELD_RECIPE):
        v = np.zeros(N, dtype=complex)
        for sgn, mu, nu, mup, nup in terms:
            v += sgn * T[(mu, nu)] * np.conj(T[(mup, nup)])
        out[:, j] = v.real if part == "re" else v.imag
    return out


def amp_sigmas(a):
    u = dict(zip(UNAMES, amp_to_u28(a)))
    sT, sL = sigma_T(u), sigma_L(u)
    return sT, sL, (sL / sT if sT > 0 else 0.0)


# ============================ SELF-TEST =====================================
if __name__ == "__main__":
    # NOTE: diehl_w.u_to_r does not exist (it never has) -- this self-test has been dead code,
    # which is part of why the sigma_T bug below survived.  Degrade gracefully so the rest runs.
    try:
        from diehl_w import u_to_r
    except ImportError:
        u_to_r = None
        print("[warn] diehl_w.u_to_r missing -> skipping the SDME (r^alpha) cross-check")
    rng = np.random.default_rng(0)
    print("=== FULL amplitude set: self-tests ===")

    # (1) natural-parity-only reproduces amplitudes.py (U=0 -> old SCHC test)
    a = {n: 0.0 for n in AMP_NAMES}
    a["T11"] = 0.8; a["T00_re"] = 1.3
    sT, sL, R = amp_sigmas(a)
    # sigma_T = |T11|^2 + |T01|^2 + |T1m1|^2  -> |T11|^2 under SCHC (NOT 2|T11|^2: the lambda_V=-1
    # partner at a transverse photon is T1m1, not T11 -- see sigma_T in diehl_w.py).
    print(f"SCHC (U=0): sigma_T={sT:.4f} (|T11|^2={0.64:.4f}), "
          f"sigma_L={sL:.4f} (|T00|^2={1.69:.4f}), R={R:.4f}")
    assert np.isclose(sT, 0.64) and np.isclose(sL, 1.69), "SCHC sigma_T/sigma_L regression"
    if u_to_r is not None:
        for eps in (0.4, 0.7):
            r = u_to_r(dict(zip(UNAMES, amp_to_u28(a))), eps)
            schc = eps * R / (1 + eps * R)
            print(f"  eps={eps}: r04_00={r['r00_04']:+.5f}  SCHC={schc:+.5f}  "
                  f"match={np.isclose(r['r00_04'], schc)}")

    # (2) positivity with unnatural amplitudes on
    ok = True
    for _ in range(5000):
        av = rng.uniform(-1, 1, NAMP); av[0] = abs(av[0])
        s, l, _ = amp_sigmas(av)
        if s <= 0 or l < 0: ok = False; break
    print(f"positivity over 5000 random FULL amplitudes: {ok}")

    # (3) global-phase invariance
    av = rng.uniform(-1, 1, NAMP); av[0] = abs(av[0])
    T = build_T(av); Tp = {k: v * np.exp(1j * 0.7) for k, v in T.items()}
    du = max(abs(field_value(n, T) - field_value(n, Tp)) for n in UNAMES)
    print(f"global-phase invariance of u: max|du|={du:.2e}")

    # (4) IDENTIFIABILITY: rank of d(u28)/d(params) -> settles 16 vs 17
    def jac(av, h=1e-6):
        J = np.zeros((len(UNAMES), NAMP))
        for k in range(NAMP):
            ap = av.copy(); am = av.copy(); ap[k] += h; am[k] -= h
            J[:, k] = (amp_to_u28(ap) - amp_to_u28(am)) / (2 * h)
        return J
    ranks = []
    for _ in range(20):
        av = rng.uniform(-1, 1, NAMP); av[0] = abs(av[0])
        s = np.linalg.svd(jac(av), compute_uv=False)
        ranks.append(int((s > 1e-6 * s[0]).sum()))
    print(f"identifiable parameters (median SVD rank of du/dA over 20 pts): "
          f"{int(np.median(ranks))} / {NAMP}   (ranks={sorted(set(ranks))})")
    print("done.")
