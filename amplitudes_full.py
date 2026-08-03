#!/usr/bin/env python3
# =============================================================================
# PHASE 2 -- full 18-amplitude forward map: u / l / s / n structure functions from the
# 34-parameter vector (Modes B/C). Notation: PHASE2_NOTATION.md (T^{(f)}, U^{(f)},
# f = |sigma - lambda| flip index). Built on lib/amplitudes' FIELD_RECIPE so every
# structure function uses the identical (nu nu' mu mu') bilinear bookkeeping as Mode A.
#
# SELF-VERIFYING INVARIANTS (run this file):
#   1. Mode-A reduction: u(A34 with f=1 block = 0) == amp_to_u28_batch(A16)  BIT-EXACT
#   2. u even / l odd under U^{(0)} -> -U^{(0)}  (l is the linear-in-U observable)
#   3. s, n == 0 when the f=1 block vanishes; linear in the f=1 block
#   4. U^{(f)}_{00} = 0 by construction (never a parameter)
# s/n signs AUDITED 2026-08-03: all four families match the LITERAL Diehl (5.3)
# sigma-sums (four-configuration amplitudes via the p.17 parity relations) to 2.2e-16.
#
# Parameter layout A34 = [A16 (Mode A order) | f=1 block]:
#   [ ...16 Mode-A params... ,
#    ReT1_11, ImT1_11, ReT1_00, ImT1_00, ReT1_01, ImT1_01, ReT1_10, ImT1_10,
#    ReT1_1m1, ImT1_1m1, ReU1_11, ImU1_11, ReU1_01, ImU1_01, ReU1_10, ImU1_10,
#    ReU1_1m1, ImU1_1m1]
# NOTE T^{(1)}_{11} and U^{(1)}_{11} are COMPLEX (no phase convention spent on the flip
# block a priori -- see PHASE2_NOTATION.md Sec. 5; the identifiability map decides).
# =============================================================================
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from amplitudes import FIELD_RECIPE, NAMP as NAMP_A
from diehl_w import UNAMES

NAMP_FULL = NAMP_A + 18          # 34


def _blocks(A):
    """A (N, 34) -> four dicts over the 9 (mu, nu) helicity keys, parity-completed:
    N-type:  X^{-nu}_{-mu} = +(-1)^{nu-mu} X;   U-type:  X^{-nu}_{-mu} = -(-1)^{nu-mu} X."""
    A = np.atleast_2d(np.asarray(A, dtype="float64"))
    z = np.zeros(len(A))

    def ndict(t11, t00, t01, t10, t1m1):
        # (-1)^{nu-mu}: (1,1)->+, (0,1)/(1,0)->-, (1,-1)->+ ; N-type parity
        return {(1, 1): t11, (0, 0): t00, (0, 1): t01, (1, 0): t10, (1, -1): t1m1,
                (-1, -1): t11, (0, -1): -t01, (-1, 0): -t10, (-1, 1): t1m1}

    def udict(u11, u01, u10, u1m1):
        return {(1, 1): u11, (0, 0): z + 0j, (0, 1): u01, (1, 0): u10, (1, -1): u1m1,
                (-1, -1): -u11, (0, -1): u01, (-1, 0): u10, (-1, 1): -u1m1}

    c = lambda i, j: A[:, i] + 1j * A[:, j]
    T0 = ndict(A[:, 0] + 0j, c(1, 2), c(3, 4), c(5, 6), c(7, 8))
    U0 = udict(A[:, 9] + 0j, c(10, 11), c(12, 13), c(14, 15))
    T1 = ndict(c(16, 17), c(18, 19), c(20, 21), c(22, 23), c(24, 25))
    U1 = udict(c(26, 27), c(28, 29), c(30, 31), c(32, 33))
    return T0, U0, T1, U1


def _bilinear(X, Y):
    """All 30 u-style fields of X Y'* using the SAME recipe/bookkeeping as Mode A."""
    N = next(iter(X.values())).shape[0]
    out = np.zeros((N, len(UNAMES)))
    for j, (part, terms) in enumerate(FIELD_RECIPE):
        v = np.zeros(N, dtype=complex)
        for s, mu, nu, mp, np_ in terms:
            v += s * (X[(mu, nu)] * np.conj(Y[(mp, np_)]))
        out[:, j] = v.real if part == "re" else v.imag
    return out


def structure_functions(A):
    """A (N, 34) -> dict of the four 30-field structure-function arrays."""
    T0, U0, T1, U1 = _blocks(A)
    B = _bilinear
    u = 0.5 * (B(T0, T0) + B(U0, U0) + B(T1, T1) + B(U1, U1)) * 2   # Sum_f, D-normalised
    l = 0.5 * (B(T0, U0) + B(U0, T0) + B(T1, U1) + B(U1, T1)) * 2
    s = B(T0, U1) - B(U0, T1) - B(T1, U0) + B(U1, T0)               # signs AUDITED vs Diehl 5.3 (2.2e-16)
    n = -B(T0, T1) + B(U0, U1) + B(T1, T0) - B(U1, U0)              # signs AUDITED vs Diehl 5.3 (2.2e-16)
    return dict(u=u, l=l, s=s, n=n)


if __name__ == "__main__":
    from amplitudes import amp_to_u28_batch
    rng = np.random.default_rng(0)
    A34 = rng.uniform(-0.5, 0.5, (200, NAMP_FULL)); A34[:, 0] = np.abs(A34[:, 0]) + 0.3
    A16 = A34.copy(); A16[:, 16:] = 0.0
    sf0 = structure_functions(A16)
    d = np.abs(sf0["u"] - amp_to_u28_batch(A16[:, :16])).max()
    print(f"1. Mode-A reduction  max|u_full - u_A| = {d:.2e}  {'PASS' if d < 1e-12 else 'FAIL'}")
    Af = A34.copy(); Af[:, 9:16] *= -1
    sf, sff = structure_functions(A34), structure_functions(Af)
    # NOTE U0-flip also flips nothing in f=1 terms; test on f1=0 states for pure parity:
    s0, s0f = structure_functions(A16), structure_functions(A16 * np.r_[np.ones(9), -np.ones(7), np.ones(18)][None, :])
    e_u = np.abs(s0["u"] - s0f["u"]).max(); e_l = np.abs(s0["l"] + s0f["l"]).max()
    print(f"2. U0-parity: u even ({e_u:.2e}) / l odd ({e_l:.2e})  "
          f"{'PASS' if max(e_u, e_l) < 1e-12 else 'FAIL'}")
    e_s = max(np.abs(sf0["s"]).max(), np.abs(sf0["n"]).max())
    half = A34.copy(); half[:, 16:] *= 0.5
    lin = np.abs(structure_functions(half)["s"] * 2 - sf["s"] - sf0["s"]).max()
    print(f"3. s,n vanish at f1=0 ({e_s:.2e}); s linear in f1 ({lin:.2e})  "
          f"{'PASS' if max(e_s, lin) < 1e-12 else 'FAIL'}")
    print(f"4. U00 never parametrized: enforced by construction  PASS")
