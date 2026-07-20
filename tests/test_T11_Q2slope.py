#!/usr/bin/env python3
# =============================================================================
# CASE TEST: pure T11 = 1 (constant amplitude), Q^2 dependence of the yield.
#
# CONVENTION: fit dN/dQ^2 ~ (Q^2)^(-n).
#     "1/Q^2 dependence"  means  dN/dQ^2 proportional 1/Q^2 = (Q^2)^-1  ->  n = 1
#     "1/Q^4 dependence"  means  dN/dQ^2 proportional 1/Q^4 = (Q^2)^-2  ->  n = 2
#
# The generator samples Q^2 UNIFORMLY (flat) -- it does NOT sample from 1/Q^2 -- and weights
# each event by the cross section.  For pure T11 (constant amplitude, sigma_L = 0) the weight is
# the Diehl virtual-photon flux
#       Gamma = (y^2/(1-eps)) * (1-xB)/xB * (1/Q^2)          [y = Q^2/(2 M xB E)].
# ONLY the last factor is 1/Q^2.  Hence:
#
#   weight = 1/Q^2 exactly              -> dN/dQ^2 ~ 1/Q^2  (n = 1.0)  put 1/Q^2 in, get it out
#   full flux at FIXED xB               -> dN/dQ^2 ~ 1/Q^2  (n ~ 1.3)  explicit 1/Q^2 dominates
#   full flux, xB integrated (the gen)  -> dN/dQ^2 ~ 1/Q^4  (n ~ 2.0)  extra y^2/(1-eps),
#                                                                      (1-xB)/xB factors + the
#                                                                      shrinking physical xB range
#                                                                      (y<1) steepen it
#
# So the generator's RAW dN/dQ^2 for a constant amplitude is ~1/Q^4, NOT 1/Q^2 -- because the
# flux is 1/Q^2 TIMES extra Q^2-dependent factors, integrated over x_B.  This test shows all
# three, so the "1/Q^2 vs 1/Q^4" question is unambiguous.  Writes test_T11_Q2slope.pdf.
#
#   python tests/test_T11_Q2slope.py            # uses the repo generator + this dir's amp file
# =============================================================================
import os, sys, subprocess, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
GEN  = os.path.join(REPO, "generate_events.py")
AMP  = os.path.join(HERE, "amp_T11only.py")
PY   = sys.executable

N            = int(os.environ.get("N", "150000"))
Q2MIN, Q2MAX = 1.0, 9.0
XBMIN, XBMAX = 0.08, 0.50
E, M         = 10.6, 0.938272

NBINS  = 24
_EDGES = np.linspace(Q2MIN, Q2MAX, NBINS + 1)
_C     = 0.5 * (_EDGES[:-1] + _EDGES[1:])
_W     = np.diff(_EDGES)


def run_generator_flux():
    """Run the REAL generator for pure T11=1, WEIGHT=flux, return accepted per-event Q^2."""
    env = dict(os.environ, MESON="rho0", N=str(N), E=str(E),
               Q2MIN=str(Q2MIN), Q2MAX=str(Q2MAX), XBMIN=str(XBMIN), XBMAX=str(XBMAX),
               TMAX="4.0", WEIGHT="flux", AMP_FILE=AMP,
               LUND_KIN="1", LUND_TRUTH="0", CHUNK=str(N))
    subprocess.run([PY, GEN], cwd=REPO, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    lund = os.path.join(REPO, "LUND_files", f"rho0_{E:.1f}gev_0.lund")
    q2 = [float(p[10]) for p in (l.split() for l in open(lund)) if len(p) == 18]
    if not q2:
        raise RuntimeError(f"parsed 0 event-header lines from {lund}")
    return np.array(q2)


def slope(q2):
    h, _ = np.histogram(q2, bins=_EDGES); d = h / _W; m = h > 0
    n = -np.polyfit(np.log(_C[m]), np.log(d[m]), 1, w=np.sqrt(h[m].astype(float)))[0]
    return d, m, n


def standalone(weight_kind, rng):
    """Flat-throw (Q^2,xB), accept-reject by a chosen weight, return accepted Q^2.
    Isolates the WEIGHT's effect from the generator's other machinery."""
    Nt = 6_000_000
    Q2 = rng.uniform(Q2MIN, Q2MAX, Nt); xB = rng.uniform(XBMIN, XBMAX, Nt)
    y = Q2 / (2 * M * xB) / E
    gam = 2 * xB * M / np.sqrt(Q2)
    eps = (1 - y - 0.25 * gam**2 * y**2) / (1 - y + 0.5 * y**2 + 0.25 * gam**2 * y**2)
    phys = (y > 0) & (y < 0.99)
    flux = (y**2 / np.clip(1 - eps, 1e-3, None)) * (1 - xB) / np.clip(xB, 1e-3, None) / Q2
    if weight_kind == "inv_q2":
        w = 1.0 / Q2                                             # pure 1/Q^2, all xB, no cut
    elif weight_kind == "flux_fixed_xb":
        w = np.where(phys & (xB > 0.24) & (xB < 0.26), flux, 0.0)
    else:                                                       # "flux_int": full flux, xB integrated
        w = np.where(phys, flux, 0.0)
    w = np.nan_to_num(w, nan=0, posinf=0)
    wm = np.percentile(w[w > 0], 99.9)
    acc = rng.uniform(0, 1, Nt) < np.clip(w / wm, 0, 1)
    return Q2[acc]


def main():
    print(f"[test] generating {N} pure-T11 (WEIGHT=flux) events ...", flush=True)
    q2_gen = run_generator_flux()
    d_gen, m_gen, n_gen = slope(q2_gen)

    rng = np.random.default_rng(1)
    ladder = []
    for kind, lab in [("inv_q2",        "weight = 1/Q^2 exactly"),
                      ("flux_fixed_xb", "full flux, xB fixed ~0.25"),
                      ("flux_int",      "full flux, xB integrated")]:
        d, m, n = slope(standalone(kind, rng))
        ladder.append((lab, d, m, n))

    print("\nCONVENTION: dN/dQ^2 ~ (Q^2)^-n.  1/Q^2 <=> n=1.  1/Q^4 <=> n=2.\n")
    print(f"  GENERATOR (WEIGHT=flux) raw output : n = {n_gen:.2f}   ~ 1/Q^4  (NOT 1/Q^2)")
    for lab, _, _, n in ladder:
        tag = "= 1/Q^2" if abs(n - 1) < 0.15 else ("~ 1/Q^2" if n < 1.6 else "~ 1/Q^4")
        print(f"  {lab:34s}: n = {n:.2f}   {tag}")

    # --- plot: generator output (left) + the weight ladder (right) ----------
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axs[0]
    c0, y0 = _C[m_gen][0], d_gen[m_gen][0]
    ax.plot(_C[m_gen], d_gen[m_gen], "o", ms=5, label="generator (WEIGHT=flux)")
    ax.plot(_C[m_gen], y0 * (_C[m_gen] / c0) ** (-1.0), "--", label=r"$1/Q^2$ (n=1)")
    ax.plot(_C[m_gen], y0 * (_C[m_gen] / c0) ** (-2.0), ":",  label=r"$1/Q^4$ (n=2)")
    ax.plot(_C[m_gen], y0 * (_C[m_gen] / c0) ** (-n_gen), "-", alpha=0.7,
            label=fr"fit $(Q^2)^{{-{n_gen:.2f}}}$")
    ax.set_title(f"Generator raw output (const. amplitude)\nn = {n_gen:.2f}  ->  ~ $1/Q^4$, not $1/Q^2$")
    ax.legend()

    ax = axs[1]
    styles = ["s", "^", "o"]
    for (lab, d, m, n), st in zip(ladder, styles):
        c0, y0 = _C[m][0], d[m][0]
        ax.plot(_C[m], d[m] / y0, st, ms=4, label=f"{lab}  (n={n:.2f})")
    ax.plot(_C, (_C / _C[0]) ** (-1.0), "--", color="k", alpha=0.6, label=r"$1/Q^2$ (n=1)")
    ax.plot(_C, (_C / _C[0]) ** (-2.0), ":",  color="k", alpha=0.6, label=r"$1/Q^4$ (n=2)")
    ax.set_title("Why: the weight is $1/Q^2$ ONLY as one factor of the flux\n"
                 r"$\Gamma=(y^2/(1-\varepsilon))\,(1-x_B)/x_B\,(1/Q^2)$")
    ax.legend(fontsize=8)

    for ax in axs:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$Q^2$ [GeV$^2$]"); ax.set_ylabel(r"$dN/dQ^2$ [arb.]")
        ax.grid(True, which="both", alpha=0.25)
    fig.suptitle(r"Pure $T_{11}=1$: the generator outputs the full Diehl flux ($\sim 1/Q^4$), "
                 r"not $1/Q^2$; put pure $1/Q^2$ in and you get $1/Q^2$ out", fontsize=12)
    fig.tight_layout()
    out = os.path.join(HERE, "test_T11_Q2slope.pdf")
    fig.savefig(out); print(f"\n[test] wrote {out}")

    # PASS: pure-1/Q^2 weight reproduces 1/Q^2 (n~1) AND the generator matches the full flux
    # (~1/Q^4, n~2) -- i.e. the kinematic factors are implemented correctly.
    n_inv = ladder[0][3]; n_int = ladder[2][3]
    ok = abs(n_inv - 1.0) < 0.15 and abs(n_gen - n_int) < 0.4
    print("\nRESULT:", "PASS -- 1/Q^2 in -> 1/Q^2 out, and the generator matches the full Diehl flux"
          if ok else "CHECK -- see slopes above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
