#!/usr/bin/env python3
# =============================================================================
# CASE TEST: pure T11 = 1 (constant amplitude), Q^2-slope of the yield.
#
# A constant amplitude carries NO Q^2 dependence of its own, so the Q^2 shape of the
# generated yield is set entirely by the generator's KINEMATIC factors.  For pure T11 the
# angle-integrated intensity is sigma_T = |T11|^2 = 1 (sigma_L = 0), so:
#
#   WEIGHT=amp   dN/dQ^2 = A(Q^2)              (phase-space acceptance ONLY: the cuts
#                                               y<0.99, E'>0.3, W^2>(M+m_V)^2 sculpt Q^2
#                                               even for a flat weight -- it is NOT flat)
#   WEIGHT=flux  dN/dQ^2 = A(Q^2) * Gamma(Q^2) with the Diehl flux
#                                               Gamma ~ (y^2/(1-eps))*(1-xB)/xB*(1/Q^2)
#   RATIO flux/amp = Gamma(Q^2)                (the acceptance A(Q^2) divides out) -> this
#                                               isolates the flux and is the clean 1/Q^2 test.
#
# The explicit 1/Q^2 in the flux dominates, modulated by y^2/(1-eps) and (1-xB)/xB
# integrated over the sampled x_B window, so the RATIO slope sits near 2 (~1/Q^2).
#
# Runs the REAL generator (subprocess), reads Q^2 from the Lund header, fits
# dN/dQ^2 ~ Q^(-n) for each weight and for the ratio, and compares to 1/Q^2.
# Writes test_T11_Q2slope.pdf.
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

N        = int(os.environ.get("N", "150000"))
Q2MIN, Q2MAX = 1.0, 9.0
XBMIN, XBMAX = 0.08, 0.50
E        = 10.6


def run_generator(weight):
    """Run the real generator for pure T11=1 and return the accepted per-event Q^2 array.
    The generator always writes to <repo>/LUND_files (relative to its own file, not cwd),
    so read that and parse it right after each run (runs are sequential)."""
    env = dict(os.environ,
               MESON="rho0", N=str(N), E=str(E),
               Q2MIN=str(Q2MIN), Q2MAX=str(Q2MAX), XBMIN=str(XBMIN), XBMAX=str(XBMAX),
               TMAX="4.0", WEIGHT=weight, AMP_FILE=AMP,
               LUND_KIN="1", LUND_TRUTH="0", CHUNK=str(N))
    subprocess.run([PY, GEN], cwd=REPO, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    lund = os.path.join(REPO, "LUND_files", f"rho0_{E:.1f}gev_0.lund")
    if not os.path.exists(lund):
        raise RuntimeError(f"no Lund file at {lund} for WEIGHT={weight}")
    # LUND_KIN=1, LUND_TRUTH=0 -> event header line = 10 std + 8 kin = 18 fields;
    # kin order (0-based after the 10 std): [10]=Q2 [11]=|t| [12]=xB [13]=W ...
    q2 = []
    with open(lund) as fh:
        for line in fh:
            p = line.split()
            if len(p) == 18:                      # event header line (has the kin columns)
                q2.append(float(p[10]))
    if not q2:
        raise RuntimeError(f"parsed 0 event-header lines from {lund} (column layout changed?)")
    return np.array(q2)


NBINS = 24
_EDGES = np.linspace(Q2MIN, Q2MAX, NBINS + 1)
_CENTRES = 0.5 * (_EDGES[:-1] + _EDGES[1:])
_WIDTH = np.diff(_EDGES)


def density(q2):
    """dN/dQ^2 and per-bin counts on the common binning."""
    counts, _ = np.histogram(q2, bins=_EDGES)
    return counts / _WIDTH, counts


def fit_power(centres, dens, counts):
    """Fit dens ~ Q^(-n) in log-log, weighting by sqrt(counts); return (n, n_err, mask)."""
    m = counts > 0
    coef, cov = np.polyfit(np.log(centres[m]), np.log(dens[m]), 1,
                           w=np.sqrt(counts[m].astype(float)), cov=True)
    return -coef[0], np.sqrt(cov[0, 0]), m


def main():
    print(f"[test] generating {N} pure-T11 events per weight (E={E}, "
          f"Q2 in [{Q2MIN},{Q2MAX}]) ...", flush=True)
    q2_amp  = run_generator("amp")
    q2_flux = run_generator("flux")
    print(f"[test] accepted: WEIGHT=amp {len(q2_amp)},  WEIGHT=flux {len(q2_flux)}")

    d_amp,  c_amp  = density(q2_amp)
    d_flux, c_flux = density(q2_flux)
    both = (c_amp > 0) & (c_flux > 0)
    ratio = np.where(both, d_flux / np.where(d_amp > 0, d_amp, 1), 0.0)
    c_ratio = np.where(both, np.minimum(c_amp, c_flux), 0)   # stats of the ratio ~ smaller count

    n_amp,   e_amp,   m_amp   = fit_power(_CENTRES, d_amp,  c_amp)
    n_flux,  e_flux,  m_flux  = fit_power(_CENTRES, d_flux, c_flux)
    n_ratio, e_ratio, m_ratio = fit_power(_CENTRES, ratio,  c_ratio)

    print("\n=== Q^2-slope fits  (dN/dQ^2 ~ Q^(-n)) ===")
    print(f"  WEIGHT=amp   : n = {n_amp:+.3f} +/- {e_amp:.3f}   phase-space acceptance only")
    print(f"  WEIGHT=flux  : n = {n_flux:+.3f} +/- {e_flux:.3f}   acceptance x flux")
    print(f"  flux / amp   : n = {n_ratio:+.3f} +/- {e_ratio:.3f}   FLUX isolated -> compare to 1/Q^2 (n=2)")

    # --- plot ---------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    panels = [(_CENTRES, d_amp,  m_amp,   n_amp,   e_amp,   "WEIGHT=amp (acceptance only)",   None),
              (_CENTRES, d_flux, m_flux,  n_flux,  e_flux,  "WEIGHT=flux (acceptance x flux)", None),
              (_CENTRES, ratio,  m_ratio, n_ratio, e_ratio, "flux / amp (FLUX isolated)",      2.0)]
    for ax, (c, y, m, n, e, tag, refn) in zip(axs, panels):
        c0, y0 = c[m][0], y[m][0]
        ax.plot(c[m], y[m], "o", ms=5, label=r"generated")
        if refn is not None:
            ax.plot(c[m], y0 * (c[m] / c0) ** (-refn), "--", label=r"$1/Q^2$ (n=2)")
        ax.plot(c[m], y0 * (c[m] / c0) ** (-n), "-", alpha=0.7, label=fr"fit $Q^{{-{n:.2f}}}$")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$Q^2$ [GeV$^2$]"); ax.set_ylabel(r"$dN/dQ^2$ [arb.]")
        ax.set_title(f"{tag}\nn = {n:.3f} $\\pm$ {e:.3f}")
        ax.legend(); ax.grid(True, which="both", alpha=0.25)
    fig.suptitle(r"Pure $T_{11}=1$ case test: $Q^2$ slope (constant amplitude "
                 r"$\Rightarrow$ kinematics/flux only)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(HERE, "test_T11_Q2slope.pdf")
    fig.savefig(out); print(f"\n[test] wrote {out}")

    # PASS: the flux-isolated ratio follows ~1/Q^2 (explicit 1/Q^2 dominant, O(1) kinematic
    # modulation) -> slope in a band around 2.
    ok = 1.5 <= n_ratio <= 2.5
    print("\nRESULT:", "PASS -- flux ~ 1/Q^2 confirmed" if ok
          else f"CHECK -- flux slope {n_ratio:.2f} outside [1.5,2.5]")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
