#!/usr/bin/env python3
"""Relative beam-energy normalisation for the vecmeson blind test.

WHERE TO PUT THIS: in the SAME directory as generate_events.py -- specifically the
working copy that produced the phi LUND files, since the sampled box (Q2MIN/Q2MAX,
XBMIN/XBMAX, TMAX) and the amplitude shapes both enter the number.

    cd <that directory>
    python sigma_calib.py

It generates NOTHING and writes NOTHING: it only throws flat events in memory and
prints one number per beam energy. Importing generate_events is safe because its
event generation sits behind `if __name__ == "__main__"`.

WHY: the three phi runs each produced exactly 1e6 events, so their yields are
1 : 1 : 1 by construction rather than following sigma(E). The ratio of the numbers
printed below restores the physical relative luminosity, which is what the
sigma_L / sigma_T separation needs.

NOTE the estimator is sum(wphys) / n_thrown, NOT mean(wphys). throw() drops events
failing the kinematic cuts (y < y_max, W above threshold) and the survival fraction
is strongly energy dependent, so a mean over the returned events divides out most of
the effect (it gave 1 : 1.02 : 1.14 instead of the correct 1 : 1.32 : 2.11 in a test).
"""
import sys
import numpy as np

sys.argv = ["sigma_calib"]          # generate_events parses sys.argv for KEY=VALUE

from generate_events import throw, MESONS
import generate_events as G

MESON = "phi"
ENERGIES = (6.5, 7.5, 10.6)
N_THROW = 400_000
SEED = 0

meta = MESONS[MESON]
rng = np.random.default_rng(SEED)

print(f"# meson={MESON}  n_thrown={N_THROW}  seed={SEED}")
print(f"# sampled box: Q2 {G.Q2MIN}-{G.Q2MAX}   xB {G.XBMIN}-{G.XBMAX}   t' 0-{G.TMAX}")
print(f"# WEIGHT={getattr(G, 'WEIGHT', '?')}  BEAM_POL={getattr(G, 'BEAM_POL', '?')}")
print(f"# {'E [GeV]':>8} {'sum(wphys)/n_thrown':>22} {'survivors':>10}")
vals = []
for E in ENERGIES:
    d = throw(E, N_THROW, rng, meta["MV"], meta["width"], meta["MH"])
    s = float(np.sum(d["wphys"])) / N_THROW
    vals.append(s)
    print(f"  {E:>8} {s:>22.8g} {len(d['wphys']):>10}")

ref = vals[0]
print(f"# relative sigma(E), normalised to {ENERGIES[0]} GeV:")
print("#   " + "  ".join(f"{E}: {v/ref:.4f}" for E, v in zip(ENERGIES, vals)))
