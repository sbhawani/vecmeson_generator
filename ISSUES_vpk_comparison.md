# Kinematic dependencies & issues — vecmeson-generator vs vpK

Findings from the like-for-like comparison (pure-`T00` config: `u^{00}_{00}` only, so the
u-matrix is **constant** and every difference below is *kinematics only*, not angular physics).
Matched inputs: E=10.6 GeV, Q²∈[1,9], xB∈[0.09,0.68], |t|≤5.5, ρ⁰; vpK 60k / vecmeson-gen 55k events.
See `vecmeson_vpk_comparison.pdf` for the plots. Code refs are `file:function` in this repo unless noted;
vpK refs are in `/group/gpd/rho/vpk/`.

Legend: ✅ agrees · ⚠️ differs · 🔴 bug/inconsistency in vecmeson-gen.

| # | Observable | Status | Root cause | Code |
|---|-----------|--------|-----------|------|
| 1 | Q² | ✅ | both weights ∝ 1/Q² dominant | — |
| 2 | Φ (Trento) | ✅ (flat) | no L-T/T-T interference for pure T00 | — |
| 3 | cosθ, φ (decay) | ✅ **resolved** | frame offset (helicity CM vs lab axis); vpK's `produce__3` patched to build the decay about the CM meson direction | `vpk/reaction.h:produce__3` (patched copy) |
| 4 | y, ε | ✅ **FIXED** | flux differed (vpK ∝ y²/(1−ε), vecmeson-gen Hand ∝ (1−y)); **`WEIGHT=vpk`** now reproduces vpK's dsigma_3fold | `generate_events.py:throw` (WEIGHT=vpk) |
| 5 | x_B | ✅ (via #4) | was vecmeson-gen-higher from the flux; agrees under `WEIGHT=vpk` | #4 |
| 6 | W | ✅ (via #4) | downstream of x_B; agrees under `WEIGHT=vpk` | — |
| 7 | \|t\| | ✅ | (a) FIXED `tprime < TMAX`; (b) the 2q*p* Jacobian cancels vpK's uniform-cosθ* sampling → no fix needed; agrees under `WEIGHT=vpk` | `generate_events.py:throw` |
| 8 | m_V (meson mass) | ⚠️ (minor) | **both** relativistic BW; pole M₀,Γ match to a few MeV — only structural diff is the **Blatt–Weisskopf barrier** (vecmeson-gen has it, vpK bare Jackson) | `generate_events.py:sample_meson_mass`; vpK `reaction.h:sample_bw` |

## Detailed notes

**#3 Decay-angle frame (the main physics finding).**
vecmeson-gen uses the standard s-channel **helicity** frame (ẑ = meson momentum in the γ*p CM);
vpK's `decay2body.h` sets `uz = parent.vect()` = meson momentum **in the lab**. The two differ by the
lab↔CM meson-direction rotation, which grows with |t|. In the correct (helicity) frame vecmeson-gen gives exact
cos²θ and flat φ; vpK gives them only in the lab frame. **✅ RESOLVED:** patched vpK's `produce__3`
(`vpk/reaction.h`, working copy) to build the decay about the meson direction in the γ*p CM — reusing
`decay()` unchanged, fed the CM-boosted meson, daughters boosted back to the lab. Now vpK and vecmeson-gen agree on
cosθ/φ in the helicity frame for **every amplitude** (all seven observables match). Full before→after in PDF
§9. Offered for H.A.'s consideration (the W-kernels are defined in the helicity frame).

**#4/#5 Flux — ✅ FIXED.** Ours `WEIGHT=flux` used the Hand transverse flux `Γ_T ∝ (1−y)·K/(Q²(1−ε))`;
vpK uses `dsigma_3fold ∝ (y²/(1−ε))·(1−xB)/xB·(1/Q²)·(1+ε)`. The y-dependence was essentially opposite
(y² vs 1−y), so vpK sat at higher y (⟨y⟩=0.49 vs 0.36), lower ε, lower x_B, higher W. **Fix:** added a
**`WEIGHT=vpk`** mode reproducing `dsigma_3fold`. Under it y (⟨y⟩=0.47 vs 0.49), ε (0.78 vs 0.76),
x_B (0.27 vs 0.26), W (2.74 vs 2.80) and |t| (2.72 vs 2.69) all agree — the ratio panels flatten to ~1.
(Neither flux is "wrong"; a matched validation just needs the SAME one in both.)

**#7 |t| — two separate issues.**
(a) ✅ **FIXED** — the acceptance mask in `throw()` now uses `tprime < TMAX` (was a hardcoded `tprime < 4`
that silently ignored `TMAX>4`).
(b) ✅ **resolved (no fix needed)** — vpK's 2·q*·p* factor only compensates its own uniform-cosθ*_CM sampling
and cancels in the accepted dN/dt; vecmeson-gen samples t′ flat, so no Jacobian is required. With the flux matched
(#4), |t| agrees (⟨|t|⟩=2.72 vs 2.69). **Residual:** vecmeson-gen' t′∈[0,TMAX] lets |t| run past vpK's |t|≤5.5 cut
(vecmeson-gen max ≈7 vs vpK 5.5) — apply a matching |t| cut in the final comparison (or in `compare.py`).

**#8 Meson lineshape.** BOTH are relativistic Breit–Wigners (denominator `(m²−M₀²)² + M₀²Γ(m)²`). They
differ only in details: (i) the running width — vecmeson-gen multiplies the Jackson P-wave width by a
**Blatt–Weisskopf L=1 barrier** (R=5 GeV⁻¹) that suppresses the high-mass tail, vpK uses the **bare Jackson**
form; (ii) the pole mass — vecmeson-gen **M₀=0.77526 GeV** (current PDG) vs vpK **0.77** (`gagrho.cpp:49`; the
README's 0.7683 is stale); (iii) Γ₀ 0.1491 vs 0.1502; (iv) mass window vecmeson-gen [2mπ, M₀+8Γ] vs vpK
[2mπ, min(W−Mp, M₀+5Γ)]. The pole mass/width agree to a few MeV → **the only structural difference is the
Blatt–Weisskopf barrier**, which sharpens the peak and trims the high-mass tail (generated ⟨m_V⟩=0.865 vpK vs
0.832 vecmeson-gen, from the corrected sample). See the PDF §8 for the lineshape figure and the discussion points for
H.A. Arguably vecmeson-gen is the more complete parametrisation; for a matched run add the barrier to vpK or a
`BW_BARRIER=0` knob in vecmeson-gen.

## Other issues (not per-observable)

- **t-sign for the u-matrix port.** vpK feeds **Mandelstam t < 0** into its u-matrix
  (`tdepl = exp(4·t)` decays); vecmeson-gen passes **|t| > 0** to `user_amplitudes`. When porting vpK's default
  u-matrix into vecmeson-gen, every `exp(a·t)` must become `exp(−a·|t|)`. (compare.py's `|t| = −(q−V)²` is correct.)
- **Acceptance cuts differ.** vecmeson-gen: `y<0.99`, `E'>0.3 GeV`, `W²>(M+m_V)²`; vpK: `W∈[1.8,100]`, t-cuts.
  The `E'>0.3` + `y<0.99` sculpt the high-y tail differently from vpK.
- **Sampling.** vecmeson-gen flat in (Q²,x_B,t′); vpK log-Q², uniform x_B, uniform cosθ*_CM. Accept–reject removes
  the sampling bias only if the weight carries the correct Jacobians (see #7b).

## Status / next
**All matched conditions now agree.** Flux (#4, `WEIGHT=vpk`) + `tprime < TMAX` (#7a) → Q², |t|, x_B, W, y, ε
match; the full amplitude sweep {T00,T01,T11,T10,T1m1} agrees one-by-one; and the decay **frame** (#3) is
reconciled by patching vpK's `produce__3` to the CM axis — so all seven observables now overlap for the whole
natural-parity set (PDF §9). Only the m_V **lineshape** (#8, minor, Blatt–Weisskopf) remains, plus the |t|≤5.5
cut. Optional next: the full interfering model (all amplitudes on, the Φ-dependence) and high statistics.
