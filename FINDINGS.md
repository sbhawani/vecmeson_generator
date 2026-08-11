
## 2026-08-10 (final): polarized BLIND path -- generator side VERIFIED, extraction side is the gap
Goal (user): produce U / U+L / U+L+T blind samples from the INDEPENDENT vecmeson_generator
so Harut can supply polarized blind data built from his own amplitudes.

GENERATOR: ALREADY COMPLETE (MODE=A/B/C, PT, PHIS, FLIPSCALE, user_amplitudes_flip hook,
per-event balanced spin states, W_full weighting, signed target polarization written to
LUND header field 4). Verified today:
  * MODE=B and MODE=C both run; field 4 carries +-PT in a balanced 50/50 draw.
  * PHYSICS CHECK (42k events, E=10.6, MODE=B): with the DEFAULT generator amplitudes
    (U == 0) there is NO spin signal -- max 1.8 sigma over 8 harmonics. That is correct
    physics, not a defect: the l sector is the natural-unnatural INTERFERENCE, linear in
    U, so it vanishes identically at U = 0.
  * With a NONZERO-U truth (amps_bhawani values, standalone copy) the same test gives
    sin(Phi) 17.9 sigma and sin(2phi) 8.7 sigma spin differences. The l-sector signal is
    genuinely present in the generated events.
REQUIREMENTS THIS PLACES ON HARUT'S AMPLITUDES:
  U      -- 16 non-flip params (he already has these).
  U+L    -- SAME 16 params, but the U (unnatural-parity) block MUST be non-zero, else the
            longitudinal arm is vacuous. No new amplitude functions needed.
  U+L+T  -- additionally the 18-parameter f=1 NUCLEON-FLIP block via user_amplitudes_flip:
            s and n vanish identically when the flip block is zero (amplitudes_full
            invariant 3), so without it the transverse arm carries no information.

EXTRACTION: THE GAP. pol_B/pol_C consume toy-chain features built at FIXED per-beam eps
slots (0.45/0.62/0.85), while real generator events in our bins sit at eps 0.226-0.937
(measured: E=6.5 ring 1.5 -> 0.805, ring 3.5 -> 0.226). Feeding LUND events to those nets
would be mis-specified by up to 0.46 in eps -- the same class of error as the 2026-08-06
Q^2 tilt. A polarized blind test therefore needs the polarized analogue of gen_ckpts:
an eps-conditioned polarized forward model + trained ensembles (a training campaign,
ifarm-scale), then a polarized blind driver. Generator work: none. Reduction work: read
field 4 (small).
