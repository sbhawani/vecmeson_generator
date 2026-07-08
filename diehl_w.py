#!/usr/bin/env python3
# Vectorised Python port of Path-2's W_unnorm (WUnnormDiehl.h):
#   W_unnorm = (sigma_T + eps*sigma_L) * W_SW(cosTh, phi, PolPhi, eps, heli; r^a)
# with the 23 SW r^a derived linearly from the 28 Diehl u's (DiehlUToNSDME).
# Provides: u_to_r, W (intensity), sample_events (accept-reject), and the
# 23 angular-basis moments used as ML features.
import numpy as np

# 28 Diehl u-fields, in DiehlTruth/CSV order
UNAMES = ["u00_pp","u00_00","Re_u0p_pp_minus_um0_pp","Re_u0p_00","Re_ump_pp","Re_ump_00",
 "u11_pp","u11_00","Re_upp_mp","u00_mp","Re_u0p_mp","Re_up0_mp","ump_mp","upm_mp",
 "Re_upp_0p_plus_umm_0p","Re_u00_0p","Re_u0p_0p_minus_um0_0p","Re_u0m_0p_minus_up0_0p",
 "Re_ump_0p","Re_upm_0p","Im_u0p_pp_minus_um0_pp","Im_ump_pp","Im_upp_0p_plus_umm_0p",
 "Im_u00_0p","Im_u0p_0p_minus_um0_0p","Im_u0m_0p_minus_up0_0p","Im_ump_0p","Im_upm_0p"]
S2 = np.sqrt(2.0)

def sigma_T(u): return 2*u["u11_pp"] + u["u00_pp"]
def sigma_L(u): return 2*u["u11_00"] + u["u00_00"]
def sigma_sum(u, eps): return sigma_T(u) + eps*sigma_L(u)

def u_to_r(u, eps):
    """28 u's -> 23 SW r^a (DiehlUToNSDME), as a dict."""
    inv = 1.0 / sigma_sum(u, eps)
    g = u.get
    return dict(
      r00_04=(g("u00_pp")+eps*g("u00_00"))*inv,
      r10_04=(0.5*g("Re_u0p_pp_minus_um0_pp")+eps*g("Re_u0p_00"))*inv,
      r1m1_04=(g("Re_ump_pp")+eps*g("Re_ump_00"))*inv,
      r11_1=g("Re_upp_mp")*inv, r00_1=g("u00_mp")*inv,
      r10_1=0.5*(g("Re_u0p_mp")+g("Re_up0_mp"))*inv,
      r1m1_1=0.5*(g("ump_mp")+g("upm_mp"))*inv,
      r10_2=0.5*(g("Re_up0_mp")-g("Re_u0p_mp"))*inv,
      r1m1_2=0.5*(g("upm_mp")-g("ump_mp"))*inv,
      r11_5=-g("Re_upp_0p_plus_umm_0p")/S2*inv, r00_5=-S2*g("Re_u00_0p")*inv,
      r10_5=(g("Re_u0m_0p_minus_up0_0p")-g("Re_u0p_0p_minus_um0_0p"))/(2*S2)*inv,
      r1m1_5=-S2*0.5*(g("Re_ump_0p")+g("Re_upm_0p"))*inv,
      r10_6=(g("Re_u0p_0p_minus_um0_0p")+g("Re_u0m_0p_minus_up0_0p"))/(2*S2)*inv,
      r1m1_6=S2*0.5*(g("Re_ump_0p")-g("Re_upm_0p"))*inv,
      r10_3=-0.5*g("Im_u0p_pp_minus_um0_pp")*inv, r1m1_3=-g("Im_ump_pp")*inv,
      r10_7=(g("Im_u0p_0p_minus_um0_0p")+g("Im_u0m_0p_minus_up0_0p"))/(2*S2)*inv,
      r1m1_7=S2*0.5*(g("Im_ump_0p")-g("Im_upm_0p"))*inv,
      r11_8=g("Im_upp_0p_plus_umm_0p")/S2*inv, r00_8=S2*g("Im_u00_0p")*inv,
      r10_8=(g("Im_u0p_0p_minus_um0_0p")-g("Im_u0m_0p_minus_up0_0p"))/(2*S2)*inv,
      r1m1_8=S2*0.5*(g("Im_ump_0p")+g("Im_upm_0p"))*inv)

# The 23 angular basis functions multiplying r^a (+ the constant A term).
R_ORDER = ["r00_04","r10_04","r1m1_04","r11_1","r00_1","r10_1","r1m1_1","r10_2",
 "r1m1_2","r11_5","r00_5","r10_5","r1m1_5","r10_6","r1m1_6","r10_3","r1m1_3",
 "r10_7","r1m1_7","r11_8","r00_8","r10_8","r1m1_8"]

def basis(c, phi, polphi, eps, heli):
    """Return (A_const, {name: f_name}) angular basis arrays (vectorised)."""
    pi = np.pi; s2 = 1-c*c; sT = np.sqrt(np.maximum(0,s2)); s2T = 2*c*sT
    cP, sP = np.cos(phi), np.sin(phi); c2P, s2P = np.cos(2*phi), np.sin(2*phi)
    cPol, sPol = np.cos(polphi), np.sin(polphi); c2Pol, s2Pol = np.cos(2*polphi), np.sin(2*polphi)
    eP = np.sqrt(2*eps*(1+eps)); eM = np.sqrt(2*eps*(1-eps)); e1 = np.sqrt(np.maximum(0,1-eps*eps))
    k = 3.0/(4*pi)
    f = {
      "r00_04": k/2*(3*c*c-1), "r10_04": -k*S2*s2T*cP, "r1m1_04": -k*s2*c2P,
      "r11_1": -eps*c2Pol*k*s2, "r00_1": -eps*c2Pol*k*c*c, "r10_1": eps*c2Pol*k*S2*s2T*cP,
      "r1m1_1": eps*c2Pol*k*s2*c2P,
      "r10_2": -eps*s2Pol*k*S2*s2T*sP, "r1m1_2": -eps*s2Pol*k*s2*s2P,
      "r11_5": eP*cPol*k*s2, "r00_5": eP*cPol*k*c*c, "r10_5": -eP*cPol*k*S2*s2T*cP,
      "r1m1_5": -eP*cPol*k*s2*c2P,
      "r10_6": eP*sPol*k*S2*s2T*sP, "r1m1_6": eP*sPol*k*s2*s2P,
      "r10_3": heli*e1*k*S2*s2T*sP, "r1m1_3": heli*e1*k*s2*s2P,
      "r10_7": heli*eM*cPol*k*S2*s2T*sP, "r1m1_7": heli*eM*cPol*k*s2*s2P,
      "r11_8": heli*eM*sPol*k*s2, "r00_8": heli*eM*sPol*k*c*c,
      "r10_8": -heli*eM*sPol*k*S2*s2T*cP, "r1m1_8": -heli*eM*sPol*k*s2*c2P}
    A = 3.0/(8*pi)*s2 + 0*c            # constant (no SDME) term, broadcast
    return A, f

def W(c, phi, polphi, eps, heli, u):
    A, f = basis(c, phi, polphi, eps, heli)
    r = u_to_r(u, eps)
    val = A.copy()
    for n in R_ORDER: val += f[n]*r[n]
    return val * sigma_sum(u, eps)

def sample_events(u, eps, heli, n, rng, oversample=4):
    """Accept-reject n decay events (c, phi, polphi) from W at fixed (eps,heli)."""
    out_c=[]; out_phi=[]; out_pol=[]
    # crude Wmax estimate
    cs=rng.uniform(-1,1,8000); ph=rng.uniform(-np.pi,np.pi,8000); po=rng.uniform(-np.pi,np.pi,8000)
    Wmax=1.3*np.max(W(cs,ph,po,eps,heli,u))
    while len(out_c)<n:
        m=int((n-len(out_c))*oversample)
        c=rng.uniform(-1,1,m); ph=rng.uniform(-np.pi,np.pi,m); po=rng.uniform(-np.pi,np.pi,m)
        w=W(c,ph,po,eps,heli,u); acc=rng.uniform(0,Wmax,m)<w
        out_c.append(c[acc]); out_phi.append(ph[acc]); out_pol.append(po[acc])
    c=np.concatenate(out_c)[:n]; phi=np.concatenate(out_phi)[:n]; pol=np.concatenate(out_pol)[:n]
    return c, phi, pol

def moments(c, phi, polphi, eps, heli):
    """23 angular-basis sample means -> the ML feature vector for this bin."""
    A, f = basis(c, phi, polphi, eps, heli)
    return np.array([np.mean(f[n]) for n in R_ORDER])


# Pretty Diehl labels for plots:  code name -> u^{nu nu'}_{mu mu'} (matplotlib mathtext)
DIEHL_LABEL = {
 "u00_pp": r"$u^{00}_{++}$",   "u00_00": r"$u^{00}_{00}$",
 "u11_pp": r"$u^{++}_{++}$",   "u11_00": r"$u^{++}_{00}$",
 "Re_u0p_pp_minus_um0_pp": r"$\mathrm{Re}(u^{0+}_{++}-u^{-0}_{++})$",
 "Re_u0p_00": r"$\mathrm{Re}\,u^{0+}_{00}$",
 "Re_ump_pp": r"$u^{-+}_{++}$",   "Re_ump_00": r"$u^{-+}_{00}$",
 "Re_upp_mp": r"$\mathrm{Re}\,u^{++}_{-+}$",   "u00_mp": r"$u^{00}_{-+}$",
 "Re_u0p_mp": r"$u^{0+}_{-+}$",   "Re_up0_mp": r"$u^{+0}_{-+}$",
 "ump_mp": r"$u^{-+}_{-+}$",      "upm_mp": r"$u^{+-}_{-+}$",
 "Re_upp_0p_plus_umm_0p": r"$\mathrm{Re}(u^{++}_{0+}+u^{--}_{0+})$",
 "Re_u00_0p": r"$\mathrm{Re}\,u^{00}_{0+}$",
 "Re_u0p_0p_minus_um0_0p": r"$\mathrm{Re}(u^{0+}_{0+}-u^{-0}_{0+})$",
 "Re_u0m_0p_minus_up0_0p": r"$\mathrm{Re}(u^{0-}_{0+}-u^{+0}_{0+})$",
 "Re_ump_0p": r"$u^{-+}_{0+}$",   "Re_upm_0p": r"$u^{+-}_{0+}$",
 "Im_u0p_pp_minus_um0_pp": r"$\mathrm{Im}(u^{0+}_{++}-u^{-0}_{++})$",
 "Im_ump_pp": r"$\mathrm{Im}\,u^{-+}_{++}$",
 "Im_upp_0p_plus_umm_0p": r"$\mathrm{Im}(u^{++}_{0+}+u^{--}_{0+})$",
 "Im_u00_0p": r"$\mathrm{Im}\,u^{00}_{0+}$",
 "Im_u0p_0p_minus_um0_0p": r"$\mathrm{Im}(u^{0+}_{0+}-u^{-0}_{0+})$",
 "Im_u0m_0p_minus_up0_0p": r"$\mathrm{Im}(u^{0-}_{0+}-u^{+0}_{0+})$",
 "Im_ump_0p": r"$\mathrm{Im}\,u^{-+}_{0+}$",   "Im_upm_0p": r"$\mathrm{Im}\,u^{+-}_{0+}$",
}
def dlabel(name): return DIEHL_LABEL.get(name, name)
