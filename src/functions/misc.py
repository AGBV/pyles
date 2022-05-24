import numpy as np

def jmult_max(num_part, lmax):
  return 2 * num_part * lmax * (lmax + 2)

def multi2single_index(j_s,tau,l,m,lmax):
  return j_s * 2 * lmax * (lmax+2) + (tau-1) * lmax * (lmax+2) + (l-1)*(l+1) + m + l

def single_index2multi(idx,lmax):
  j_s = idx // (2 * lmax * (lmax + 2))
  idx_new = idx % (2 * lmax * (lmax + 2))
  tau = idx_new // (lmax * (lmax + 2)) + 1
  idx_new = idx_new % (lmax * (lmax + 2))
  l = np.floor(np.sqrt(idx_new+1))
  m = idx_new - (l*l + l - 1)
  return j_s, tau, l, m

def transformation_coefficients(pilm, taulm, tau, l, m, pol, dagger: bool=False):
    if dagger:
      ifac = -1j
    else:
      ifac = 1j

    if tau == pol:
      spher_fun = taulm[l, np.abs(m)]
    else:
      spher_fun = m * pilm[l, np.abs(m)] * ~np.isnan(pilm[l, np.abs(m)])

    return -1 / np.power(ifac, l+1) / np.sqrt(2 * l * (l+1)) * (ifac * (pol == 1) + (pol == 2)) * spher_fun