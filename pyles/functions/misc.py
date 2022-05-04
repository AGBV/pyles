import numpy as np

def jmult_max(num_part, lmax):
  return 2 * num_part * lmax * (lmax + 2)

def multi2single_index(j_s,tau,l,m,lmax):
  return j_s * 2 * lmax * (lmax+2) + (tau-1) * lmax * (lmax+2) + (l-1)*(l+1) + m + l

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