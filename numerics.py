from distutils.log import warn
import numpy as np
from scipy.special import legendre

from functions.legendre_normalized_trigon import legendre_normalized_trigon

class Numerics:
  def __init__(self, lmax, polar_angles, azimuth_angles, gpu=False):
    self.lmax = lmax
    self.polar_angles = polar_angles
    self.azumuth_angles = azimuth_angles
    self.gpu = gpu

    self.nmax = Numerics.compute_nmax(lmax)

    if gpu != False:
      warn("GPU functionality isn't implemented yet!")

  @staticmethod
  def compute_nmax(lmax):
    return 2 * lmax * (lmax + 2)

  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.lpmn.html
  def plm_coefficients(self):
    self.plm_coeff_table = np.zeros((
      2 * self.lmax + 1,
      2 * self.lmax + 1,
      self.lmax+1))
    
    import sympy as sym
    from functions.legendre_normalized_trigon import legendre_normalized_trigon
    ct = sym.Symbol('ct')
    st = sym.Symbol('st')
    plm = legendre_normalized_trigon(ct, 2*self.lmax, y=st)
    
    for l in range(2*self.lmax+1):
      for m in range(l+1):
        cf = sym.poly(plm[l,m], ct, st).coeffs()
        self.plm_coeff_table[l,m,0:len(cf)] = cf
