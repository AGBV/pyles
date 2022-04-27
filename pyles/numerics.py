import logging

import numpy as np
import sympy as sym
from scipy.special import legendre

from pyles.functions.legendre_normalized_trigon import legendre_normalized_trigon

class Numerics:
  def __init__(self, lmax, polar_angles, azimuthal_angles, gpu=False, particle_distance_resolution = 10.0):
    self.lmax = lmax
    self.polar_angles = polar_angles
    self.azumuthal_angles = azimuthal_angles

    self.gpu = gpu
    self.particle_distance_resolution = particle_distance_resolution

    self.nmax = Numerics.compute_nmax(lmax)
    
    self.log = logging.getLogger(__name__)

    if gpu != False:
      self.log.warning('GPU functionality isn\'t implemented yet!\nReverting to CPU.')

  @staticmethod
  def compute_nmax(lmax):
    return 2 * lmax * (lmax + 2)

  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.lpmn.html
  def plm_coefficients(self):
    self.plm_coeff_table = np.zeros((
      2 * self.lmax + 1,
      2 * self.lmax + 1,
      self.lmax+1))
    
    ct = sym.Symbol('ct')
    st = sym.Symbol('st')
    plm = legendre_normalized_trigon(ct, y=st, lmax=2*self.lmax)
    
    for l in range(2*self.lmax+1):
      for m in range(l+1):
        cf = sym.poly(plm[l,m], ct, st).coeffs()
        self.plm_coeff_table[l,m,0:len(cf)] = cf
