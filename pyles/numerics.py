import logging

import numpy as np
import sympy as sym
import pywigxjpf as wig
from scipy.special import legendre

from pyles.functions.misc import jmult_max
from pyles.functions.misc import multi2single_index
from pyles.functions.legendre_normalized_trigon import legendre_normalized_trigon


class Numerics:
  def __init__(self, lmax, polar_angles, azimuthal_angles, gpu=False, particle_distance_resolution = 10.0):
    self.lmax = lmax
    self.polar_angles = polar_angles
    self.azumuthal_angles = azimuthal_angles

    self.gpu = gpu
    self.particle_distance_resolution = particle_distance_resolution

    self.nmax = 2 * lmax * (lmax + 2)
    
    self.log = logging.getLogger(__name__)

    if gpu != False:
      self.log.warning('GPU functionality isn\'t implemented yet!\nReverting to CPU.')

    self.__setup()

  def __compute_nmax(self):
    self.nmax = jmult_max(1, self.lmax)

  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.lpmn.html
  def __plm_coefficients(self):
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

  def __setup(self):
    self.__compute_nmax()
    # self.__plm_coefficients()

  def compute_translation_table(self):
    jmax = jmult_max(1, self.lmax)
    self.translation_ab5 = np.zeros((jmax, jmax, 2 * self.lmax + 1), dtype=complex)

    wig.wig_table_init(3 * self.lmax, 3)
    wig.wig_temp_init(3 * self.lmax)

    for tau1 in range(1,3):
      for l1 in range(1,self.lmax+1):
        for m1 in range(-l1, l1+1):
          j1 = multi2single_index(0, tau1, l1, m1, self.lmax)
          for tau2 in range(1, 3):
            for l2 in range(1, self.lmax+1):
              for m2 in range(-l2, l2+1):
                j2 = multi2single_index(0, tau2, l2, m2, self.lmax)
                for p in range(0, 2*self.lmax+1):
                  if tau1 == tau2:
                    self.translation_ab5[j1,j2,p] = np.power(1j, abs(m1 - m2) - abs(m1) - abs(m2) + l2 - l1 + p) * np.power(-1.0, m1-m2) * \
                      np.sqrt((2 * l1 + 1) * (2 * l2 + 1) / (2 * l1 * (l1 + 1) * l2 * (l2 + 1))) * \
                      (l1 * (l1 + 1) + l2 * (l2 + 1) - p * (p + 1)) * np.sqrt(2 * p + 1) * \
                      wig.wig3jj(2 * l1, 2 * l2, 2 * p, 2 * m1, -2 * m2, 2 * (-m1+m2)) * wig.wig3jj(2 * l1, 2 * l2, 2 * p, 0, 0, 0)
                  elif p> 0:
                    self.translation_ab5[j1,j2,p] = np.power(1j, abs(m1 - m2) - abs(m1) - abs(m2) + l2 - l1 + p) * np.power(-1.0, m1-m2) * \
                      np.sqrt((2 * l1 + 1) * (2 * l2 + 1) / (2 * l1 * (l1 + 1) * l2 * (l2 + 1))) * \
                      np.lib.scimath.sqrt((l1 + l2 + 1 + p) * (l1 + l2 + 1 - p) * (p + l1 - l2) * (p - l1 + l2) * (2 * p + 1)) * \
                      wig.wig3jj(2 * l1, 2 * l2, 2 * p, 2 * m1, -2 * m2, 2 * (-m1+m2)) * wig.wig3jj(2 * l1, 2 * l2, 2 * (p-1), 0, 0, 0)

    wig.wig_table_free()
    wig.wig_temp_free()