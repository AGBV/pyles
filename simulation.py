from unicodedata import numeric
import numpy as np
from sympy import numer

from parameters import Parameters
from numerics import Numerics

from functions.T_entry import T_entry

class Simulation:
  def __init__(self, parameters: Parameters, numerics: Numerics):
    self.parameters = parameters
    self.numerics = numerics

  @staticmethod
  def jmult_max(num_part, lmax):
    return 2 * num_part * lmax * (lmax + 2)

  @staticmethod
  def multi2single_index(jS,tau,l,m,lmax):
    return jS * 2 * lmax * (lmax+2) + (tau-1) * lmax * (lmax+2) + (l-1)*(l+1) + m + l
    # return (jS-1)*2*lmax*(lmax+2)+(tau-1)*lmax*(lmax+2)+(l-1)*(l+1)+m+l+1

  def compute_mie_coefficients(self):
    print(self.parameters.wavelength.shape)
    self.mie_coefficients = np.zeros(
      (self.parameters.particles.num_unique_pairs,
      self.numerics.nmax,
      self.parameters.wavelength.shape[0]),
      dtype=complex)

    for u_i in range(self.parameters.particles.num_unique_pairs):
      for tau in range(1, 3):
        for l in range(1, self.numerics.lmax+1):
          for m in range(-l,l+1):
            jmult = self.multi2single_index(0, tau, l, m, self.numerics.lmax)
            self.mie_coefficients[u_i, jmult, :] = T_entry(tau=tau, l=l,
              kM = self.parameters.k_medium,
              kS = self.parameters.omega * self.parameters.particles.unique_radius_index_pairs[u_i, 1],
              R = np.real(self.parameters.particles.unique_radius_index_pairs[u_i, 0]))
            # print(t)