import numpy as np

from particles import Particles

class Pyles:
  def __init__(self, particles: Particles):
    self.particles = particles

  @staticmethod
  def __jmult_max(num_part, lmax):
    return 2 * num_part * lmax * (lmax + 2)


  def compute_mie_coefficients(self):
    mie_coefficients = np.zeros(self.particles.num_unique_pairs, )
    print('Compute Mie')