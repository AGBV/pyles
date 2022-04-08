import numpy as np

class Pyles:
  def __init__(self, particles):
    self.particles = particles

  def compute_mie_coefficients(self):
    mie_coefficients = np.zeros()
    print('Compute Mie')