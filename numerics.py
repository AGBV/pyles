import numpy as np

class Numerics:
  def __init__(self, lmax, polar_angles, azimuth_angles, gpu=False):
    self.lmax = lmax
    self.polar_angles = polar_angles
    self.azumuth_angles = azimuth_angles
    self.gpu = gpu

    self.nmax = Numerics.compute_nmax(lmax)

  @staticmethod
  def compute_nmax(lmax):
    return 2 * lmax * (lmax + 2)

  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.lpmn.html
  def plm_coefficients(self):
    self.plm_coeff_table = np.zeros(
      2 * self.lmax + 1,
      2 * self.lmax + 1,
      np.ceil(self.lmax))
