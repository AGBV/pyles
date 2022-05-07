from time import time
import logging

import numpy as np
from .misc import multi2single_index

def coupling_matrix_multiply(simulation, x: np.array):
  log = logging.getLogger('coupling_matrix_multiply')
  log.info('prepare particle coupling ... ')
  preparation_timer = time()

  lmax = simulation.numerics.lmax
  ns = simulation.parameters.particles.number
  # plm_coeff_table = simulation.numerics.plm_coeff_table

  real_hTab = np.real(simulation.h3_table)
  imag_hTab = np.imag(simulation.h3_table)
  r_resol = simulation.numerics.particle_distance_resolution

  real_x = np.real(x)
  imag_x = np.imag(x)

  real_ab5_table = np.empty(0, dtype=float)
  imag_ab5_table = np.empty(0, dtype=float)

  for tau1 in range(1,3):
    for l1 in range(1, lmax+1):
      for m1 in range(-l1, l1+1):
        j1 = multi2single_index(0, tau1, l1, m1, lmax)
        for tau2 in range(1, 3):
          for l2 in range(1, lmax+1):
            for m2 in range(-l2, l2+1):
              j2 = multi2single_index(0, tau2, l2, m2, lmax)
              for p in range(np.max([np.abs(m1-m2), np.abs(l1-l2) + np.abs(tau1-tau2)]), (l1+l2+1)):
                real_ab5_table = np.append(real_ab5_table, np.real(simulation.numerics.translation_ab5[j2, j1, p]))
                imag_ab5_table = np.append(imag_ab5_table, np.imag(simulation.numerics.translation_ab5[j2, j1, p]))

  real_Wx, imag_Wx = coupling_matrix_multiply_cpu(\
    simulation.parameters.particles.pos, x, simulation.h3_table, \
    real_ab5_table, imag_ab5_table, simulation.numerics.particle_distance_resolution)

def coupling_matrix_multiply_cpu(position, value, h3_table, real_ab5, imag_ab5, resolution):
  for s1 in range(position.shape[0]):
    for s2 in range(position.shape[0]):
      if s1 == s2:
        continue

      r = np.sqrt(np.sum(np.power(position[s1,:] - position[s2,:], 2)))
      ct = (position[s1,2] - position[s2,2]) / r
      st = np.sqrt(1 - ct * ct)
      phi = np.arctan2(position[s1,1] - position[s2,1], position[s1,0] - position[s2,0])

  return None, None