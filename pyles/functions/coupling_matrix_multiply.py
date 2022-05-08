from time import time
import logging

import numpy as np
from scipy.special import spherical_jn, spherical_yn, lpmv

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
              for p in range(np.max([np.abs(m1-m2), np.abs(l1-l2) + np.abs(tau1-tau2)]), l1+l2+1):
                real_ab5_table = np.append(real_ab5_table, np.real(simulation.numerics.translation_ab5[j2, j1, p]))
                imag_ab5_table = np.append(imag_ab5_table, np.imag(simulation.numerics.translation_ab5[j2, j1, p]))

  ab5_table = real_ab5_table + 1j * imag_ab5_table

  real_Wx, imag_Wx = coupling_matrix_multiply_cpu(\
    lmax, simulation.parameters.particles.pos, x, simulation.h3_table, \
    ab5_table, simulation.numerics.particle_distance_resolution)

def coupling_matrix_multiply_cpu(lmax, position, value, h3_table, ab5_table, resolution):
  for s1 in range(position.shape[0]):

    for s2 in range(position.shape[0]):
      if s1 == s2:
        continue

      r = np.sqrt(np.sum(np.power(position[s1,:] - position[s2,:], 2)))
      ct = (position[s1,2] - position[s2,2]) / r
      st = np.sqrt(1 - ct * ct)
      phi = np.arctan2(position[s1,1] - position[s2,1], position[s1,0] - position[s2,0])

      re_h = np.zeros(2 * lmax + 1, dtype=complex)
      im_h = np.zeros(2 * lmax + 1, dtype=complex)
      h = np.zeros(2 * lmax + 1, dtype=complex)
      p_pdm = np.zeros((2 * lmax + 1, 2 * lmax + 1)) * np.nan

      for p in range(2 * lmax + 1):
        re_h[p] = spherical_jn(p, r)
        im_h[p] = spherical_yn(p, r)
        h[p] = spherical_jn(p, r) + 1j * spherical_yn(p, r)
        for absdm in range(p + 1):
          p_pdm[p, absdm] = lpmv(absdm, p, ct)

      w_x = np.zeros_like(value, dtype=complex)

      loop_counter = 0
      for tau1 in range(1, 3):
        for l1 in range(1, lmax+1):
          for m1 in range(-l1, l1+1):
            # n1 = multi2single_index()
            for tau2 in range(1, 3):
              for l2 in range(1, lmax+1):
                for m2 in range(-l1, l1+1):
                  #
                  for p in range(np.max([np.abs(m1-m2), np.abs(l1-l2) + np.abs(tau1-tau2)]), l1+l2+1):
                    ab_p = ab5_table[loop_counter] * p_pdm[p, np.abs(m1-m2)]
                    ab_p_h = ab_p * h[p]
                    ab_p_h_eimp = ab_p_h * np.exp(1j * (m2 - m1) * phi)




  return None, None