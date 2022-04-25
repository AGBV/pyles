from time import time
import logging

import numpy as np
from simulation import Simulation

def coupling_matrix_multiply(simulation: Simulation, x: np.array):
  log = logging.getLogger('coupling_matrix_multiply')
  log.info('prepare particle coupling ... ')
  preparation_timer = time()

  lmax = simulation.numerics.lmax
  ns = simulation.parameters.particles.number
  plmCoeffTable = simulation.numerics.plm_coeff_table

  real_hTab = np.real(simulation.h3_table)
  imag_hTab = np.imag(simulation.h3_table)
  r_resol = simulation.numerics.particle_distance_resolution

  real_x = np.real(x)
  imag_x = np.imag(x)

  real_ab5_table = np.array([])
  imag_ab5_table = np.array([])

  loop_counter = 1;
  for tau1 in range(1,3):
    for l1 in range(1, lmax+1):
      for m1 in range(-l1, l1+1):
        j1 = Simulation.multi2single_index(1, tau1, l1, m1, lmax)
        for tau2 in range(1, 3):
          for l2 in range(1, lmax+1):
            for m2 in range(-l2, l2+1):
              j2 = Simulation.multi2single_index(1, tau2, l2, m2, lmax)
              for p in range(np.max(np.abs(m1-m2), np.abs(l1-l2) + np.abs(tau1-tau2)), (l1+l2+1)):
                real_ab5_table = np.concatenate(np.real(simulation.translation_ab5(j2, j1, p)), real_ab5_table)
                imag_ab5_table = np.concatenate(np.imag(simulation.translation_ab5(j2, j1, p)), imag_ab5_table)
                loop_counter += 1