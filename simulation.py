import logging

import numpy as np
from sympy.physics.wigner import wigner_3j as wigner_3j_sympy

from parameters import Parameters
from numerics import Numerics
from scipy.special import spherical_jn, spherical_yn

from functions.wigner3j import wigner3j
from functions.incidence.initial_field_coefficients_wavebundle_normal_incidence import initial_field_coefficients_wavebundle_normal_incidence
from functions.incidence.initial_field_coefficients_planewave import initial_field_coefficients_planewave


from functions.T_entry import T_entry

class Simulation:
  def __init__(self, parameters: Parameters, numerics: Numerics):
    self.parameters = parameters
    self.numerics = numerics

    self.__setup()
    self.log = logging.getLogger(__name__)

  @staticmethod
  def jmult_max(num_part, lmax):
    return 2 * num_part * lmax * (lmax + 2)

  @staticmethod
  def multi2single_index(jS,tau,l,m,lmax):
    return jS * 2 * lmax * (lmax+2) + (tau-1) * lmax * (lmax+2) + (l-1)*(l+1) + m + l
    # return (jS-1)*2*lmax*(lmax+2)+(tau-1)*lmax*(lmax+2)+(l-1)*(l+1)+m+l+1
  
  def __compute_lookup_particle_distances(self):
    # add two zeros at beginning to allow interpolation
    # also in the first segment
    step = self.numerics.particle_distance_resolution
    maxdist = self.parameters.particles.max_particle_distance + 3 * self.numerics.particle_distance_resolution
    self.lookup_particle_distances = np.concatenate((np.array([0]), np.arange(0, maxdist+1, step)))

  def __compute_h3_table(self):
    self.h3_table = np.zeros((2 * self.numerics.lmax + 1, self.lookup_particle_distances.shape[0]), dtype=complex)

    for p in range(2 * self.numerics.lmax + 1):
      self.h3_table[p, :] = spherical_jn(p, self.parameters.k_medium * self.lookup_particle_distances)

  def __setup(self):
    self.__compute_lookup_particle_distances()
    self.__compute_h3_table()

  def compute_mie_coefficients(self):
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

  def compute_translation_table(self):
    jmax = Simulation.jmult_max(1, self.numerics.lmax)
    self.translation_ab5 = np.zeros((jmax, jmax, 2 * self.numerics.lmax + 1), dtype=np.complex)

    for tau1 in range(1,3):
      for l1 in range(1,self.numerics.lmax+1):
        for m1 in range(-l1, l1+1):
          j1 = Simulation.multi2single_index(0, tau1, l1, m1, self.numerics.lmax)
          for tau2 in range(1, 3):
            for l2 in range(1, self.numerics.lmax+1):
              for m2 in range(-l2, l2+1):
                j2 = Simulation.multi2single_index(0, tau2, l2, m2, self.numerics.lmax)
                for p in range(0, 2*self.numerics.lmax+1):
                  if tau1 == tau2:
                    self.translation_ab5[j1,j2,p] = np.power(1j, abs(m1 - m2) - abs(m1) - abs(m2) + l2 - l1 + p) * np.power(-1.0, m1-m2) * \
                      np.sqrt((2 * l1 + 1) * (2 * l2 + 1) / (2 * l1 * (l1 + 1) * l2 * (l2 + 1))) * \
                      (l1 * (l1 + 1) + l2 * (l2 + 1) - p * (p + 1)) * np.sqrt(2 * p + 1) * \
                      wigner3j(l1, l2, p, m1, -m2, -m1+m2) * wigner3j(l1, l2, p, 0, 0, 0)
                  elif p> 0:
                    self.translation_ab5[j1,j2,p] = np.power(1j, abs(m1 - m2) - abs(m1) - abs(m2) + l2 - l1 + p) * np.power(-1.0, m1-m2) * \
                      np.sqrt((2 * l1 + 1) * (2 * l2 + 1) / (2 * l1 * (l1 + 1) * l2 * (l2 + 1))) * \
                      np.lib.scimath.sqrt((l1 + l2 + 1 + p) * (l1 + l2 + 1 - p) * (p + l1 - l2) * (p - l1 + l2) * (2 * p + 1)) * \
                      wigner3j(l1, l2, p, m1, -m2, -m1+m2) * wigner3j(l1, l2, p-1, 0, 0, 0)

  def compute_initial_field_coefficients(self):
    self.log.info('compute initial field coefficients ...')
    
    if np.isfinite(self.parameters.initial_field.beam_width) and (self.parameters.initial_field.beam_width > 0):
      self.log.info('  Gaussian beam ...')
      if self.parameters.initial_field.normal_incidence:
        self.initial_field_coefficients = initial_field_coefficients_wavebundle_normal_incidence(self)
      else:
        self.log.error('  this case is not implemented')
    else:
      self.log.info('  plane wave ...')
      self.initial_field_coefficients = initial_field_coefficients_planewave(self)
    
    self.log.info('done')


  #   function obj = computeInitialFieldCoefficients(obj)
  #     fprintf(1,'compute initial field coefficients ...');
  #     if isfinite(obj.input.initialField.beamWidth) && obj.input.initialField.beamWidth
  #         fprintf(1,' Gaussian beam ...');
  #         if obj.input.initialField.normalIncidence
  #             obj.tables.initialFieldCoefficients = ...
  #                 initial_field_coefficients_wavebundle_normal_incidence(obj);
  #         else
  #             error('this case is not implemented')
  #         end
  #     else % infinite or 0 beam width
  #         fprintf(1,' plane wave ...');
  #         obj.tables.initialFieldCoefficients = initial_field_coefficients_planewave(obj);
  #     end
  #     fprintf(1,' done\n');
  # end

  def compute_right_hand_side(self):
    self.right_hand_side = self.mie_coefficients[self.parameters.particles.single_unique_array_idx, :] * self.initial_field_coefficients