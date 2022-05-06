import logging

import numpy as np
from scipy.special import spherical_jn, spherical_yn
from scipy.special import jv, yv

from pyles.parameters import Parameters
from pyles.numerics import Numerics
from pyles.functions.spherical_functions_trigon import spherical_functions_trigon
from pyles.functions.t_entry import t_entry

from pyles.functions.misc import transformation_coefficients
from pyles.functions.misc import multi2single_index

class Simulation:
  """Pyles Simulation Class"""

  def __init__(self, parameters: Parameters, numerics: Numerics):
    self.parameters = parameters
    self.numerics = numerics

    self.__setup()
    self.log = logging.getLogger(__name__)
  
  def __compute_lookup_particle_distances(self):
    # add two zeros at beginning to allow interpolation
    # also in the first segment
    step = self.numerics.particle_distance_resolution
    maxdist = self.parameters.particles.max_particle_distance + 3 * self.numerics.particle_distance_resolution
    self.lookup_particle_distances = np.concatenate((np.array([0]), np.arange(0, maxdist + np.finfo(float).eps, step)))

  def __compute_h3_table(self):
    self.h3_table = np.zeros(
      (2 * self.numerics.lmax + 1, self.lookup_particle_distances.shape[0], self.parameters.medium_mefractive_index.shape[0]), 
      dtype=complex)
    size_param = np.outer(self.lookup_particle_distances, self.parameters.k_medium)

    for p in range(2 * self.numerics.lmax + 1):
      self.h3_table[p, :, :] = spherical_jn(p, size_param) + 1j * spherical_yn(p, size_param)

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
            jmult = multi2single_index(0, tau, l, m, self.numerics.lmax)
            self.mie_coefficients[u_i, jmult, :] = t_entry(tau=tau, l=l,
              k_medium = self.parameters.k_medium,
              k_sphere = self.parameters.omega * self.parameters.particles.unique_radius_index_pairs[u_i, 1],
              radius = np.real(self.parameters.particles.unique_radius_index_pairs[u_i, 0]))

  def compute_initial_field_coefficients(self):
    self.log.info('compute initial field coefficients ...')
    
    if np.isfinite(self.parameters.initial_field.beam_width) and (self.parameters.initial_field.beam_width > 0):
      self.log.info('  Gaussian beam ...')
      if self.parameters.initial_field.normal_incidence:
        pass
        # self.initial_field_coefficients = initial_field_coefficients_wavebundle_normal_incidence(self)
      else:
        self.log.error('  this case is not implemented')
    else:
      self.log.info('  plane wave ...')
      self.__compute_initial_field_coefficients_planewave()
    
    self.log.info('done')

  def compute_right_hand_side(self):
    self.right_hand_side = self.mie_coefficients[self.parameters.particles.single_unique_array_idx, :] * self.initial_field_coefficients

  def __compute_initial_field_coefficients_planewave(self):
    lmax = self.numerics.lmax
    E0 = self.parameters.initial_field.amplitude
    k = self.parameters.k_medium

    beta = self.parameters.initial_field.polar_angle
    cb = np.cos(beta)
    sb = np.sin(beta)
    alpha = self.parameters.initial_field.azimuthal_angle

    # pi and tau symbols for transformation matrix B_dagger
    pilm,taulm = spherical_functions_trigon(beta,lmax)

    # cylindrical coordinates for relative particle positions
    relative_particle_positions = self.parameters.particles.pos - self.parameters.initial_field.focal_point
    kvec = np.outer(np.array((sb * np.cos(alpha), sb * np.sin(alpha), cb)), k)
    eikr = np.exp(1j * np.matmul(relative_particle_positions, kvec))

    # clean up some memory?
    del (k, beta, cb, sb, kvec, relative_particle_positions)

    self.initial_field_coefficients = np.zeros((self.parameters.particles.number, self.numerics.nmax, self.parameters.k_medium.shape[0]), dtype=complex)
    for m in range(-lmax, lmax+1):
      for tau in range(1, 3):
        for l in range(np.max([1, np.abs(m)]), lmax+1):
          n = multi2single_index(0, tau, l, m, lmax)
          self.initial_field_coefficients[:,n,:] = 4 * E0 * np.exp(-1j * m * alpha) * eikr * transformation_coefficients(pilm, taulm, tau, l, m, self.parameters.initial_field.pol, dagger=True)



# function aI = initial_field_coefficients_planewave(simulation)

# lmax = simulation.numerics.lmax;
# E0 = simulation.input.initialField.amplitude;
# k = simulation.input.k_medium;

# beta = simulation.input.initialField.polarAngle;
# cb = cos(beta);
# sb = sin(beta);
# alpha = simulation.input.initialField.azimuthalAngle;

# % pi and tau symbols for transformation matrix B_dagger
# [pilm,taulm] = spherical_functions_trigon(cb,sb,lmax);  % Nk x 1

# % cylindrical coordinates for relative particle positions
# relativeParticlePositions = simulation.input.particles.positionArray - simulation.input.initialField.focalPoint;
# kvec = k*[sb*cos(alpha);sb*sin(alpha);cb];
# eikr = exp(1i*relativeParticlePositions*kvec);

# clear k beta cb sb kvec relativeParticlePositions % clean up some memory?

# % compute initial field coefficients
# aI = simulation.numerics.deviceArray(zeros(simulation.input.particles.number,simulation.numerics.nmax,'single'));
# for m=-lmax:lmax
#     for tau=1:2
#         for l=max(1,abs(m)):lmax
#             n=multi2single_index(1,tau,l,m,lmax);
#             aI(:,n) = 4 * E0 * exp(-1i*m*alpha) .* eikr .* transformation_coefficients(pilm,taulm,tau,l,m,simulation.input.initialField.pol,'dagger');
#         end
#     end
# end
# end