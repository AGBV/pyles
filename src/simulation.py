import logging
from time import time

import numpy as np
from numba import cuda
from math import ceil
from scipy.sparse.linalg import LinearOperator
from scipy.spatial.distance import pdist, squareform
from scipy.special import spherical_jn, spherical_yn
from scipy.special import hankel1
from scipy.special import lpmv

from src.parameters import Parameters
from src.numerics import Numerics
from src.functions.spherical_functions_trigon import spherical_functions_trigon
from src.functions.t_entry import t_entry
from src.functions.coupling_matrix_multiply import coupling_matrix_multiply

from src.functions.misc import transformation_coefficients
from src.functions.misc import multi2single_index

# from scipy.special import spherical_jn, spherical_yn, lpmv
# from scipy.spatial.distance import pdist, squareform

# from .misc import multi2single_index, jmult_max
# from .coupling_matrix.cpu.cffi.coupling_matrix.lib import coupling_matrix
# import ctypes
# import cffi
# from .coupling_matrix.cpu.cffi.coupling_matrix import lib
# from .coupling_matrix.coupling_matrix_cpu_cython import coupling_matrix_cpu_cython

from src.functions.coupling_matrix.cpu.parallel import compute_idx_lookups
from src.functions.coupling_matrix.cpu.parallel import particle_interaction
from src.functions.coupling_matrix.gpu.cuda import particle_interaction_gpu

class Simulation:
  """Pyles Simulation Class"""

  def __init__(self, parameters: Parameters, numerics: Numerics):
    self.parameters = parameters
    self.numerics = numerics

    self.log = logging.getLogger(__name__)
    self.__setup()
  
  def legacy_compute_lookup_particle_distances(self):
    # add two zeros at beginning to allow interpolation
    # also in the first segment
    step = self.numerics.particle_distance_resolution
    maxdist = self.parameters.particles.max_particle_distance + 3 * self.numerics.particle_distance_resolution
    self.lookup_particle_distances = np.concatenate((np.array([0]), np.arange(0, maxdist + np.finfo(float).eps, step)))

  def legacy_compute_h3_table(self):
    self.h3_table = np.zeros(
      (2 * self.numerics.lmax + 1, self.lookup_particle_distances.shape[0], self.parameters.medium_mefractive_index.shape[0]), 
      dtype=complex)
    size_param = np.outer(self.lookup_particle_distances, self.parameters.k_medium)

    for p in range(2 * self.numerics.lmax + 1):
      self.h3_table[p, :, :] = spherical_jn(p, size_param) + 1j * spherical_yn(p, size_param)
      # self.h3_table[p, :, :] = np.sqrt(np.pi / size_param) * 2 /hankel1(p, size_param)

  def __compute_lookups(self):
    lookup_computation_time_start = time()
    lmax = self.numerics.lmax
    particle_number = self.parameters.particles.number

    dists = squareform(pdist(self.parameters.particles.pos))
    ct = np.divide(
      np.subtract.outer(self.parameters.particles.pos[:,2], self.parameters.particles.pos[:,2]),
      dists,
      out=np.zeros((particle_number, particle_number)),
      where=dists != 0)
    phi = np.arctan2(
      np.subtract.outer(self.parameters.particles.pos[:,1], self.parameters.particles.pos[:,1]),
      np.subtract.outer(self.parameters.particles.pos[:,0], self.parameters.particles.pos[:,0]))

    size_param = np.outer(dists.ravel(), self.parameters.k_medium).reshape([particle_number, particle_number, self.parameters.k_medium.shape[0]])

    self.sph_h = np.zeros((2 * lmax + 1, particle_number, particle_number, self.parameters.k_medium.shape[0]), dtype=complex)
    self.e_j_dm_phi = np.zeros((4 * lmax + 1, particle_number, particle_number), dtype=complex)
    self.plm = np.zeros(((lmax + 1) * (2 * lmax + 1), particle_number, particle_number))

    for p in range(2 * lmax + 1):
      # self.sph_h[p, :, :, :] = spherical_jn(p, size_param) + 1j * spherical_yn(p, size_param)
      self.sph_h[p, :, :] = np.sqrt(np.divide(np.pi / 2, size_param, out=np.zeros_like(size_param), where=size_param != 0)) * hankel1(p + 1/2, size_param)
      self.e_j_dm_phi[p, :, :]            = np.exp(1j * (p - 2 * lmax) * phi)
      self.e_j_dm_phi[p + 2 * lmax, :, :] = np.exp(1j * p * phi)
      for absdm in range(p + 1):
        cml = np.sqrt((2 * p + 1) / 2 / np.prod(np.arange(p - absdm + 1, p + absdm + 1)))
        self.plm[p * (p + 1) // 2 + absdm, :, :] = cml * np.power(-1.0, absdm) * lpmv(absdm, p, ct)
    
    lookup_computation_time_stop = time()
    self.log.info('Computing lookup tables took %f s' % (lookup_computation_time_stop - lookup_computation_time_start))

  def __setup(self):
    # self.__compute_lookup_particle_distances()
    # self.__compute_h3_table()
    self.__compute_lookups()
    # self.__compute_right_hand_side()

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
        self.__compute_initial_field_coefficients_wavebundle_normal_incidence()
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

    self.initial_field_coefficients = np.zeros((self.parameters.particles.number, self.numerics.nmax, self.parameters.k_medium.size), dtype=complex)
    for m in range(-lmax, lmax+1):
      for tau in range(1, 3):
        for l in range(np.max([1, np.abs(m)]), lmax+1):
          n = multi2single_index(0, tau, l, m, lmax)
          self.initial_field_coefficients[:,n,:] = 4 * E0 * np.exp(-1j * m * alpha) * eikr * transformation_coefficients(pilm, taulm, tau, l, m, self.parameters.initial_field.pol, dagger=True)

  def __compute_initial_field_coefficients_wavebundle_normal_incidence(self):
    # TODO
    # https://github.com/disordered-photonics/celes/blob/master/src/initial/initial_field_coefficients_wavebundle_normal_incidence.m
    self.initial_field_coefficients = None

  def coupling_matrix_multiply(self, x: np.ndarray, idx: int=None):
    log = logging.getLogger('coupling_matrix_multiply_numba')
    log.info('prepare particle coupling ... ')
    preparation_time = time()

    lmax = self.numerics.lmax
    particle_number = self.parameters.particles.number
    wavelengths = self.parameters.k_medium.shape[0]
    translation_table = self.numerics.translation_ab5
    associated_legendre_lookup = self.plm
    spherical_hankel_lookup = self.sph_h
    e_j_dm_phi_loopup = self.e_j_dm_phi

    idx_lookup = np.empty((2 * lmax * (lmax + 2) * particle_number, 5), dtype=int)
    idx_lookup = compute_idx_lookups(lmax, particle_number, idx_lookup)

    if idx != None:
      spherical_hankel_lookup = spherical_hankel_lookup[:,:,:,idx]
      spherical_hankel_lookup = spherical_hankel_lookup[:,:,:,np.newaxis]
      wavelengths = 1

    
    log.info('Starting Wx computation')
    if self.numerics.gpu:
      idx_device                  = cuda.to_device(idx_lookup)
      x_device                    = cuda.to_device(x)
      wx_real_device              = cuda.to_device(wx_real)
      wx_imag_device              = cuda.to_device(wx_imag)
      translation_device          = cuda.to_device(translation_table)
      associated_legendre_device  = cuda.to_device(associated_legendre_lookup)
      spherical_hankel_device     = cuda.to_device(spherical_hankel_lookup)
      e_j_dm_phi_device           = cuda.to_device(e_j_dm_phi_loopup)


      jmax = particle_number * 2 * lmax * (lmax + 2)
      threads_per_block = (16, 16, 2)
      blocks_per_grid_x = ceil(jmax         / threads_per_block[0])
      blocks_per_grid_y = ceil(jmax         / threads_per_block[1])
      blocks_per_grid_z = ceil(wavelengths  / threads_per_block[2])
      blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)
      
      wx_real = np.zeros(x.shape + (wavelengths,), dtype=float)
      wx_imag = np.zeros_like(wx_real)

      coupling_matrix_time = time()
      particle_interaction_gpu[blocks_per_grid,threads_per_block](lmax, particle_number,
        idx_device, x_device, wx_real_device, wx_imag_device,
        translation_device, associated_legendre_device, 
        spherical_hankel_device, e_j_dm_phi_device)
      wx_real = wx_real_device.copy_to_host()
      wx_imag = wx_imag_device.copy_to_host()
      wx = wx_real + 1j * wx_imag
      # particle_interaction.parallel_diagnostics(level=4)
      time_end = time()
      log.info("Time taken for preparation: %f" % (coupling_matrix_time - preparation_time))
      log.info("Time taken for coupling matrix: %f" % (time_end - coupling_matrix_time))
    else:
      wx = particle_interaction(lmax, particle_number,
        idx_lookup, x,
        translation_table, associated_legendre_lookup, 
        spherical_hankel_lookup, e_j_dm_phi_loopup)
      time_end = time()
      log.info("Time taken for coupling matrix: %f" % (time_end - preparation_time))

    if idx != None:
      wx = np.squeeze(wx)

    return wx

  def master_matrix_multiply(self, value: np.ndarray, idx: int):
    wx = self.coupling_matrix_multiply(value, idx)

    self.log.info('apply T-matrix ...')
    t_matrix_start = time()

    wx = wx.reshape((self.parameters.particles.number, self.numerics.nmax))
    twx = self.mie_coefficients[self.parameters.particles.single_unique_array_idx, :, idx] * wx
    mx = value - twx.ravel()

    t_matrix_stop = time()
    self.log.info(' done in %f seconds.' % (t_matrix_stop - t_matrix_start))

    return mx

  def compute_scattered_field_coefficients(self):
    self.log.info('compute scattered field coefficients ...')
    jmax = self.parameters.particles.number * self.numerics.nmax
    self.scattered_field_coefficients = np.zeros_like(self.initial_field_coefficients)
    for w in range(self.parameters.wavelengths_number):
      print(w)
      mmm = lambda x: self.master_matrix_multiply(x, w)
      A = LinearOperator(shape=(jmax, jmax), matvec=mmm)
      # b = self.right_hand_side[:,:,w]
      b = self.right_hand_side[:,:,w].ravel()
      x = self.numerics.solver.run(A, b)
      self.scattered_field_coefficients[:,:,w] = x.reshape(b.shape)