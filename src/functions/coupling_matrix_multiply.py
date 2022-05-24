from time import time
import logging
from math import ceil

import numpy as np
from numba import cuda
# from scipy.special import spherical_jn, spherical_yn, lpmv
# from scipy.spatial.distance import pdist, squareform

# from .misc import multi2single_index, jmult_max
# from .coupling_matrix.cpu.cffi.coupling_matrix.lib import coupling_matrix
# import ctypes
# import cffi
# from .coupling_matrix.cpu.cffi.coupling_matrix import lib
# from .coupling_matrix.coupling_matrix_cpu_cython import coupling_matrix_cpu_cython

from .coupling_matrix.cpu.parallel import compute_idx_lookups
from .coupling_matrix.cpu.parallel import particle_interaction
from .coupling_matrix.gpu.cuda import particle_interaction_gpu

def coupling_matrix_multiply(simulation, x: np.ndarray, idx: int=None):
  log = logging.getLogger('coupling_matrix_multiply_numba')
  log.info('prepare particle coupling ... ')
  preparation_time = time()

  lmax = simulation.numerics.lmax
  particle_number = simulation.parameters.particles.number
  wavelengths = simulation.parameters.k_medium.shape[0]
  translation_table = simulation.numerics.translation_ab5
  associated_legendre_lookup = simulation.plm
  spherical_hankel_lookup = simulation.sph_h
  e_j_dm_phi_loopup = simulation.e_j_dm_phi

  idx_lookup = np.empty((2 * lmax * (lmax + 2) * particle_number, 5), dtype=int)
  idx_lookup = compute_idx_lookups(lmax, particle_number, idx_lookup)


  
  log.info('Starting Wx computation')
  if simulation.numerics.gpu:
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

  return wx

# def coupling_matrix_multiply(simulation, x: np.ndarray, k_medium: complex=None):
#   log = logging.getLogger('coupling_matrix_multiply')
#   log.info('prepare particle coupling ... ')
#   preparation_time_start = time()

#   lmax = simulation.numerics.lmax
#   particle_number = simulation.parameters.particles.number
#   dists = squareform(pdist(simulation.parameters.particles.pos))
#   ct = np.divide(
#     np.subtract.outer(simulation.parameters.particles.pos[:,2], simulation.parameters.particles.pos[:,2]),
#     dists,
#     out=np.zeros((particle_number, particle_number)),
#     where=dists != 0)
#   phi = np.arctan2(
#     np.subtract.outer(simulation.parameters.particles.pos[:,1], simulation.parameters.particles.pos[:,1]),
#     np.subtract.outer(simulation.parameters.particles.pos[:,0], simulation.parameters.particles.pos[:,0]))
#   cos_dm_phi = np.zeros((4 * lmax + 1, particle_number, particle_number))
#   sin_dm_phi = np.zeros((4 * lmax + 1, particle_number, particle_number))
#   for p in range(4 * lmax + 1):
#     cos_dm_phi[p, :, :] = np.cos((p - 2 * lmax) * phi)
#     sin_dm_phi[p, :, :] = np.sin((p - 2 * lmax) * phi)
#   del phi
  
#   if k_medium == None:
#     size_param = dists * simulation.parameters.k_medium[0]
#   else:
#     size_param = dists * k_medium
#   spherical_hankel_lookup = np.zeros((2 * lmax + 1, particle_number, particle_number), dtype=complex)
#   associated_legendre_lookup = np.zeros(((lmax + 1) * (2 * lmax + 1), particle_number, particle_number))
#   for p in range(2 * lmax + 1):
#     spherical_hankel_lookup[p, :, :] = \
#       np.nan_to_num(spherical_jn(p, size_param), posinf=0, neginf=0) + 1j * \
#       np.nan_to_num(spherical_yn(p, size_param), posinf=0, neginf=0)
#     for absdm in range(p + 1):
#       cml = np.sqrt((2 * p + 1) / 2 * np.math.factorial(p - absdm) / np.math.factorial(p + absdm))
#       associated_legendre_lookup[p * (p + 1) // 2 + absdm, :, :] = cml * np.power(-1.0, absdm) * lpmv(absdm, p, ct)
#   real_spherical_hankel_lookup = np.copy(np.real(spherical_hankel_lookup))
#   imag_spherical_hankel_lookup = np.copy(np.imag(spherical_hankel_lookup))
#   del (dists, size_param, spherical_hankel_lookup)

#   real_ab5_table = np.copy(np.real(simulation.numerics.translation_ab5))
#   imag_ab5_table = np.copy(np.imag(simulation.numerics.translation_ab5))

#   real_x = np.copy(np.real(x))
#   imag_x = np.copy(np.imag(x))

#   real_wx = np.zeros_like(x, dtype=float)
#   imag_wx = np.zeros_like(x, dtype=float)


#   ffi = cffi.FFI()
#   coupling_matrix_start = time()
#   lib.coupling_matrix(particle_number, lmax, 
#     ffi.cast("double *", real_x.ctypes.data),                       ffi.cast("double *", imag_x.ctypes.data),
#     ffi.cast("double *", real_ab5_table.ctypes.data),               ffi.cast("double *", imag_ab5_table.ctypes.data),
#     ffi.cast("double *", associated_legendre_lookup.ctypes.data),
#     ffi.cast("double *", real_spherical_hankel_lookup.ctypes.data), ffi.cast("double *", imag_spherical_hankel_lookup.ctypes.data),
#     ffi.cast("double *", cos_dm_phi.ctypes.data),                   ffi.cast("double *", sin_dm_phi.ctypes.data),
#     ffi.cast("double *", real_wx.ctypes.data),                      ffi.cast("double *", imag_wx.ctypes.data))
#   preparation_time_stop = time()
#   print("Time taken: %f" % (preparation_time_stop - preparation_time_start))
#   print("Time taken for coupling matrix: %f" % (preparation_time_stop - coupling_matrix_start))

#   return real_wx + 1j * imag_wx

# def coupling_matrix_multiply_legacy(simulation, x: np.ndarray):
#   log = logging.getLogger('coupling_matrix_multiply')
#   log.info('prepare particle coupling ... ')
#   preparation_time_start = time()

#   lmax = simulation.numerics.lmax
#   particle_number = simulation.parameters.particles.number

#   real_ab5_table = np.empty(0, dtype=float)
#   imag_ab5_table = np.empty(0, dtype=float)

#   loop = 0
#   for tau1 in range(1,3):
#     for l1 in range(1, lmax+1):
#       for m1 in range(-l1, l1+1):
#         j1 = multi2single_index(0, tau1, l1, m1, lmax)
#         for tau2 in range(1, 3):
#           for l2 in range(1, lmax+1):
#             for m2 in range(-l2, l2+1):
#               j2 = multi2single_index(0, tau2, l2, m2, lmax)
#               for p in range(np.max([np.abs(m1-m2), np.abs(l1-l2) + np.abs(tau1-tau2)]), l1+l2+1):
#                 real_ab5_table = np.append(real_ab5_table, np.real(simulation.numerics.translation_ab5[j2, j1, p]))
#                 imag_ab5_table = np.append(imag_ab5_table, np.imag(simulation.numerics.translation_ab5[j2, j1, p]))
#                 loop += 1

#   real_wx = np.zeros_like(x, dtype=float)
#   imag_wx = np.zeros_like(x, dtype=float)


#   ffi = cffi.FFI()
  
#   particle_position_copy = np.copy(simulation.parameters.particles.pos)
#   particle_position = ffi.cast("double *", particle_position_copy.ctypes.data)
#   real_x = np.copy(np.real(x))
#   imag_x = np.copy(np.imag(x))
#   re_x = ffi.cast("double *", real_x.ctypes.data)
#   im_x = ffi.cast("double *", imag_x.ctypes.data)
#   re_ab5_table = ffi.cast("double *", real_ab5_table.ctypes.data)
#   im_ab5_table = ffi.cast("double *", imag_ab5_table.ctypes.data)

#   # Compute the associated legendre coefficients
#   simulation.numerics.compute_plm_coefficients()
#   plm_coeff_copy = np.copy(simulation.numerics.plm_coeff_table, order='F')
#   plm_coeff = ffi.cast("double *", plm_coeff_copy.ctypes.data)

#   # Prepare the lookup table for the hankel function
#   r_resol = simulation.numerics.particle_distance_resolution
#   real_spherical_hankel_lookup = np.copy(np.squeeze(np.nan_to_num(np.real(simulation.h3_table[:,:,0]))), order='F')
#   imag_spherical_hankel_lookup = np.copy(np.squeeze(np.nan_to_num(np.imag(simulation.h3_table[:,:,0]))), order='F')
#   re_sph_h_lookup = ffi.cast("double *", real_spherical_hankel_lookup.ctypes.data)
#   im_sph_h_lookup = ffi.cast("double *", imag_spherical_hankel_lookup.ctypes.data)

#   # Array for storing the data into
#   re_wx = ffi.cast("double *", real_wx.ctypes.data)
#   im_wx = ffi.cast("double *", imag_wx.ctypes.data)
#   coupling_matrix_start = time()
#   lib.coupling_matrix_legacy(particle_number, particle_position,
#     lmax, re_x, im_x,
#     re_ab5_table, im_ab5_table,
#     plm_coeff, r_resol,
#     re_sph_h_lookup, im_sph_h_lookup,
#     re_wx, im_wx)
#   preparation_time_stop = time()
#   print("Time taken: %f" % (preparation_time_stop - preparation_time_start))
#   print("Time taken for coupling matrix: %f" % (preparation_time_stop - coupling_matrix_start))

#   return real_wx + 1j * imag_wx

# def coupling_matrix_multiply_legacy_ab_free(simulation, x: np.ndarray):
#   log = logging.getLogger('coupling_matrix_multiply')
#   log.info('prepare particle coupling ... ')
#   preparation_time_start = time()

#   lmax = simulation.numerics.lmax
#   particle_number = simulation.parameters.particles.number

#   real_ab5_table = np.copy(np.real(simulation.numerics.translation_ab5), order='F')
#   imag_ab5_table = np.copy(np.imag(simulation.numerics.translation_ab5), order='F')

#   real_wx = np.zeros_like(x, dtype=float)
#   imag_wx = np.zeros_like(x, dtype=float)


#   ffi = cffi.FFI()
  
#   particle_position_copy = np.copy(simulation.parameters.particles.pos)
#   particle_position = ffi.cast("double *", particle_position_copy.ctypes.data)
#   real_x = np.copy(np.real(x))
#   imag_x = np.copy(np.imag(x))
#   re_x = ffi.cast("double *", real_x.ctypes.data)
#   im_x = ffi.cast("double *", imag_x.ctypes.data)
#   re_ab5_table = ffi.cast("double *", real_ab5_table.ctypes.data)
#   im_ab5_table = ffi.cast("double *", imag_ab5_table.ctypes.data)

#   # Compute the associated legendre coefficients
#   simulation.numerics.compute_plm_coefficients()
#   plm_coeff_copy = np.copy(simulation.numerics.plm_coeff_table, order='F')
#   plm_coeff = ffi.cast("double *", plm_coeff_copy.ctypes.data)

#   # Prepare the lookup table for the hankel function
#   r_resol = simulation.numerics.particle_distance_resolution
#   real_spherical_hankel_lookup = np.copy(np.squeeze(np.nan_to_num(np.real(simulation.h3_table[:,:,0]))), order='F')
#   imag_spherical_hankel_lookup = np.copy(np.squeeze(np.nan_to_num(np.imag(simulation.h3_table[:,:,0]))), order='F')
#   re_sph_h_lookup = ffi.cast("double *", real_spherical_hankel_lookup.ctypes.data)
#   im_sph_h_lookup = ffi.cast("double *", imag_spherical_hankel_lookup.ctypes.data)

#   # Array for storing the data into
#   re_wx = ffi.cast("double *", real_wx.ctypes.data)
#   im_wx = ffi.cast("double *", imag_wx.ctypes.data)
#   coupling_matrix_start = time()
#   lib.coupling_matrix_legacy_ab_free(particle_number, particle_position,
#     lmax, re_x, im_x,
#     re_ab5_table, im_ab5_table,
#     plm_coeff, r_resol,
#     re_sph_h_lookup, im_sph_h_lookup,
#     re_wx, im_wx)
#   preparation_time_stop = time()
#   print("Time taken: %f" % (preparation_time_stop - preparation_time_start))
#   print("Time taken for coupling matrix: %f" % (preparation_time_stop - coupling_matrix_start))

#   return real_wx + 1j * imag_wx