from calendar import c
import numpy as np
from numba import jit, prange, complex128

@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def compute_scattering_cross_section(lmax: int, particle_number: int, idx: np.ndarray, sfc: np.ndarray, translation_table: np.ndarray, plm: np.ndarray, sph_h: np.ndarray, e_j_dm_phi, progress_proxy=None):
  
  jmax = particle_number * 2 * lmax * (lmax + 2)
  wavelengths = sph_h.shape[-1]

  c_sca_complex = np.zeros(wavelengths, dtype=complex128)

  for w_idx in prange(jmax * jmax * wavelengths):
    w     = w_idx %  wavelengths
    j_idx = w_idx // wavelengths
    j1 = j_idx // jmax
    j2 = j_idx %  jmax
    s1, n1, _, _, m1 = idx[j1, :]
    s2, n2, _, _, m2 = idx[j2, :]

    delta_m = np.absolute(m1 - m2);

    f = sfc[:,:,w]

    val = 0j
    for p in range(2 * lmax + 1):
      if delta_m > p:
        continue

      val += (translation_table[n2, n1, p] *
        plm[p * (p + 1) // 2 + delta_m, s1, s2] * 
        sph_h[p, s1, s2, w])
    val *= np.conj(f[s1, n1]) * e_j_dm_phi[m2 - m1 + 2 * lmax, s1, s2] * f[s2, n2]

    c_sca_complex[w] += val

    if progress_proxy is not None:
      progress_proxy.update(1)

  return c_sca_complex

@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def compute_radial_independent_scattered_field(lmax: int, particles_position: np.ndarray, idx: np.ndarray, sfc: np.ndarray, k_medium: np.ndarray, azimuthal_angles: np.ndarray, e_r: np.ndarray, e_phi: np.ndarray, e_theta: np.ndarray, pilm: np.ndarray, taulm: np.ndarray, progress_proxy=None):
  
  e_1_sca = np.zeros(azimuthal_angles.size * 3 * k_medium.size, dtype=complex128).reshape((azimuthal_angles.size, 3, k_medium.size))
  jmax = particles_position.shape[0] * 2 * lmax * (lmax + 2)

  for global_idx in prange(jmax * azimuthal_angles.size * k_medium.size):
    w_idx = global_idx %  (jmax * k_medium.size)
    g_idx = global_idx // (jmax * k_medium.size)

    a = g_idx

    w     = w_idx %  k_medium.size
    j_idx = w_idx // k_medium.size
    s, n, tau, l, m = idx[j_idx, :]
    
    t = (
      np.power(1j, tau-l-2) * sfc[s, n, w] / np.sqrt(2 * l * (l+1)) * 
      np.exp(1j * (m * azimuthal_angles[a] - k_medium[w] * np.sum(particles_position[s, :] * e_r[a, :]))))

    if tau == 1:
      e_1_sca[a, :, w] += t * (e_theta[a, :] * pilm[l,  np.abs(m), a] * 1j * m - e_phi[a, :]   * taulm[l, np.abs(m), a])
    else:
      e_1_sca[a, :, w] += t * (e_phi[a, :]   * pilm[l,  np.abs(m), a] * 1j * m + e_theta[a, :] * taulm[l, np.abs(m), a])

    

    if progress_proxy != None:
      progress_proxy.update(1)

  return e_1_sca