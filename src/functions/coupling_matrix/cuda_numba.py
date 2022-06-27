import numpy as np
import numba as nb
from numba import cuda

@cuda.jit(fastmath=True)
def particle_interaction_gpu(lmax: int, particle_number: int, idx: np.ndarray, x: np.ndarray, wx_real: np.ndarray, wx_imag: np.ndarray, translation_table: np.ndarray, plm: np.ndarray, sph_h: np.ndarray, e_j_dm_phi):

  j1, j2, w = cuda.grid(3)

  jmax = particle_number * 2 * lmax * (lmax + 2)
  wavelengths = sph_h.shape[-1]

  if (j1 >= jmax) or (j2 >= jmax) or (w >= wavelengths):
    return

  s1, n1, tau1, l1, m1 = idx[j1, :]
  s2, n2, tau2, l2, m2 = idx[j2, :]

  if s1 == s2:
    return

  delta_tau = abs(tau1 - tau2);
  delta_l   = abs(l1   - l2);
  delta_m   = abs(m1   - m2);

  # p_dependent = complex(0)
  # for p in range(max(delta_m, delta_l + delta_tau), l1 + l2 + 1):
  #   p_dependent += translation_table[n2, n1, p] * \
  #       plm[p * (p + 1) // 2 + delta_m, s1, s2] * \
  #       sph_h[p, s1, s2, w]
  # p_dependent *= e_j_dm_phi[m2 - m1 + 2 * lmax, s1, s2] * x[j2]

  p_dependent = complex(0)
  for p in range(max(delta_m, delta_l + delta_tau), l1 + l2 + 1):
    p_dependent += translation_table[n2, n1, p] * \
        plm[p * (p + 1) // 2 + delta_m, s1, s2] * \
        sph_h[p, s1, s2, w]
  p_dependent *= e_j_dm_phi[m2 - m1 + 2 * lmax, s1, s2] * x[j2]

  # atomic.add performs the += operation in sync
  cuda.atomic.add(wx_real, (j1, w), p_dependent.real)
  cuda.atomic.add(wx_imag, (j1, w), p_dependent.imag)
          