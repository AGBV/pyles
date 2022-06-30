import numpy as np
from numba import cuda
from cmath import exp, sqrt


@cuda.jit(fastmath=True)
def particle_interaction_gpu(lmax: int, particle_number: int, idx: np.ndarray, x: np.ndarray, wx_real: np.ndarray, wx_imag: np.ndarray, translation_table: np.ndarray, plm: np.ndarray, sph_h: np.ndarray, e_j_dm_phi):

    j1, j2, w = cuda.grid(3)

    jmax = particle_number * 2 * lmax * (lmax + 2)
    channels = sph_h.shape[-1]

    if (j1 >= jmax) or (j2 >= jmax) or (w >= channels):
        return

    s1, n1, tau1, l1, m1 = idx[j1, :]
    s2, n2, tau2, l2, m2 = idx[j2, :]

    if s1 == s2:
        return

    delta_tau = abs(tau1 - tau2)
    delta_l = abs(l1 - l2)
    delta_m = abs(m1 - m2)

    p_dependent = complex(0)
    for p in range(max(delta_m, delta_l + delta_tau), l1 + l2 + 1):
        p_dependent += translation_table[n2, n1, p] * \
            plm[p * (p + 1) // 2 + delta_m, s1, s2] * \
            sph_h[p, s1, s2, w]
    p_dependent *= e_j_dm_phi[m2 - m1 + 2 * lmax, s1, s2] * x[j2]

    # atomic.add performs the += operation in sync
    cuda.atomic.add(wx_real, (j1, w), p_dependent.real)
    cuda.atomic.add(wx_imag, (j1, w), p_dependent.imag)


@cuda.jit(fastmath=True)
def compute_scattering_cross_section_gpu(lmax: int, particle_number: int, idx: np.ndarray, sfc: np.ndarray, translation_table: np.ndarray, plm: np.ndarray, sph_h: np.ndarray, e_j_dm_phi: np.ndarray, c_sca_real: np.ndarray, c_sca_imag: np.ndarray):

    jmax = particle_number * 2 * lmax * (lmax + 2)
    channels = sph_h.shape[-1]

    j1, j2, w = cuda.grid(3)

    if (j1 >= jmax) or (j2 >= jmax) or (w >= channels):
        return

    s1, n1, _, _, m1 = idx[j1, :]
    s2, n2, _, _, m2 = idx[j2, :]

    delta_m = abs(m1 - m2)
    f = sfc[:, :, w]

    p_dependent = complex(0)
    for p in range(delta_m, 2 * lmax + 1):
        p_dependent += (translation_table[n2, n1, p] *
                        plm[p * (p + 1) // 2 + delta_m, s1, s2] *
                        sph_h[p, s1, s2, w])
    p_dependent *= f[s1, n1].conjugate() * e_j_dm_phi[m2 -
                                                      m1 + 2 * lmax, s1, s2] * f[s2, n2]

    # atomic.add performs the += operation in sync
    cuda.atomic.add(c_sca_real, w, p_dependent.real)
    cuda.atomic.add(c_sca_imag, w, p_dependent.imag)


@cuda.jit(fastmath=True)
def compute_radial_independent_scattered_field_gpu(lmax: int, particles_position: np.ndarray, idx: np.ndarray, sfc: np.ndarray, k_medium: np.ndarray, azimuthal_angles: np.ndarray, e_r: np.ndarray, e_phi: np.ndarray, e_theta: np.ndarray, pilm: np.ndarray, taulm: np.ndarray, e_1_sca_real: np.ndarray, e_1_sca_imag: np.ndarray):

    j_idx, a_idx, w_idx = cuda.grid(3)

    jmax = particles_position.shape[0] * 2 * lmax * (lmax + 2)

    if (j_idx >= jmax) or (a_idx >= azimuthal_angles.size) or (w_idx >= k_medium.size):
        return

    s, n, tau, l, m = idx[j_idx, :]

    # Temp variable
    t = (
        1j**(tau-l-2) * sfc[s, n, w_idx] / sqrt(2 * l * (l+1)) *
        exp(1j * (m * azimuthal_angles[a_idx] - k_medium[w_idx] * (
            particles_position[s, 0] * e_r[a_idx, 0] +
            particles_position[s, 1] * e_r[a_idx, 1] +
            particles_position[s, 2] * e_r[a_idx, 2])))
    )

    for c in range(3):
        if tau == 1:
            e_1_sca = t * (e_theta[a_idx, c] * pilm[l,  abs(m), a_idx]
                           * 1j * m - e_phi[a_idx, c] * taulm[l, abs(m), a_idx])
        else:
            e_1_sca = t * (e_phi[a_idx, c] * pilm[l,  abs(m), a_idx]
                           * 1j * m + e_theta[a_idx, c] * taulm[l, abs(m), a_idx])

        cuda.atomic.add(e_1_sca_real, (a_idx, c, w_idx), e_1_sca.real)
        cuda.atomic.add(e_1_sca_imag, (a_idx, c, w_idx), e_1_sca.real)
