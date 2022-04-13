import numpy as np
from sympy.physics.wigner import wigner_3j

from parameters import Parameters
from numerics import Numerics

from functions.T_entry import T_entry

class Simulation:
  def __init__(self, parameters: Parameters, numerics: Numerics):
    self.parameters = parameters
    self.numerics = numerics

  @staticmethod
  def jmult_max(num_part, lmax):
    return 2 * num_part * lmax * (lmax + 2)

  @staticmethod
  def multi2single_index(jS,tau,l,m,lmax):
    return jS * 2 * lmax * (lmax+2) + (tau-1) * lmax * (lmax+2) + (l-1)*(l+1) + m + l
    # return (jS-1)*2*lmax*(lmax+2)+(tau-1)*lmax*(lmax+2)+(l-1)*(l+1)+m+l+1

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
                  # print(j1,j2,p)
                  if tau1==tau2:
                    self.translation_ab5[j1,j2,p] = np.power(1j, abs(m1 - m2) - abs(m1) - abs(m2) + l2 - l1 + p) * np.power(-1.0, m1-m2) * \
                      np.sqrt((2 * l1 + 1) * (2 * l2 + 1) / (2 * l1 * (l1 + 1) * l2 * (l2 + 1))) * \
                      (l1 * (l1 + 1) + l2 * (l2 + 1) - p * (p + 1)) * np.sqrt(2 * p + 1) * \
                      wigner_3j(l1, l2, p, m1, -m2, -m1+m2) * wigner_3j(l1, l2, p, 0, 0, 0)
                  else:
                    self.translation_ab5[j1,j2,p] = np.power(1j, abs(m1 - m2) - abs(m1) - abs(m2) + l2 - l1 + p) * np.power(-1.0, m1-m2) * \
                      np.sqrt((2 * l1 + 1) * (2 * l2 + 1) / (2 * l1 * (l1 + 1) * l2 * (l2 + 1))) * \
                      np.lib.scimath.sqrt((l1 + l2 + 1 + p) * (l1 + l2 + 1 - p) * (p + l1 - l2) * (p - l1 + l2) * (2 * p + 1)) * \
                      wigner_3j(l1, l2, p, m1, -m2, -m1+m2) * wigner_3j(l1, l2, p-1, 0, 0, 0)
    # function translation = translation_table_ab(lmax)

    # jmax = jmult_max(1,lmax);

    # translation.ab5 = zeros(jmax,jmax,2*lmax+1,'single');

    # for tau1 = 1:2
    #     for l1 = 1:lmax
    #         for m1 = -l1:l1
    #             j1 = multi2single_index(1,tau1,l1,m1,lmax);
    #             for tau2 = 1:2
    #                 for l2 = 1:lmax
    #                     for m2 = -l2:l2
    #                         j2 = multi2single_index(1,tau2,l2,m2,lmax);
    #                         for p = 0:2*lmax
    #                             if tau1 == tau2
    #                                 translation.ab5(j1,j2,p+1) = (1i)^(abs(m1-m2)-abs(m1)-abs(m2)+l2-l1+p) * (-1)^(m1-m2) ...
    #                                     * sqrt((2*l1+1)*(2*l2+1)/(2*l1*(l1+1)*l2*(l2+1))) * (l1*(l1+1)+l2*(l2+1)-p*(p+1)) * sqrt(2*p+1) ...
    #                                     * Wigner3j([l1,l2,p],[m1,-m2,-m1+m2]) * Wigner3j([l1,l2,p],[0,0,0]);
    #                             elseif p > 0
    #                                 translation.ab5(j1,j2,p+1) = (1i)^(abs(m1-m2)-abs(m1)-abs(m2)+l2-l1+p) * (-1)^(m1-m2) ...
    #                                     * sqrt((2*l1+1)*(2*l2+1)/(2*l1*(l1+1)*l2*(l2+1))) * sqrt((l1+l2+1+p)*(l1+l2+1-p)*(p+l1-l2)*(p-l1+l2)*(2*p+1)) ...
    #                                     * Wigner3j([l1,l2,p],[m1,-m2,-m1+m2]) * Wigner3j([l1,l2,p-1],[0,0,0]);
    #                             end
    #                         end
    #                     end
    #                 end
    #             end
    #         end
    #     end
    # end
    # end