import logging

import numpy as np

from src.simulation import Simulation

class Optics:
  def __init__(self, simulation: Simulation):
    self.simulation = simulation
    print("hi")

    self.log = logging.getLogger(__name__)

  def compute_cross_sections(self):
    pass

    # a = gather(sCeles.simul.tables.initialFieldCoefficients);
    # f = gather(sCeles.simul.tables.scatteredFieldCoefficients);

    # sResults.Cext(j)    = sResults.Cext(j)    - sum(conj(a(:)) .* f(:));
    # sResults.CscaLin(j) = sResults.CscaLin(j) + sum(abs(f(:)).^2);

    # for p = 0:2*sCeles.lmax
    #     for n = 1:numel(fmMMcomp)
    #         if abs(fmMMcomp(n)) <= p && all(~isnan(Plm{p+1,abs(fmMMcomp(n))+1}(:)))
    #             sResults.Csca(j) =  sResults.Csca(j) + ...
    #                 real(sCeles.simul.tables.translationTable.ab5(n2(n),n1(n),p+1) .* ...
    #                 f(:,n1(n))' * ...
    #                 (Plm{p+1,abs(fmMMcomp(n))+1} .* sphBesel{j, p+1} .* exp(1i*fmMMcomp(n)*phiTab)) * ...
    #                 f(:, n2(n)));
    #         end
    #     end
    # end

  def compute_phase_funcition(self):
    pass

    # E1sca2 = zeros(numel(sResults.fmPhi), 3);
    # for s = 1:length(sParticles.fvRadii)
    #     for l = 1:sCeles.lmax
    #         for m = -l:l
    #             for tau = 1:2
    #                 n = multi2single_index(1,tau,l,m,sCeles.lmax);
    #                 t = (1i)^(tau-l-2) * f(s,n) * exp(-1i*k(j)*dot(repmat(sParticles.fmPosition(s,:), size(e_r,1), 1), e_r, 2)) .* exp(1i*m*sResults.fmPhi) / sqrt(2*l*(l+1));
    #                 if tau == 1     %select M
    #                     E1sca2 = E1sca2 + t .* (e_theta .* pi_all{l+1,abs(m)+1} * 1i * m - e_phi   .* tau_all{l+1,abs(m)+1});
    #                 else            %select N
    #                     E1sca2 = E1sca2 + t .* (e_phi   .* pi_all{l+1,abs(m)+1} * 1i * m + e_theta .* tau_all{l+1,abs(m)+1});
    #                 end
    #             end
    #         end
    #     end
    # end

    # sResults.pCmpl(j,:) = sResults.pCmpl(j,:) + sum(abs(E1sca2).^2, 2)';