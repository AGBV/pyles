import sys
sys.path.append('../..', '..')

import numpy as np

from simulation import Simulation

from spherical_functions_trigon import spherical_functions_trigon

def initial_field_coefficients_planewave(simulation: Simulation):
  lmax = simulation.numerics.lmax
  E0 = simulation.parameters.initial_field.amplitude
  k = simulation.parameters.k_medium

  beta = simulation.parameters.initial_field.polar_angle
  cb = np.cos(beta)
  sb = np.sin(beta)
  alpha = simulation.parameters.initial_field.azimuthal_angle

  # pi and tau symbols for transformation matrix B_dagger
  pilm,taulm = spherical_functions_trigon(beta,lmax)


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