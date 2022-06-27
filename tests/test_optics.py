import unittest
import glob

import numpy as np
from scipy.io import loadmat

from src.particles import Particles
from src.initial_field import InitialField
from src.parameters import Parameters
from src.solver import Solver
from src.numerics import Numerics
from src.simulation import Simulation
from src.optics import Optics



class TestOptics(unittest.TestCase):

  def test_cross_sections_and_phase_function(self):
    executions = 0
    module = 'mstm'
    relative_precision = 6e-1
    for data_file in glob.glob('tests/data/%s_*.mat' % module):

      data = loadmat(data_file)

      lmax = 4
      spheres = data['spheres']

      wavelength = data['wavelength'].squeeze()
      medium_ref_idx = data['medium_ref_idx'].squeeze()

      polar_angle = data['polar_angle'][0][0]
      azimuthal_angle = data['azimuthal_angle'][0][0]
      polarization = str(data['polarization'][0])
      beam_width = float(data['beamwidth'][0][0])

      solver_type = str(data['solver_type'][0])
      solver_type = 'lgmres'
      tolerance = float(data['tolerance'][0][0])
      max_iter = int(data['max_iter'][0][0])
      restart = int(data['restart'][0][0])

      particles = Particles(spheres[:,0:3], spheres[:,3], spheres[:,4:])
      initial_field = InitialField(beam_width=beam_width,
                                    focal_point=np.array((0,0,0)),
                                    polar_angle=polar_angle,
                                    azimuthal_angle=azimuthal_angle,
                                    polarization=polarization)

      parameters = Parameters(wavelength=wavelength,
                              medium_mefractive_index=medium_ref_idx,
                              particles=particles,
                              initial_field=initial_field)

      solver = Solver(solver_type=solver_type,
                      tolerance=tolerance,
                      max_iter=max_iter,
                      restart=restart)

      numerics = Numerics(lmax=lmax,
                          sampling_points_number=[360, 181],
                          gpu=True,
                          particle_distance_resolution=1,
                          solver=solver)

      simulation = Simulation(parameters, numerics)
      optics = Optics(simulation)

      particles.compute_volume_equivalent_area()
      numerics.compute_translation_table()
      numerics.compute_spherical_unity_vectors()
      simulation.compute_mie_coefficients()
      simulation.compute_initial_field_coefficients()
      simulation.compute_right_hand_side()
      simulation.compute_scattered_field_coefficients()

      optics.compute_cross_sections()
      optics.compute_phase_funcition(legendre_coefficients_number=15, c_and_b=([-np.inf, 0], [np.inf, 1]))

      q_ext_test = data['q_ext'].squeeze()
      q_sca_test = data['q_sca'].squeeze()
      albedo_test = data['albedo'].squeeze()
      angle_test = data['phase_fun_angle'].squeeze()
      phase_function_test = np.transpose(data['phase_fun'])

      q_ext  = optics.c_ext / particles.geometric_projection
      q_sca  = optics.c_sca / particles.geometric_projection
      albedo = optics.albedo
      phase_function = np.empty((angle_test.size, optics.phase_function.shape[1]))
      for w in range(optics.phase_function.shape[1]):
        phase_function[:, w] = np.interp(np.deg2rad(angle_test), optics.scattering_angles, optics.phase_function[:, w])
    
      np.testing.assert_allclose(q_ext,  q_ext_test, relative_precision,                    0, True, 'The extinction efficiencies do not match.')
      np.testing.assert_allclose(q_sca,  q_sca_test, relative_precision,                    0, True, 'The scattering efficiencies do not match.')
      np.testing.assert_allclose(albedo, albedo_test, relative_precision,                   0, True, 'The albedos do not match.')
      np.testing.assert_allclose(phase_function,  phase_function_test, relative_precision,  0, True, 'The phase function does not match.')

      executions += 1

    self.assertGreater(executions, 0, 'No test data provided to be run.')


if __name__ == '__main__':
  unittest.main()
