import unittest
import glob
import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
from scipy.io import loadmat

from src.particles import Particles
from src.initial_field import InitialField
from src.parameters import Parameters
from src.solver import Solver
from src.numerics import Numerics
from src.simulation import Simulation

class TestSimulation(unittest.TestCase):

  def test_setup(self):
    module = 'test'
    relative_precision = 1e-6
    p = re.compile(module + r'_(.+)(\.big)?\.json')
    for data_file in glob.glob('tests/data/%s_*.json' % module):

      # res = p.search(data_file)
      # identifier = res.group(1)
      # print(identifier)

      data = pd.read_json(data_file, dtype=True)
      lmax = data['input']['lmax']
      spheres = np.array(data['input']['particles'])
      wavelength = np.array(data['input']['wavelength'])
      medium_ref_idx = np.array([complex(x.replace('i', 'j')) for x in data['input']['medium_ref_idx']])

      polar_angle = data['input']['polar_angle']
      azimuthal_angle = data['input']['azimuthal_angle']
      polar_angles = np.array(data['input']['polar_angles'])
      azimuthal_angles = np.array(data['input']['azimuthal_angles'])

      particles = Particles(spheres[:,0:3], spheres[:,3], spheres[:,4:])
      initial_field = InitialField(beam_width=0,
                             focal_point=np.array((0,0,0)),
                             polar_angle=polar_angle,
                             azimuthal_angle=azimuthal_angle,
                             polarization='TE')
      
      parameters = Parameters(wavelength=wavelength,
                    medium_mefractive_index=medium_ref_idx,
                    particles=particles,
                    initial_field=initial_field)

      numerics = Numerics(lmax=lmax,
                    polar_angles=polar_angles,
                    azimuthal_angles=azimuthal_angles,
                    particle_distance_resolution=1,
                    gpu=False)

      simulation = Simulation(parameters, numerics)
      simulation.legacy_compute_lookup_particle_distances()
      simulation.legacy_compute_h3_table()

      lookup_particle_distances = np.array(data['output']['simulation']['setup']['lookup_particle_distances'])
      h3_table = np.array(data['output']['simulation']['setup']['h3_table'], dtype=float)

      np.testing.assert_allclose(simulation.lookup_particle_distances, lookup_particle_distances, relative_precision, 0, True, 'The lookup particles distances do not match.')
      np.testing.assert_allclose(np.real(simulation.h3_table), h3_table[:,:,0,:], relative_precision, 0, True, 'The real part of the h3 table does not match.')
      np.testing.assert_allclose(np.imag(simulation.h3_table), h3_table[:,:,1,:], relative_precision, 0, True, 'The imag part of the h3 table does not match.')
      

  def test_initial_field_coefficients_planewave(self):
    executions = 0
    module = 'precoupling'
    relative_precision = 1e-3
    for data_file in glob.glob('tests/data/simulation/%s_*.mat' % module):

      data = loadmat(data_file)

      lmax = int(data['lmax'][0][0])
      spheres = data['spheres']

      wavelength = data['wavelength'].squeeze()
      medium_ref_idx = data['medium_ref_idx'].squeeze()

      polar_angle = data['polar_angle'][0][0]
      azimuthal_angle = data['azimuthal_angle'][0][0]
      polar_angles = data['polar_angles'].squeeze()
      azimuthal_angles = data['azimuthal_angles'].squeeze()

      particles = Particles(spheres[:,0:3], spheres[:,3], spheres[:,4:])
      initial_field = InitialField(
                      beam_width=0,
                      focal_point=np.array((0,0,0)),
                      polar_angle=polar_angle,
                      azimuthal_angle=azimuthal_angle,
                      polarization='TE')

      parameters = Parameters(
                    wavelength=wavelength,
                    medium_mefractive_index=medium_ref_idx,
                    particles=particles,
                    initial_field=initial_field)

      numerics = Numerics(
                  lmax=lmax,
                  polar_angles=polar_angles,
                  azimuthal_angles=azimuthal_angles,
                  gpu=False,
                  particle_distance_resolution=1)

      simulation = Simulation(parameters, numerics)

      simulation.compute_mie_coefficients()
      simulation.compute_initial_field_coefficients()
      initial_field_coefficients_test = data['ifc']

      np.testing.assert_allclose(simulation.initial_field_coefficients, initial_field_coefficients_test, relative_precision, 0, True, 'The initial field coefficients do not match.')
      executions += 1

    self.assertGreater(executions, 0, 'No test data provided to be run.')

  def test_scattered_field_coefficients(self):
    np.set_printoptions(precision=4, edgeitems=3, linewidth=np.inf, suppress=True, formatter={
      'complexfloat': lambda x: '% +11.4e% +11.4ej ' % (x.real, x.imag)
    },floatmode='fixed')
    # import logging
    # logging.basicConfig(format='%(levelname)s (%(name)s): %(message)s', level=logging.INFO)
    executions = 0
    module = 'postcoupling'
    relative_precision = 5e-1
    for data_file in glob.glob('tests/data/simulation/%s_*.mat' % module):

      data = loadmat(data_file)

      lmax = int(data['lmax'][0][0])
      spheres = data['spheres']

      wavelength = data['wavelength'].squeeze()
      medium_ref_idx = data['medium_ref_idx'].squeeze()

      polar_angle = data['polar_angle'][0][0]
      azimuthal_angle = data['azimuthal_angle'][0][0]
      polarization = str(data['polarization'][0])
      beam_width = float(data['beamwidth'][0][0])

      polar_angles = data['polar_angles'].squeeze()
      azimuthal_angles = data['azimuthal_angles'].squeeze()

      solver_type = str(data['solver_type'][0])
      # solver_type = 'lgmres'
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
                          polar_angles=polar_angles,
                          azimuthal_angles=azimuthal_angles,
                          gpu=False,
                          particle_distance_resolution=1,
                          solver=solver)

      simulation = Simulation(parameters, numerics)

      numerics.compute_translation_table()
      simulation.compute_mie_coefficients()
      simulation.compute_initial_field_coefficients()
      simulation.compute_right_hand_side()

      scattered_field_coefficients_test = data['sfc']

      simulation.compute_scattered_field_coefficients()

      np.testing.assert_allclose(simulation.scattered_field_coefficients, scattered_field_coefficients_test, relative_precision, 0, True, 'The scattered field cofficients do not match.')
      executions += 1

    self.assertGreater(executions, 0, 'No test data provided to be run.')


if __name__ == '__main__':
  unittest.main()
