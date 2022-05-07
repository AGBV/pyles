import unittest
import glob
import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

from pyles.particles import Particles
from pyles.initial_field import InitialField
from pyles.parameters import Parameters
from pyles.numerics import Numerics
from pyles.simulation import Simulation

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

      lookup_particle_distances = np.array(data['output']['simulation']['setup']['lookup_particle_distances'])
      h3_table = np.array(data['output']['simulation']['setup']['h3_table'], dtype=float)

      np.testing.assert_allclose(simulation.lookup_particle_distances, lookup_particle_distances, relative_precision, 0, True, 'The lookup particles distances do not match.')
      np.testing.assert_allclose(np.real(simulation.h3_table), h3_table[:,:,0,:], relative_precision, 0, True, 'The real part of the h3 table does not match.')
      np.testing.assert_allclose(np.imag(simulation.h3_table), h3_table[:,:,1,:], relative_precision, 0, True, 'The imag part of the h3 table does not match.')
      

  def test_initial_field_coefficients_planewave(self):
    executions = 0
    module = 'test'
    relative_precision = 1e-8
    p = re.compile(module + r'_(.+)(\.big)?\.json')
    for data_file in glob.glob('tests/data/%s_*.json' % module):

      # res = p.search(data_file)
      # identifier = res.group(1)
      # print(identifier)

      data = pd.read_json(data_file)
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
                    gpu=False)

      simulation = Simulation(parameters, numerics)

      simulation.compute_mie_coefficients()
      simulation.compute_initial_field_coefficients()
      initial_field_coefficients_test = np.array(data['output']['simulation']['initial_field_coefficients'], dtype=complex)

      np.testing.assert_allclose(simulation.initial_field_coefficients, initial_field_coefficients_test, relative_precision, 0, True, 'The initial field coefficients do not match.')
      executions += 1

    self.assertGreater(executions, 0, 'No test data provided to be run.')


if __name__ == '__main__':
  unittest.main()
