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
    p = re.compile(module + r'_(.+)(\.big)?\.json')
    for data_file in glob.glob('tests/data/%s_*.json' % module):

      res = p.search(data_file)
      identifier = res.group(1)
      print(identifier)

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

      h3_table = np.array(data['output']['simulation']['setup']['h3_table'], dtype=float)
      h3_table_real = h3_table[:,:,0,:]

      simulation_h3_table_real = np.real(simulation.h3_table)
      simulation_h3_table_imag = np.imag(simulation.h3_table)


      # for k in range(h3_table.shape[0]):
      #   for l in range(h3_table.shape[1]):
      #     if not np.allclose(h3_table[k,l,0,:], simulation_h3_table_real[k,l,:], rtol=1e-3, atol=1e-5, equal_nan=True):
      #       for m in range(h3_table.shape[3]):
      #         if not np.allclose(h3_table[k,l,0,m], simulation_h3_table_real[k,l,m], rtol=1e-3, atol=1e-5, equal_nan=True):
      #           print(k, simulation.size_param[l, m], simulation_h3_table_real[k,l,m], h3_table[k,l,0,m])


      np.testing.assert_array_almost_equal(simulation.lookup_particle_distances, data['output']['simulation']['setup']['lookup_particle_distances'], 8, 'The lookup particles distances do not match.')
      # np.testing.assert_array_almost_equal(simulation_h3_table_real, h3_table_real, 0, 'The h3 table does not match.')

  # def test_initial_field_coefficients_planewave(self):

  #   module = 'initial_field_coefficients_planewave'
  #   p = re.compile(module + r'_(.+)(\.big)?\.json')
  #   for data_file in glob.glob('tests/data/simulation/%s_*.json' % module):

  #     res = p.search(data_file)
  #     identifier = res.group(1)

  #     data = pd.read_json(data_file)
  #     lmax = data['input']['lmax']
  #     spheres = np.array(data['input']['particles'])
  #     wavelength = np.array(data['input']['wavelength'])
  #     medium_ref_idx = np.array([complex(x.replace('i', 'j')) for x in data['input']['medium_ref_idx']])

  #     polar_angle = data['input']['polar_angle']
  #     azimuthal_angle = data['input']['azimuthal_angle']
  #     polar_angles = np.array(data['input']['polar_angles'])
  #     azimuthal_angles = np.array(data['input']['azimuthal_angles'])

  #     particles = Particles(spheres[:,0:3], spheres[:,3], spheres[:,4:])
  #     initial_field = InitialField(beam_width=0,
  #                            focal_point=np.array((0,0,0)),
  #                            polar_angle=polar_angle,
  #                            azimuthal_angle=azimuthal_angle,
  #                            polarization='TE')
  #     inputs = Parameters(wavelength=wavelength,
  #                   medium_mefractive_index=medium_ref_idx,
  #                   particles=particles,
  #                   initial_field=initial_field)

  #     numerics = Numerics(lmax=lmax,
  #                   polar_angles=polar_angles,
  #                   azimuthal_angles=azimuthal_angles,
  #                   gpu=False)

  #     # simulation = Simulation(inputs, numerics)


  #     # self.assertEqual(particles.num_unique_refractive_indices, num_unique_refractive_indices, 'Number of unique refractive indices is not equal.')
  #     # np.testing.assert_array_equal(particles.unique_refractive_indices, unique_refractive_indices, 'Unique refractive indices do not matchh.')
  #     # np.testing.assert_array_equal(particles.refractive_index_array_idx, refractive_index_array_idx, 'Refractive index array indices do not match.')


if __name__ == '__main__':
  unittest.main()