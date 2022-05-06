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

class TestParameters(unittest.TestCase):
  pass

  def test_setup(self):
    module = 'test'
    p = re.compile(module + r'_(.+)(\.big)?\.json')
    for data_file in glob.glob('tests/data/%s_*.json' % module):

      res = p.search(data_file)
      identifier = res.group(1)

      data = pd.read_json(data_file, dtype=True)
      # lmax = data['input']['lmax']
      spheres = np.array(data['input']['particles'])
      wavelength = np.array(data['input']['wavelength'])
      medium_ref_idx = np.array([complex(x.replace('i', 'j')) for x in data['input']['medium_ref_idx']])

      polar_angle = data['input']['polar_angle']
      azimuthal_angle = data['input']['azimuthal_angle']

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

      test_omega = np.array(data['output']['parameters']['setup']['omega'])
      test_k_medium = np.array(data['output']['parameters']['setup']['k_medium']).astype(complex)
      test_k_particle = np.array(data['output']['parameters']['setup']['k_particle'])

      np.testing.assert_array_almost_equal(parameters.omega, test_omega, 10, 'The omega does not match.')
      np.testing.assert_array_almost_equal(parameters.k_medium, test_k_medium, 10, 'The wavenumber of the medium does not match.')
      np.testing.assert_array_almost_equal(parameters.k_particle, test_k_particle, 10, 'The wavenumber of the particles do not match.')


if __name__ == '__main__':
  unittest.main()