import unittest
import glob
import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

from src.numerics import Numerics

class TestNumerics(unittest.TestCase):

  def test_translation_table(self):
    executions = 0
    module = 'numerics'
    relative_precision = 1e-7
    # p = re.compile(module + r'_(.+)(\.big)?\.json')
    for data_file in glob.glob('tests/data/%s_*.json' % module):
      executions += 1

      # res = p.search(data_file)
      # identifier = res.group(1)
      # print(identifier)

      data = pd.read_json(data_file)
      lmax = data['input']['lmax']
      polar_angles = np.array(data['input']['polar_angles'])
      azimuthal_angles = np.array(data['input']['azimuthal_angles'])

      numerics = Numerics(lmax=lmax,
                    polar_angles=polar_angles,
                    azimuthal_angles=azimuthal_angles,
                    gpu=False,
                    particle_distance_resolution=1)

      numerics.compute_translation_table()
      translation_table_test = np.array(data['output']['numerics']['translation_table'], dtype=complex)

      np.testing.assert_allclose(numerics.translation_ab5, translation_table_test, relative_precision, relative_precision**2, True, 'The translation table coefficients do not match.')
    
    self.assertGreater(executions, 0, 'No test data provided to be run.')


if __name__ == '__main__':
  np.set_printoptions(edgeitems=3, linewidth=np.inf)
  unittest.main()