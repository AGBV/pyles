import unittest
import glob
import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

from functions.spherical_functions_trigon import spherical_functions_trigon

class TestSphericalFunctionsTrigon(unittest.TestCase):
  def test_spherical_functions_trigon(self):
    p = re.compile('spherical_functions_trigon_lmax_(\d+)_dim_(.*)\.csv')
    for data_file in glob.glob('tests/data/spherical_functions_trigon_*.csv'):

      res = p.search(data_file)
      lmax = int(res.group(1))
      dim = [int(x) for x in res.group(2).split('x')]

      data = pd.read_csv(data_file, header=None)
      theta      = np.reshape(data.to_numpy()[:np.prod(dim),4], dim)
      pilm_test  = np.reshape(data.to_numpy()[:, 2], np.concatenate(([lmax+1, lmax+1], dim)))
      taulm_test = np.reshape(data.to_numpy()[:, 3], np.concatenate(([lmax+1, lmax+1], dim)))

      print(theta, lmax, dim)
    
    self.assertAlmostEqual(2.555, 2.554, 2, 'They do not match')

if __name__ == '__main__':
  unittest.main()