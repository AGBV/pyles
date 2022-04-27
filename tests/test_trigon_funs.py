import unittest
import glob
import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

from pyles.functions.spherical_functions_trigon import spherical_functions_trigon
from pyles.functions.legendre_normalized_trigon import legendre_normalized_trigon

# If pylint shows problems with np.testing, use @staticmethod
# Source: https://stackoverflow.com/questions/61950738/using-numpy-testing-functions-with-unittest
class TestTrigonFunctions(unittest.TestCase):

  def test_legendre_normalized_trigon(self):
    p = re.compile('legendre_normalized_trigon_lmax_(\d+)_dim_(.*)\.csv')
    for data_file in glob.glob('tests/data/legendre_normalized_trigon_*.csv'):

      res = p.search(data_file)
      lmax = int(res.group(1))
      dim = [int(x) for x in res.group(2).split('x')]

      data = pd.read_csv(data_file, header=None)
      theta     = np.reshape(data.to_numpy()[:np.prod(dim),0], dim, order='F')
      plm_test  = np.reshape(data.to_numpy()[:, 1], np.concatenate(([lmax+1, lmax+1], dim)), order='F')

      plm = legendre_normalized_trigon(np.radians(theta), lmax=lmax)

      np.testing.assert_array_almost_equal(plm,  plm_test,  decimal=10, err_msg='P_lm do not match in %s' % (data_file), verbose=True)

  def test_spherical_functions_trigon(self):
    p = re.compile('spherical_functions_trigon_lmax_(\d+)_dim_(.*)\.csv')
    for data_file in glob.glob('tests/data/spherical_functions_trigon_*.csv'):

      res = p.search(data_file)
      lmax = int(res.group(1))
      dim = [int(x) for x in res.group(2).split('x')]

      data = pd.read_csv(data_file, header=None)
      theta      = np.reshape(data.to_numpy()[:np.prod(dim),0], dim, order='F')
      pilm_test  = np.reshape(data.to_numpy()[:, 1], np.concatenate(([lmax+1, lmax+1], dim)), order='F')
      taulm_test = np.reshape(data.to_numpy()[:, 2], np.concatenate(([lmax+1, lmax+1], dim)), order='F')

      pilm, taulm = spherical_functions_trigon(np.radians(theta), lmax)

      np.testing.assert_array_almost_equal(pilm,  pilm_test,  decimal=10, err_msg='PI_lm do not match in %s' % (data_file), verbose=True)
      np.testing.assert_array_almost_equal(taulm, taulm_test, decimal=10, err_msg='TAU_lm do not match in %s' % (data_file), verbose=True)

if __name__ == '__main__':
  unittest.main()