import unittest
import glob
import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

from pyles.functions.t_entry import t_entry
from pyles.functions.misc import multi2single_index

class TestScattering(unittest.TestCase):

  def test_t_entry_single_wavelength(self):
    p = re.compile(r'mie_coefficients_lmax_(\d+)_wavelength_(\d+\.?\d*)_medium_(-?\d+\.?\d*)_(-?\d+\.?\d*)\.csv')
    for data_file in glob.glob('tests/data/mie_coefficients_lmax*.csv'):

      res = p.search(data_file)
      lmax = int(res.group(1))
      wavelength = float(res.group(2))
      medium_real = float(res.group(3))
      medium_imag = float(res.group(4))
      medium = medium_real + 1j*medium_imag

      omega = 2 * np.pi / wavelength

      data = pd.read_csv(data_file, header=None).applymap(lambda val: complex(val.replace('i', 'j'))).to_numpy()
      radii     = np.real(data[:,0])
      ref_idx   = data[:,1]
      mie_test  = data[:,2:]

      mie = np.zeros(
        (data.shape[0],
        2 * lmax * (lmax + 2)),
        dtype=complex)
      
      for u_i in range(data.shape[0]):
        for tau in range(1, 3):
          for l in range(1, lmax+1):
            for m in range(-l,l+1):
              jmult = multi2single_index(0, tau, l, m, lmax)
              mie[u_i,jmult] = t_entry(tau=tau, l=l,
                k_medium = omega * medium,
                k_sphere = omega * ref_idx[u_i],
                radius = radii[u_i])

      np.testing.assert_array_almost_equal(mie,  mie_test,  decimal=5, err_msg='Mie coefficients do not match in %s' % (data_file), verbose=True)

  def test_t_entry_multiple_wavelength(self):
    p = re.compile(r'mie_coefficients_.+_lmax_(\d+)_(.+)\.csv')
    for data_file in glob.glob('tests/data/mie_coefficients_*_lmax*.csv'):
      
      res = p.search(data_file)
      lmax = int(res.group(1))

      data = pd.read_csv(data_file, header=None).applymap(lambda val: complex(str(val).replace('i', 'j'))).to_numpy()

      num_particles = np.where(np.isnan(data[:,2]))[0][0]

      wavelength  = np.real(data[:,0])
      med_ref_idx = data[:,1]
      radii       = np.real(data[:num_particles,2])
      ref_idx     = data[:num_particles,3]
      mie_test    = np.reshape(data[:,4:].T, (num_particles, 2 * lmax * (lmax + 2), wavelength.size), order='F');

      omega = 2 * np.pi / wavelength

      mie = np.zeros(
        (num_particles,
        2 * lmax * (lmax + 2),
        wavelength.size),
        dtype=complex)
      
      for u_i in range(num_particles):
        for tau in range(1, 3):
          for l in range(1, lmax+1):
            for m in range(-l,l+1):
              jmult = multi2single_index(0, tau, l, m, lmax)
              mie[u_i,jmult,:] = t_entry(tau=tau, l=l,
                k_medium = omega * med_ref_idx,
                k_sphere = omega * ref_idx[u_i],
                radius = radii[u_i])

      np.testing.assert_array_almost_equal(mie,  mie_test,  decimal=5, err_msg='Mie coefficients do not match in %s' % (data_file), verbose=True)

if __name__ == '__main__':
  unittest.main()