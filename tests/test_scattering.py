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
    for data_file in glob.glob('tests/data/mie_coefficients_*.csv'):

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
        data.shape[1]-2),
        dtype=complex)
      
      for u_i in range(data.shape[0]):
        for tau in range(1, 3):
          for l in range(1, lmax+1):
            for m in range(-l,l+1):
              jmult = multi2single_index(0, tau, l, m, lmax)
              mie[u_i,jmult] = t_entry(tau=tau, l=l,
                kM = omega * medium,
                kS = omega * ref_idx[u_i],
                R = radii[u_i])

      np.testing.assert_array_almost_equal(mie,  mie_test,  decimal=6, err_msg='Mie coefficients do not match in %s' % (data_file), verbose=True)

if __name__ == '__main__':
  unittest.main()