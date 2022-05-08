import unittest
import glob
import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

from pyles.functions.misc import jmult_max, multi2single_index, single_index2multi

class TestSimulation(unittest.TestCase):

  def test_jmult_max(self):
    lmax_array = [2, 4, 9 , 100]
    num_part_array = [10, 50, 100, 500]
    for lmax in lmax_array:
      for num_part in num_part_array:
        self.assertEquals(jmult_max(num_part, lmax), 2 * num_part * lmax * (lmax + 2), 'The number of indices for the configuration num_part=%d and lmax=%d is not mathing.' % (num_part, lmax))

  def test_indexing(self):
    lmax_array = [2, 4, 9]
    num_part_array = [10, 50, 100]
    for lmax in lmax_array:
      for num_part in num_part_array:
        for j_s in range(num_part):
          for tau in range(1, 3):
            for l in range(1, lmax+1):
              for m in range(-l, l+1):
                idx = multi2single_index(j_s, tau, l, m, lmax)
                j_s_calc, tau_calc, l_calc, m_calc = single_index2multi(idx, lmax)
                
                self.assertEquals(j_s, j_s_calc, 'The particle index does not match')
                self.assertEquals(tau, tau_calc, 'The polarization does not match')
                self.assertEquals(l, l_calc, 'The degree does not match')
                self.assertEquals(m, m_calc, 'The order does not match')      

if __name__ == '__main__':
  unittest.main()
