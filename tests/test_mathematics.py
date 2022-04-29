import unittest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

from sympy.physics.wigner import wigner_3j
from pyles.functions.wigner3j import wigner3j

class TestMathematics(unittest.TestCase):

  def test_wigner3j(self):
    precision = 10
    lmax = 3

    for tau1 in range(1,3):
      for l1 in range(1,lmax+1):
        for m1 in range(-l1, l1+1):
          for tau2 in range(1,3):
            for l2 in range(1,lmax+1):
              for m2 in range(-l2,l2+1):
                for p in range(0,2*lmax+1):
                  if tau1 == tau2:
                    self.assertAlmostEqual(
                      wigner3j(l1, l2, p, m1, -m2, -m1+m2),
                      wigner_3j(l1, l2, p, m1, -m2, -m1+m2),
                      precision,
                      'Wigner 3j symbols (%d, %d, %d; %d, %d, %d) does not match' % (l1, l2, p, m1, -m2, -m1+m2))
                    self.assertAlmostEqual(
                      wigner3j(l1, l2, p, 0, 0, 0),
                      wigner_3j(l1, l2, p, 0, 0, 0),
                      precision,
                      'Wigner 3j symbols (%d, %d, %d; %d, %d, %d) does not match' % (l1, l2, p, 0, 0, 0))

if __name__ == '__main__':
  unittest.main()