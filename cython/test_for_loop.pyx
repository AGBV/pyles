#!python
#cython: language_level=3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

from time import time

import numpy as np
# cimport numpy as np

from pyles.functions.misc import jmult_max
from pyles.functions.misc import multi2single_index

cdef int lmax = 4
cdef int num_part = 500

cdef int j1, j2
cdef int tau1, l1, m1
cdef int tau2, l2, m2
cdef float t1, t2, t

cdef long int x = 0

t1 = time()
for s1 in range(num_part):
  # for s2 in range(num_part):
    for tau1 in range(1, 3):
      for l1 in range(1, lmax+1):
        for m1 in range(-l1, l1+1):
          j1 = multi2single_index(0, tau1, l1, m1, lmax)
          for tau2 in range(1, 3):
            for l2 in range(1, lmax+1):
              for m2 in range(-l2, l2+1):
                j2 = multi2single_index(0, tau2, l2, m2, lmax)
                x += np.sqrt(j1 * j2)
t2 = time()
t = t2 - t1
print(x)
print("%.5f" % t)

x = 0
t1 = time()
for s1 in range(num_part):
  # for s2 in range(num_part):
    for j1 in range(jmult_max(1, lmax)):
      for j2 in range(jmult_max(1, lmax)):
        x += np.sqrt(j1 * j2)
t2 = time()
t = t2 - t1
print(x)
print("%.5f" % t)
