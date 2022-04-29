import logging
import numpy as np
from scipy.special import gammaln

def wigner3j(j1, j2, j3, m1, m2, m3, verbose=False):
  """
  Calculates the Wigner 3j symbol for given j1, j2, j3, m1, m2, m3
  Based on the work of https://de.mathworks.com/matlabcentral/fileexchange/5275-wigner3j-m

  Faster approach: https://paperzz.com/doc/8141260/wigner-3j--6j-and-9j-symbols
  """

  log = logging.getLogger('wigner3j')

  # Input error checking
  if np.any(np.array([j1, j2, j3]) < 0):
    log.error('The j must be non-negative')
    return None
  elif np.any(np.array([j1, j2, j3, m1, m2, m3]) % 0.5 != 0):
    log.error('All arguments must be integers or half-integers')
    return None
  elif np.any((np.array([j1, j2, j3]) - np.array([m1, m2, m3])) % 1 != 0):
    log.error('[j1, j2, j3] and [m1, m2, m3] must to have the same pairwise parity')
    return None

  # Selection rules
  if (j3 > (j1 + j2)) or (j3 < np.abs(j1 - j2)) \
    or (m1 + m2 + m3 != 0) \
    or np.any(np.array(np.abs([m1, m2, m3])) > np.array([j1, j2, j3])):
    return 0

  # Simple common case
  if np.all(np.array([m1, m2, m3]) == 0) and (np.sum([j1, j2, j3]) % 2 == 1):
    return 0

  # Evalutation
  t1 = j2 - m1 - j3
  t2 = j1 + m2 - j3
  t3 = j1 + j2 - j3
  t4 = j1 - m1
  t5 = j2 + m2

  tmin = np.max([0,  t1, t2])
  tmax = np.min([t3, t4, t5])

  t = np.arange(tmin, tmax+1)

  w = np.sum(np.power(-1.0, t+j1-j2-m3) * \
    np.exp(\
      -1 * np.matmul(np.ones((1,6)), gammaln(np.stack((t, t-t1, t-t2, t3-t, t4-t, t5-t))+1)) + \
      0.5 * np.matmul(\
        gammaln(np.array([j1+j2+j3+1, j1+j2-j3, j1-j2+j3, -j1+j2+j3, j1+m1, j1-m1, j2+m2, j2-m2, j3+m3, j3-m3])+1), \
        np.array([-1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
      )
    ))
  if np.isnan(w) and verbose:
    print('Result is NaN!')
  elif np.isinf(w) and verbose:
    print('Result is Inf!')
  
  return w

if __name__ == '__main__':
  wigner3j(1,1,1,0,0,0)