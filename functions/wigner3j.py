import numpy as np
from scipy.special import gammaln

# def wigner3j(j123, m123):
def wigner3j(j1, j2, j3, m1, m2, m3):
  # Input error checking
  if np.any(np.array([j1, j2, j3]) < 0):
    print('The j must be non-negative')
    return None
  elif np.any(np.array([j1, j2, j3, m1, m2, m3]) % 0.5 != 0):
    print('All arguments must be integers or half-integers')
    return None
  elif np.any((np.array([j1, j2, j3]) - np.array([m1, m2, m3])) % 1 != 0):
    print('[j1, j2, j3] and [m1, m2, m3] do not match')

  # Selection rules
  if (j3 > (j1 + j2)) or (j3 < np.abs(j1 - j2)) \
    or (m1 + m2 + m3 != 0) or np.any(np.array([m1, m2, m3]) > np.array([j1, j2, j3])):
    return 

  # Simple common case
  if np.all(np.array([m1, m2, m3]) == 0) and (np.sum([j1, j2, j3]) % 2 == 1):
    return 0

  # Evalutation
  t1 = j2 - m1 -j3
  t2 = j1 + m2 - j3
  t3 = j1 + j2 - j3
  t4 = j1 - m1
  t5 = j2 + m2

  tmin = np.max([0, t1, t2])
  tmax = np.min([t4, t5])

  t = np.arange(tmin, tmax+1)
  print('hi')
  print(t)


if __name__ == '__main__':
  w = wigner3j(2, 3, 1, 1, -2, -3)
  w = wigner3j(2,6,4,0,0,0)
  print(w)