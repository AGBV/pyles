import logging

from scipy.special import spherical_jn, spherical_yn

def t_entry(tau, l, k_medium, k_sphere, radius, field_type = 'scattered'):
  """
  Computes an entry in the T Matrix for a given l, k, and tau
  
  **Note**: scipy.special has also derivative function. Why is it not the same?
  Example:
    Now:      djx  = x *  spherical_jn(l-1, x)  - l * jx
    Possible: djx  = spherical_jn(l, x, derivative=True)
  """

  m  = k_sphere / k_medium
  x  = k_medium * radius
  mx = k_sphere * radius

  jx  = spherical_jn(l, x)
  jmx = spherical_jn(l, mx)
  hx  = spherical_jn(l, x) + 1j * spherical_yn(l, x)

  djx  = x *  spherical_jn(l-1, x)  - l * jx
  djmx = mx * spherical_jn(l-1, mx) - l * jmx
  dhx  = x * (spherical_jn(l-1, x) + 1j * spherical_yn(l-1, x)) - l * hx

  match (field_type, tau):
    case ('scattered', 1):
      return -(jmx * djx - jx * djmx) / (jmx * dhx - hx * djmx) # -b
    case ('scattered', 2):
      return -(m**2 * jmx * djx - jx * djmx) / (m**2 * jmx * dhx - hx * djmx) # -a
    case ('internal', 1):
      return (jx * dhx - hx * djx) / (jmx * dhx - hx * djmx); # c
    case ('internal', 2):
      return (m * jx * dhx - m * hx * djx) / (m**2 * jmx * dhx - hx * djmx); # d
    case _:
      log = logging.getLogger('t_entry')
      log.warning('Not a valid field type provided. Returning None!')
      return None