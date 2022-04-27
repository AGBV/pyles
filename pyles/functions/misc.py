def jmult_max(num_part, lmax):
  return 2 * num_part * lmax * (lmax + 2)

def multi2single_index(jS,tau,l,m,lmax):
  return jS * 2 * lmax * (lmax+2) + (tau-1) * lmax * (lmax+2) + (l-1)*(l+1) + m + l