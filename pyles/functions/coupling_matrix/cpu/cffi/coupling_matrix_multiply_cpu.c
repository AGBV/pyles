#include "coupling_matrix_multiply_cpu.h"

#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>

#if !defined(MAX)
#define MAX(A, B) ((A) > (B) ? (A) : (B))
#endif

#if !defined(MIN)
#define MIN(A, B) ((A) < (B) ? (A) : (B))
#endif

void coupling_matrix(
  size_t const particle_number, size_t const lmax, 
  double const *re_x, double const *im_x,
  double const *re_ab_table, double const *im_ab_table,
  double const *associated_legendre_lookup, 
  double const *re_spherical_hankel_lookup, double const *im_spherical_hankel_lookup,
  double const *cos_dm_phi_loopup, double const *sin_dm_phi_loopup,
  double *re_wx, double *im_wx)
{
  size_t nmax = 2 * lmax * (lmax + 2);
  long delta_tau;
  long delta_l;
  long delta_m;

  size_t x_idx, wx_idx = 0;
  size_t tau1_idx, l1_idx, m1_idx = 0;
  size_t tau2_idx, l2_idx, m2_idx = 0;

  double re_ab_p = 0;
  double im_ab_p = 0;

  double re_ab_p_h = 0;
  double im_ab_p_h = 0;
  
  double re_ab_p_h_eimp = 0;
  double im_ab_p_h_eimp = 0;

  for(size_t s1 = 0; s1 < particle_number; s1++) {
    for(size_t s2 = 0; s2 < particle_number; s2++) {
      if(s1 == s2) continue;
      for(size_t tau1 = 1; tau1 <= 2; tau1++) {
        tau1_idx = (tau1 - 1) * lmax * (lmax + 2);
        for(size_t l1 = 1; l1 <= lmax; l1++) {
          l1_idx = tau1_idx + (l1 - 1) * (l1 + 1) + l1;
          for(long m1 = -l1; m1 <= (long)l1; m1++) {
            m1_idx = l1_idx + m1;
            wx_idx = m1_idx * particle_number + s1;
            for(size_t tau2 = 1; tau2 <= 2; tau2++) {
              tau2_idx = (tau2 - 1) * lmax * (lmax + 2);
              for(size_t l2 = 1; l2 <= lmax; l2++) {
                l2_idx = tau2_idx + (l2 - 1) * (l2 + 1) + l2;
                for(long m2 = -l2; m2 <= (long)l2; m2++) {
                  m2_idx = l2_idx + m2;
                  x_idx = m2_idx * particle_number + s2;

                  delta_tau = abs(tau1 - tau2);
                  delta_l   = abs(l1   - l2);
                  delta_m   = abs(m1   - m2);

                  // delta_tau = (tau1 - tau2 > 0) ? (tau1 - tau2) : (tau2 - tau1);
                  // delta_l   = (l1   - l2   > 0) ? (l1   - l2)   : (l2   - l1);
                  // delta_m   = (m1   - m2   > 0) ? (m1   - m2)   : (m2   - m1);

                  for(long p = MAX(delta_m, delta_l + delta_tau); p <= (long)(l1 + l2); p++) {
                    long loop_idx = (p * (p + 1) / 2 + delta_m) * particle_number * particle_number + s1 * particle_number + s2;
                    re_ab_p = re_ab_table[m2_idx * nmax * (2 * lmax + 1) + m1_idx * (2 * lmax + 1) +  p] * associated_legendre_lookup[loop_idx];
                    im_ab_p = im_ab_table[m2_idx * nmax * (2 * lmax + 1) + m1_idx * (2 * lmax + 1) +  p] * associated_legendre_lookup[loop_idx];

                    loop_idx = p * particle_number * particle_number + s1 * particle_number + s2;
                    re_ab_p_h = re_ab_p * re_spherical_hankel_lookup[loop_idx] - im_ab_p * im_spherical_hankel_lookup[loop_idx];
                    im_ab_p_h = re_ab_p * im_spherical_hankel_lookup[loop_idx] + im_ab_p * re_spherical_hankel_lookup[loop_idx];

                    loop_idx = (m2 - m1 + 2 * lmax) * particle_number * particle_number + s1 * particle_number + s2;
                    re_ab_p_h_eimp = re_ab_p_h * cos_dm_phi_loopup[loop_idx] - im_ab_p_h * sin_dm_phi_loopup[loop_idx];
                    im_ab_p_h_eimp = re_ab_p_h * sin_dm_phi_loopup[loop_idx] + im_ab_p_h * cos_dm_phi_loopup[loop_idx];

                    re_wx[wx_idx] += re_ab_p_h_eimp * re_x[x_idx] - im_ab_p_h_eimp * im_x[x_idx];
                    im_wx[wx_idx] += re_ab_p_h_eimp * im_x[x_idx] + im_ab_p_h_eimp * re_x[x_idx];
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

double assoc_legendre_function_legacy(int const lmax, int const l, int const m, double const ct, double const st, double const *plm_coeffs)
{
	double Plm = 0.0f;
	int jj = 0;
	for (int lambda=l-m; lambda>=0; lambda-=2) {
		Plm += pow(st,m) * pow(ct,lambda) * plm_coeffs[jj*(2*lmax+1)*(2*lmax+1)+m*(2*lmax+1)+l];
		jj++;
	}
	return Plm;
}

void assoc_legendre_function_lookup_print(int const lmax, int const l, int const m, double const *plm_coeffs)
{
	int jj = 0;
  printf("% 4d | % 4d", l, m);
	for (int lambda=l-m; lambda>=0; lambda-=2) {
    printf(" % 15f", plm_coeffs[jj*(2*lmax+1)*(2*lmax+1)+m*(2*lmax+1)+l]);
    jj++;
	}
  printf("\n");
}

double spherical_hankel_lookup_legacy(int const lmax, int const p, double const r, double const *spjTable, double const rResol)
{
	double spj = 0;
	double rPos = r / rResol;
	int rIdx = (int)rPos;    			// points to table position -1, because for each p, the first entry with respect to r in the spjTable is copied 
	rPos -= rIdx; 							 	// (remainder of r / rResol) / rResol
	double rPos2 = pow(rPos,2);
	double rPos3 = pow(rPos,3);
	spj = ((-rPos3+2*rPos2-rPos) * spjTable[rIdx*(2*lmax+1)+p]
			+ (3*rPos3-5*rPos2+2) * spjTable[(rIdx+1)*(2*lmax+1)+p]
			+ (-3*rPos3+4*rPos2+rPos) * spjTable[(rIdx+2)*(2*lmax+1)+p]
			+ (rPos3-rPos2) * spjTable[(rIdx+3)*(2*lmax+1)+p])/2;
	return spj;
}

void coupling_matrix_legacy(
  int const particle_number, double const *particle_position,
  int const lmax, double const *re_x, double const *im_x,
  double const *re_ab_table, double const *im_ab_table,
  double const *plm_coeffs, double const r_resol,
  double const *re_spherical_hankel_table, double const *im_spherical_hankel_table,
  double *re_wx, double *im_wx)
{

  int delta_tau;
  int delta_l;
  int delta_m;

  int x_idx, wx_idx = 0;
  int tau1_idx, l1_idx, m1_idx = 0;
  int tau2_idx, l2_idx, m2_idx = 0;

  double re_ab_p, im_ab_p = 0;
  double re_ab_p_h, im_ab_p_h = 0;
  double re_ab_p_h_eimp, im_ab_p_h_eimp = 0;
  double re_incr, im_incr = 0;

  double x21, y21, z21 = 0;
  double r, cos_theta, sin_theta, phi = 0;
  double *re_h = (double *) malloc((2 * lmax + 1) * sizeof(double));
	double *im_h = (double *) malloc((2 * lmax + 1) * sizeof(double));
  double *p_p_dm = (double *) malloc((2 * lmax + 1) * (lmax + 1) * sizeof(double));
  double *cosmphi = (double *) malloc((4 * lmax + 1) * sizeof(double));
	double *sinmphi = (double *) malloc((4 * lmax + 1) * sizeof(double));

  // for (int p = 0; p <= 2 * (int)lmax; p++)	{
  //   for (int absdm = 0; absdm <= p; absdm++) {
  //     assoc_legendre_function_lookup_print(lmax, p, absdm, plm_coeffs);
  //   }
  // }

  // for(int p = 0; p < particle_number; p++) {
  //   printf("% .10e | % .10e | % .10e\n", particle_position[3 * p + 0], particle_position[3 * p + 1], particle_position[3 * p + 2]);
  // }

  // for(int p = 0; p < particle_number * 2 * lmax * (lmax + 2); p++) {
  //   printf("% .10e | % .10e\n", re_x[p], im_x[p]);
  // }

  // for(int p = 0; p < 9336; p++) {
  //   printf("% .10e | % .10e\n", re_ab_table[p], im_ab_table[p]);
  // }

  // for(int p = 0; p < (2 * lmax + 1) * (2 * lmax + 1) * (lmax + 1); p++) {
  //   printf("% .10e\n", plm_coeffs[p]);
  // }

  // for(int p = 0; p < 30852; p++) {
  //   printf("% .10e | % .10e\n", re_spherical_hankel_table[p], im_spherical_hankel_table[p]);
  // }

  int loop = 0;
  for(int s1 = 0; s1 < particle_number; s1++) {
    for(int s2 = 0; s2 < particle_number; s2++) {
      if(s1 == s2) continue;

      x21 = particle_position[3 * s1 + 0] - particle_position[3 * s2 + 0];
      y21 = particle_position[3 * s1 + 1] - particle_position[3 * s2 + 1];
      z21 = particle_position[3 * s1 + 2] - particle_position[3 * s2 + 2];

      r = sqrt(x21 * x21 + y21 * y21 + z21 * z21);
      cos_theta = z21 / r;
      sin_theta = sqrt(1 - cos_theta * cos_theta);
      phi = atan2(y21, x21);

      for (int p = 0; p <= 2 * lmax; p++)	{// precompute spherical Hankel functions and Legendre functions
        re_h[p] = spherical_hankel_lookup_legacy(lmax, p, r, re_spherical_hankel_table, r_resol);
        im_h[p] = spherical_hankel_lookup_legacy(lmax, p, r, im_spherical_hankel_table, r_resol);
        for (int absdm = 0; absdm <= p; absdm++) {
          p_p_dm[p * (p + 1) / 2 + absdm] = assoc_legendre_function_legacy(lmax, p, absdm, cos_theta, sin_theta, plm_coeffs);
        }
      }
      
      for (int dm = -2 * lmax; dm <= 2 * lmax; dm++) { // precompute exp(i(m-m')phi)
        cosmphi[dm+2*lmax] = cos(dm*phi);
        sinmphi[dm+2*lmax] = sin(dm*phi);
      }

      loop = 0;
      for(int tau1 = 1; tau1 <= 2; tau1++) {
        tau1_idx = (tau1 - 1) * lmax * (lmax + 2);
        for(int l1 = 1; l1 <= lmax; l1++) {
          l1_idx = tau1_idx + (l1 - 1) * (l1 + 1) + l1;
          for(int m1 = -l1; m1 <= l1; m1++) {
            m1_idx = l1_idx + m1;
            wx_idx = m1_idx * particle_number + s1;
            re_incr = 0;
            im_incr = 0;
            for(int tau2 = 1; tau2 <= 2; tau2++) {
              tau2_idx = (tau2 - 1) * lmax * (lmax + 2);
              for(int l2 = 1; l2 <= lmax; l2++) {
                l2_idx = tau2_idx + (l2 - 1) * (l2 + 1) + l2;
                for(int m2 = -l2; m2 <= l2; m2++) {
                  m2_idx = l2_idx + m2;
                  x_idx = m2_idx * particle_number + s2;

                  delta_tau = abs(tau1 - tau2);
                  delta_l   = abs(l1   - l2);
                  delta_m   = abs(m1   - m2);

                  for(int p = MAX(delta_m, delta_l + delta_tau); p <= l1 + l2; p++) {
                    re_ab_p = re_ab_table[loop] * p_p_dm[p * (p + 1) / 2 + delta_m];
                    im_ab_p = im_ab_table[loop] * p_p_dm[p * (p + 1) / 2 + delta_m];

                    re_ab_p_h = re_ab_p * re_h[p] - im_ab_p * im_h[p];
                    im_ab_p_h = re_ab_p * im_h[p] + im_ab_p * re_h[p];

                    re_ab_p_h_eimp = re_ab_p_h * cosmphi[m2 - m1 + 2 * lmax] - im_ab_p_h * sinmphi[m2 - m1 + 2 * lmax];
                    im_ab_p_h_eimp = re_ab_p_h * sinmphi[m2 - m1 + 2 * lmax] + im_ab_p_h * cosmphi[m2 - m1 + 2 * lmax];

                    re_incr += re_ab_p_h_eimp * re_x[x_idx] - im_ab_p_h_eimp * im_x[x_idx];
                    im_incr += re_ab_p_h_eimp * im_x[x_idx] + im_ab_p_h_eimp * re_x[x_idx];

                    loop++;
                  } // p
                } // m2
              } // l2
            } // tau2
            re_wx[wx_idx] += re_incr;
            im_wx[wx_idx] += im_incr;
            // re_wx[wx_idx] += 1;
            // im_wx[wx_idx] += 1;
          } // m1
        } // l1
      } // tau1
    }
  }
  free(re_h);
	free(im_h);
  free(p_p_dm);
  free(cosmphi);
	free(sinmphi);

  printf("Loop: %d\n", loop);
  printf("re_wx: %f\n", re_wx[0]);
  printf("im_wx: %f\n", im_wx[0]);
}

void coupling_matrix_legacy_ab_free(
  int const particle_number, double const *particle_position,
  int const lmax, double const *re_x, double const *im_x,
  double const *re_ab_table, double const *im_ab_table,
  double const *plm_coeffs, double const r_resol,
  double const *re_spherical_hankel_table, double const *im_spherical_hankel_table,
  double *re_wx, double *im_wx)
{
  int nmax = 2 * lmax * (lmax + 2);

  int delta_tau;
  int delta_l;
  int delta_m;

  int x_idx, wx_idx = 0;
  int tau1_idx, l1_idx, m1_idx = 0;
  int tau2_idx, l2_idx, m2_idx = 0;

  double re_ab_p, im_ab_p = 0;
  double re_ab_p_h, im_ab_p_h = 0;
  double re_ab_p_h_eimp, im_ab_p_h_eimp = 0;

  double x21, y21, z21 = 0;
  double r, cos_theta, sin_theta, phi = 0;
  double *re_h = (double *) malloc((2 * lmax + 1) * sizeof(double));
	double *im_h = (double *) malloc((2 * lmax + 1) * sizeof(double));
  double *p_p_dm = (double *) malloc((2 * lmax + 1) * (lmax + 1) * sizeof(double));
  double *cosmphi = (double *) malloc((4 * lmax + 1) * sizeof(double));
	double *sinmphi = (double *) malloc((4 * lmax + 1) * sizeof(double));

  int loop = 0;
  for(int s1 = 0; s1 < particle_number; s1++) {
    for(int s2 = 0; s2 < particle_number; s2++) {
      if(s1 == s2) continue;

      x21 = particle_position[3 * s1 + 0] - particle_position[3 * s2 + 0];
      y21 = particle_position[3 * s1 + 1] - particle_position[3 * s2 + 1];
      z21 = particle_position[3 * s1 + 2] - particle_position[3 * s2 + 2];

      r = sqrt(x21 * x21 + y21 * y21 + z21 * z21);
      cos_theta = z21 / r;
      sin_theta = sqrt(1 - cos_theta * cos_theta);
      phi = atan2(y21, x21);

      for (int p = 0; p <= 2 * lmax; p++)	{// precompute spherical Hankel functions and Legendre functions
        re_h[p] = spherical_hankel_lookup_legacy(lmax, p, r, re_spherical_hankel_table, r_resol);
        im_h[p] = spherical_hankel_lookup_legacy(lmax, p, r, im_spherical_hankel_table, r_resol);
        for (int absdm = 0; absdm <= p; absdm++) {
          p_p_dm[p * (p + 1) / 2 + absdm] = assoc_legendre_function_legacy(lmax, p, absdm, cos_theta, sin_theta, plm_coeffs);
        }
      }
      
      for (int dm = -2 * lmax; dm <= 2 * lmax; dm++) { // precompute exp(i(m-m')phi)
        cosmphi[dm+2*lmax] = cos(dm*phi);
        sinmphi[dm+2*lmax] = sin(dm*phi);
      }

      loop = 0;
      for(int tau1 = 1; tau1 <= 2; tau1++) {
        tau1_idx = (tau1 - 1) * lmax * (lmax + 2);
        for(int l1 = 1; l1 <= lmax; l1++) {
          l1_idx = tau1_idx + (l1 - 1) * (l1 + 1) + l1;
          for(int m1 = -l1; m1 <= l1; m1++) {
            m1_idx = l1_idx + m1;
            wx_idx = m1_idx * particle_number + s1;

            for(int tau2 = 1; tau2 <= 2; tau2++) {
              tau2_idx = (tau2 - 1) * lmax * (lmax + 2);
              for(int l2 = 1; l2 <= lmax; l2++) {
                l2_idx = tau2_idx + (l2 - 1) * (l2 + 1) + l2;
                for(int m2 = -l2; m2 <= l2; m2++) {
                  m2_idx = l2_idx + m2;
                  x_idx = m2_idx * particle_number + s2;

                  delta_tau = abs(tau1 - tau2);
                  delta_l   = abs(l1   - l2);
                  delta_m   = abs(m1   - m2);

                  for(int p = MAX(delta_m, delta_l + delta_tau); p <= l1 + l2; p++) {
                    loop = m2_idx + m1_idx * nmax +  p * nmax * nmax;
                    re_ab_p = re_ab_table[loop] * p_p_dm[p * (p + 1) / 2 + delta_m];
                    im_ab_p = im_ab_table[loop] * p_p_dm[p * (p + 1) / 2 + delta_m];

                    re_ab_p_h = re_ab_p * re_h[p] - im_ab_p * im_h[p];
                    im_ab_p_h = re_ab_p * im_h[p] + im_ab_p * re_h[p];

                    re_ab_p_h_eimp = re_ab_p_h * cosmphi[m2 - m1 + 2 * lmax] - im_ab_p_h * sinmphi[m2 - m1 + 2 * lmax];
                    im_ab_p_h_eimp = re_ab_p_h * sinmphi[m2 - m1 + 2 * lmax] + im_ab_p_h * cosmphi[m2 - m1 + 2 * lmax];

                    re_wx[wx_idx] += re_ab_p_h_eimp * re_x[x_idx] - im_ab_p_h_eimp * im_x[x_idx];
                    im_wx[wx_idx] += re_ab_p_h_eimp * im_x[x_idx] + im_ab_p_h_eimp * re_x[x_idx];

                    // re_wx[wx_idx] += re_ab_p_h_eimp;
                    // im_wx[wx_idx] += im_ab_p_h_eimp;
                  } // p
                } // m2
              } // l2
            } // tau2
          } // m1
        } // l1
      } // tau1
    }
  }
  free(re_h);
	free(im_h);
  free(p_p_dm);
  free(cosmphi);
	free(sinmphi);
}
