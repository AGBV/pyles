#ifndef COUPLING_MATRIX_MULTIPLY_CPU_H
#define COUPLING_MATRIX_MULTIPLY_CPU_H
#include <stddef.h>

void coupling_matrix(size_t const particle_number, size_t const lmax, 
                    double const *re_x, double const *im_x,
                    double const *re_ab_table, double const *im_ab_table,
                    double const *associated_legendre_lookup, 
                    double const *re_spherical_hankel_lookup, double const *im_spherical_hankel_lookup,
                    double const *cos_dm_phi_loopup, double const *sin_dm_phi_loopup,
                    double *re_wx, double *im_wx);

double assoc_legendre_function_legacy(int const lmax, int const l, int const m, double const ct, double const st, double const *plm_coeffs);
void assoc_legendre_function_lookup_print(int const lmax, int const l, int const m, double const *plm_coeffs);
double spherical_hankel_lookup_legacy(int const lmax, int const p, double const r, double const *spjTable, double const rResol);
void coupling_matrix_legacy(int const particle_number, double const *particle_position,
                    int const lmax, double const *re_x, double const *im_x,
                    double const *re_ab_table, double const *im_ab_table,
                    double const *plm_coeffs,  double const r_resol,
                    double const *re_spherical_hankel_table, double const *im_spherical_hankel_table,
                    double *re_wx, double *im_wx);
void coupling_matrix_legacy_ab_free(
  int const particle_number, double const *particle_position,
  int const lmax, double const *re_x, double const *im_x,
  double const *re_ab_table, double const *im_ab_table,
  double const *plm_coeffs, double const r_resol,
  double const *re_spherical_hankel_table, double const *im_spherical_hankel_table,
  double *re_wx, double *im_wx);

#endif