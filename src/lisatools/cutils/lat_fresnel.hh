#ifndef __LAT_FRESNEL_HH__
#define __LAT_FRESNEL_HH__

// Fresnel integrals C and S, their auxiliary functions f and g, and the derivatives of f and g,
// shared by STFTFresnel (domains.cu) and the DFFR kernels. Ported from the standalone DFFR package
// (direct-fast-fresnel-response, commit 01530a5). tests/test_fresnel.py checks the numbers below.
//
// * Definitions (DLMF 7.2): C(x) + i S(x) = int_0^x exp(i pi t^2 / 2) dt and, for x > 0,
// *   f(x) = (1/2 - S) cos(pi x^2 / 2) - (1/2 - C) sin(pi x^2 / 2),
// *   g(x) = (1/2 - C) cos(pi x^2 / 2) + (1/2 - S) sin(pi x^2 / 2),
// * so that C + i S = sign(x) [ (1 + i)/2 - (g + i f)(|x|) exp(i pi x^2 / 2) ]. The derivatives
// * f' = -pi x g and g' = pi x f - 1 are exact.
//
// * Branches on |x|, with their errors against scipy.special.fresnel (C and S) and against
// * quadrature of the integral representations of f and g (DLMF 7.7):
// *   |x| <= 1.6     Maclaurin series of C and S (DLMF 7.6), 17 terms in x^4. The truncation is
// *                  7.6e-20 relative at the seam, so only rounding remains, 4e-16 absolute.
// *   1.6 < |x| < 8  minimax rational fits of f and g in 1/x^2 (degrees 5/5 and 6/6), made for
// *                  STFTFresnel in place of the approximations of Abramowitz and Stegun
// *                  (section 7.3, good to 2e-3). Largest relative errors 8.6e-10 (f) and 2.2e-9
// *                  (g and f'), both at the seam; medians 8e-11 and 2e-10. In C and S the fits
// *                  leave at most 1.6e-10 and 7e-11 absolute, also at the seam.
// *   |x| >= 8       five-term asymptotic series of f and g (DLMF 7.12) in w = 1 / (pi x^2). The
// *                  truncation is 6e-15 (f) and 1.3e-13 (g) relative at |x| = 8 and falls as w^10.
// * The linear-chirp closed form that STFTFresnel builds on these functions is exact, so these
// * errors set the accuracy floor of its pixel values.
// ! At the seam |x| = 1.6, C and S jump by 1.6e-10 and 7e-11, the error of the fits there; in a
// ! template mismatch that counts as its square over two, about 1e-20.
// ! Near |x| = 8, g' falls to -7.5e-5 while the absolute error of the fit stays near 2e-10, so
// ! there g' is good to 2.4e-6 relative only; its absolute error stays below 8.3e-10.
// ! CPU and CUDA builds may differ in the last bits: GCC and nvcc fuse multiply-adds in
// ! different places.
//
// * Speed: everything is inline, so that a kernel can fold constants and share subexpressions
// * across the many evaluations of one pixel. The series costs no division, the rational branch
// * two and the asymptotic branch one.

#include "gbt_global.h"  // CUDA_DEVICE

#include <cmath>

// * Seam between the Maclaurin series and the auxiliary form.
#define LAT_FRESNEL_SEAM 1.6
// * Seam between the rational fits of f and g and their asymptotic series.
#define LAT_FRESNEL_ASYMPTOTIC_SEAM 8.0
#define LAT_FRESNEL_INV_PI 0.318309886183790671537767526745028724

// Maclaurin coefficients of C and S as polynomials in y = x^4, highest power first:
//   C = x   sum_n (-1)^n (pi/2)^{2n}   y^n / ((2n)!   (4n+1))
//   S = x^3 sum_n (-1)^n (pi/2)^{2n+1} y^n / ((2n+1)! (4n+3))
// * Truncated at y^16, which is 7.6e-20 relative at the seam.
// Kept in the constant bank, not as literals: a DFMA reads a constant-bank operand
// directly, while a double literal costs two moves. Not const, or nvcc folds it back.
#if defined(__CUDACC__)
#define LAT_FRESNEL_CONSTANT_BANK __constant__
#else
#define LAT_FRESNEL_CONSTANT_BANK const
#endif
#define LAT_FRESNEL_SERIES_TERMS 17
static LAT_FRESNEL_CONSTANT_BANK double
    lat_fresnel_cosine_coefficients[LAT_FRESNEL_SERIES_TERMS] = {
    1.1034456863523295e-31,  -4.727226384742681e-29,  1.7837783103437512e-26,
    -5.877896118036892e-24,  1.6748476126215183e-21,  -4.079981449233878e-19,
    8.384729705118554e-17,   -1.4309189731715198e-14, 1.989685792418022e-12,
    -2.2022769254454663e-10, 1.8843499115272686e-08,  -1.2000972558600288e-06,
    5.4074133814083916e-05,  -1.6048831356425355e-03, 2.8185500877894225e-02,
    -2.4674011002723398e-01, 1.0};
static LAT_FRESNEL_CONSTANT_BANK double
    lat_fresnel_sine_coefficients[LAT_FRESNEL_SERIES_TERMS] = {
    5.0956014478106894e-33,  -2.3192836677213777e-30, 9.334382689020993e-28,
    -3.295271477907068e-25,  1.011069642466722e-22,   -2.6678713628413992e-20,
    5.980053239210405e-18,   -1.1223244787983955e-15, 1.7334102088874846e-13,
    -2.1574306805843444e-11, 2.1082121933214546e-09,  -1.564714450092211e-07,
    8.444272883545254e-06,   -3.121169423545792e-04,  7.244784204197004e-03,
    -9.228058535803518e-02,  5.235987755982989e-01};

// Auxiliary Fresnel functions f and g at |x| > 1.6, and their derivatives in |x| when both
// slope_f and slope_g are given; nullptr for either skips both.
// ! The fits hold only above the seam: callers test |x| > 1.6 first and pass |x|.
CUDA_DEVICE
inline void lat_fresnel_aux(double *aux_f_out, double *aux_g_out, double abs_argument,
                            double *slope_f = nullptr, double *slope_g = nullptr) {
  const bool with_derivatives = (slope_f != nullptr && slope_g != nullptr);

  if (abs_argument < LAT_FRESNEL_ASYMPTOTIC_SEAM) {
    // * Rational fits in 1 / x^2, coefficients ascending, evaluated by Horner.
    const double inverse_square = 1.0 / (abs_argument * abs_argument);

    double numerator_f = -2.8568630009945513e+00;
    numerator_f = numerator_f * inverse_square - 1.2714841072845186e+01;
    numerator_f = numerator_f * inverse_square + 9.9741235553347174e+00;
    numerator_f = numerator_f * inverse_square + 1.6530235844406874e+01;
    numerator_f = numerator_f * inverse_square + 5.6393336781284269e+00;
    numerator_f = numerator_f * inverse_square + 9.9999998788975186e-01;
    double denominator_f = -6.5764635236542199e+00;
    denominator_f = denominator_f * inverse_square - 8.5850028996594485e+00;
    denominator_f = denominator_f * inverse_square + 1.1683893414197762e+01;
    denominator_f = denominator_f * inverse_square + 1.6834326314502963e+01;
    denominator_f = denominator_f * inverse_square + 5.6393317184256810e+00;
    denominator_f = denominator_f * inverse_square + 1.0000000000000000e+00;
    double numerator_g = 3.2177684059265470e+00;
    numerator_g = numerator_g * inverse_square + 1.1588596586401721e+01;
    numerator_g = numerator_g * inverse_square - 2.1885461791121589e+01;
    numerator_g = numerator_g * inverse_square + 1.4001161175318936e+01;
    numerator_g = numerator_g * inverse_square + 2.2643028376343988e+01;
    numerator_g = numerator_g * inverse_square + 5.6691485084817055e+00;
    numerator_g = numerator_g * inverse_square + 9.9999994455231545e-01;
    double denominator_g = 1.9504389595032652e+01;
    denominator_g = denominator_g * inverse_square - 1.9482374004004097e+01;
    denominator_g = denominator_g * inverse_square + 5.8164191581105191e+00;
    denominator_g = denominator_g * inverse_square + 2.2588896377169348e+01;
    denominator_g = denominator_g * inverse_square + 2.4163564112837470e+01;
    denominator_g = denominator_g * inverse_square + 5.6691386346869024e+00;
    denominator_g = denominator_g * inverse_square + 1.0000000000000000e+00;

    // * One reciprocal serves both ratios, and 1 / (pi |x|) is (1/pi) |x| / x^2. Both
    // * denominators stay between 1.09 and 8.28 on this branch, so nothing is lost.
    const double reciprocal = 1.0 / (denominator_f * denominator_g);
    const double ratio_f = numerator_f * denominator_g * reciprocal;  // = pi |x| f
    const double ratio_g = numerator_g * denominator_f * reciprocal;
    const double inverse_pi_argument = LAT_FRESNEL_INV_PI * abs_argument * inverse_square;

    *aux_f_out = inverse_pi_argument * ratio_f;
    *aux_g_out = inverse_pi_argument * LAT_FRESNEL_INV_PI * inverse_square * ratio_g;

    if (with_derivatives) {
      // * g' = pi x f - 1 = ratio_f - 1, where ratio_f lies between 0.96 and 1. The subtraction
      // * is then exact, so g' keeps only the rounding of ratio_f, not that of f and pi x f.
      *slope_f = -LAT_FRESNEL_INV_PI * inverse_square * ratio_g;
      *slope_g = ratio_f - 1.0;
    }
  } else {
    // * Asymptotic series in w = 1 / (pi x^2):
    // * f = (1/(pi x)) sum (-1)^m (4m-1)!! w^{2m}, g = (1/(pi x)) sum (-1)^m (4m+1)!! w^{2m+1}.
    const double series_variable = 1.0 / (M_PI * abs_argument * abs_argument);
    const double series_square = series_variable * series_variable;
    const double sum_f =
        1.0 + series_square *
                  (-3.0 + series_square * (105.0 + series_square *
                                                       (-10395.0 + series_square * 2027025.0)));
    const double sum_g =
        series_variable *
        (1.0 + series_square *
                   (-15.0 + series_square *
                                (945.0 + series_square *
                                             (-135135.0 + series_square * 34459425.0))));
    const double inverse_pi_argument = series_variable * abs_argument;

    *aux_f_out = inverse_pi_argument * sum_f;
    *aux_g_out = inverse_pi_argument * sum_g;

    if (with_derivatives) {
      // ! Here g' = pi x f - 1 would cancel catastrophically: g' falls as -3 w^2 (5e-20 at
      // ! x = 5e4) while the subtraction rounds at 1e-16. The series is differentiated term by
      // ! term instead, and both derivative series collapse exactly to the polynomials below.
      const double slope_g_sum =
          3.0 + series_square *
                    (-105.0 + series_square *
                                  (10395.0 + series_square *
                                                 (-2027025.0 + series_square * 654729075.0)));
      *slope_f = -sum_g;
      *slope_g = -series_square * slope_g_sum;
    }
  }
}

// Maclaurin branch of C and S, valid for |x| <= 1.6. Both are odd in x.
// * A fixed Horner sum: no division and no exit that depends on x, so the threads of a warp
// * stay together.
// ! Separate from the full integral, so that a caller that already tested the seam
// ! does not inline the auxiliary branch a second time.
CUDA_DEVICE
inline void lat_fresnel_series(double *cosine_out, double *sine_out, double argument) {
  const double abs_argument = std::fabs(argument);
  const double argument_square = abs_argument * abs_argument;
  const double fourth_power = argument_square * argument_square;
  double cosine_poly = lat_fresnel_cosine_coefficients[0];
  double sine_poly = lat_fresnel_sine_coefficients[0];
#pragma unroll
  for (int power = 1; power < LAT_FRESNEL_SERIES_TERMS; power += 1) {
    cosine_poly = cosine_poly * fourth_power + lat_fresnel_cosine_coefficients[power];
    sine_poly = sine_poly * fourth_power + lat_fresnel_sine_coefficients[power];
  }
  const double cosine_value = abs_argument * cosine_poly;
  const double sine_value = abs_argument * argument_square * sine_poly;
  *cosine_out = (argument < 0.0) ? -cosine_value : cosine_value;
  *sine_out = (argument < 0.0) ? -sine_value : sine_value;
}

// C(x) and S(x) for any x. Above the seam they are assembled from f and g at the phase
// pi x^2 / 2, which grows without bound; the phase kernels avoid this form and use f and g.
// ! Rounding that phase costs about |x| eps / 2 absolute in C and S: 1.1e-13 at |x| = 1000.
CUDA_DEVICE
inline void lat_fresnel_integrals(double *cosine_out, double *sine_out, double argument) {
  const double abs_argument = std::fabs(argument);
  if (abs_argument <= LAT_FRESNEL_SEAM) {
    lat_fresnel_series(cosine_out, sine_out, argument);
    return;
  }
  const double half_pi_square = 0.5 * (M_PI * abs_argument) * abs_argument;
  const double cosine_phase = std::cos(half_pi_square);
  const double sine_phase = std::sin(half_pi_square);
  double aux_f, aux_g;
  lat_fresnel_aux(&aux_f, &aux_g, abs_argument);
  const double cosine_value = 0.5 + aux_f * sine_phase - aux_g * cosine_phase;
  const double sine_value = 0.5 - aux_f * cosine_phase - aux_g * sine_phase;
  *cosine_out = (argument < 0.0) ? -cosine_value : cosine_value;
  *sine_out = (argument < 0.0) ? -sine_value : sine_value;
}

#endif  // __LAT_FRESNEL_HH__
