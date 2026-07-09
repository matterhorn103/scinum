//! Implementation of the `num_traits::Float` trait for [`SciDecimal`].

use std::num::FpCategory;

use num_traits::{Float, Inv, Pow};

use crate::{
    RoundingMode, SciDecimal, SciNum,
    scicast::{SciCast, SciCastFrom},
};

#[allow(unused_variables)] // TODO Remove once all methods are implemented
impl Float for SciDecimal {
    #[inline]
    fn nan() -> Self {
        Self::NAN
    }

    #[inline]
    fn infinity() -> Self {
        Self::INFINITY
    }

    #[inline]
    fn neg_infinity() -> Self {
        Self::NEG_INFINITY
    }

    #[inline]
    fn neg_zero() -> Self {
        Self::NEG_ZERO
    }

    fn min_value() -> Self {
        Self::MIN
    }

    fn min_positive_value() -> Self {
        Self::MIN_POSITIVE
    }

    fn max_value() -> Self {
        Self::MAX
    }

    #[inline]
    fn is_nan(self) -> bool {
        self.nan_bit()
    }

    #[inline]
    fn is_infinite(self) -> bool {
        // The NaN flag overrides the infinity flag i.e. if a `SciDecimal` has
        // both `1` then it is considered a NaN and therefore *not infinite*.
        // We therefore have to compare against both bits
        self.flags & 0xC0 == 0x70
    }

    #[inline]
    fn is_finite(self) -> bool {
        self.flags & 0xC0 == 0
    }

    #[inline]
    fn is_normal(self) -> bool {
        self.is_finite() && self.significand != 0
    }

    #[inline]
    fn classify(self) -> FpCategory {
        if self.nan_bit() {
            FpCategory::Nan
        } else if self.inf_bit() {
            FpCategory::Infinite
        } else if self.significand == 0 {
            FpCategory::Zero
        } else {
            FpCategory::Normal
        }
    }

    fn floor(self) -> Self {
        self.round_precision(0, RoundingMode::Floor)
    }

    fn ceil(self) -> Self {
        self.round_precision(0, RoundingMode::Ceiling)
    }

    fn round(self) -> Self {
        self.round_precision(0, RoundingMode::HalfUp)
    }

    fn trunc(self) -> Self {
        self.round_precision(0, RoundingMode::Down)
    }

    fn fract(self) -> Self {
        self - self.trunc()
    }

    fn abs(self) -> Self {
        if self.is_nan() {
            Self::NAN
        } else {
            Self {
                flags: self.flags & !0x01,
                ..self
            }
        }
    }

    fn signum(self) -> Self {
        if self.is_nan() {
            Self::NAN
        } else if self.sign_bit() {
            Self::NEG_ONE
        } else {
            Self::ONE
        }
    }

    #[inline]
    //#[must_use]
    fn is_sign_positive(self) -> bool {
        !self.sign_bit()
    }

    #[inline]
    //#[must_use]
    fn is_sign_negative(self) -> bool {
        self.sign_bit()
    }

    /// Fused multiply-add. Computes (self * a) + b with only one rounding error,
    /// yielding a more accurate result than an unfused multiply-add.
    fn mul_add(self, a: Self, b: Self) -> Self {
        todo!()
    }

    /// Takes the reciprocoal (inverse) of the number, `1/x`.
    #[inline]
    fn recip(self) -> Self {
        self.inv()
    }

    /// Raises the number to an integer power.
    fn powi(self, n: i32) -> Self {
        let exact = if n <= i8::MAX.into() && n >= i8::MIN.into() {
            self.unbounded_powi(
                n.try_into()
                    .expect("n has already been checked and should fit into even an i8"),
            )
        } else {
            self.unbounded_powf(n.into())
        };
        let result = if self.is_exact() {
            exact
        } else if !exact.is_finite() {
            // Uncertainty is infinity or NaN by definition anyway
            exact
        } else {
            let uncertainty = (self.relative_uncertainty() * n.into()) * exact.abs();
            exact.with_uncertainty(uncertainty)
        };
        if result.significand > Self::MAX_SIGNIFICAND {
            result.round_sf(16, RoundingMode::HalfUp)
        } else {
            result
        }
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        self.pow(n)
    }

    fn sqrt(self) -> Self {
        todo!()
    }

    fn cbrt(self) -> Self {
        todo!()
    }

    fn exp(self) -> Self {
        let exact = Self::E_PRECISE.pow(self);
        if self.is_exact() {
            exact
        } else {
            // For C = a e^bA, σ_c = |C|×|b × σ_A|
            // If a = b = 1,   σ_c = |C|×|σ_A|
            let uncertainty = exact.abs() * self.uncertainty();
            exact.with_uncertainty(uncertainty)
        }
    }

    fn exp2(self) -> Self {
        let exact = Self::TWO.pow(self);
        if self.is_exact() {
            exact
        } else {
            // For C = a^bA,    σ_c = |C|×|b × ln(a) × σ_A|
            // If a = 2, b = 1, σ_c = |C|×|ln(2) × σ_A|
            let uncertainty = exact.abs() * Self::LN_2_PRECISE * self.uncertainty();
            exact.with_uncertainty(uncertainty)
        }
    }

    // For logarithms, take advantage of logₐx = log₂x/log₂a = log₁₀x/log₁₀a
    // as for the binary integer significand, log₂ will be efficient, but for
    // the base 10 exponent log₁₀ is ideal, and for the common bases we have
    // precomputed values for the divisors as associated constants
    // Thus, as log(xy) = log(x) + log(y),
    // we can calculate the result with base a as:
    //       logₐ(m⋅10^n) = log₂(m⋅10^n) / log₂(a)
    // and since
    //       log₂(m⋅10^n) = log₂(m) + log₂(10^n)
    //                    = log₂(m) + (n⋅log₂(10))
    // then
    //       logₐ(m⋅10^n) = (log₂(m)/log₂(a)) + ((n⋅log₂(10)) / log₂(a))
    //
    // Increase the precision of m by multiplying it by 2^k (where k is the number
    // of trailing zeros):
    //            log₂(m) = log₂(m⋅2^k/2^k)
    //                    = log₂(m⋅2^k) - log₂(2^k)
    //                    = log₂(m⋅2^k) - k
    // therefore
    //       logₐ(m⋅10^n) = ((log₂(m⋅2^k) - k)/log₂(a)) + ((n⋅log₂(10))/log₂(a))
    // But that doesn't improve the precision of log(m) because the fractional
    // part remains identical...

    fn log(self, base: Self) -> Self {
        todo!()
    }

    fn log2(self) -> Self {
        todo!()
    }

    fn ln(self) -> Self {
        if !self.is_finite() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        let number =
            rust_decimal::MathematicalOps::ln(&rust_decimal::Decimal::cast_from(self.number()));
        if self.is_exact() {
            number.cast()
        } else {
            let uncertainty = self.relative_uncertainty().abs();
            number.cast().with_uncertainty(uncertainty)
        }
    }

    fn log10(self) -> Self {
        if !self.is_finite() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        let number =
            rust_decimal::MathematicalOps::log10(&rust_decimal::Decimal::cast_from(self.number()));
        if self.is_exact() {
            number.cast()
        } else {
            let uncertainty = (rust_decimal::Decimal::cast_from(self.uncertainty())
                / (rust_decimal::MathematicalOps::ln(&rust_decimal::Decimal::TEN)
                    * rust_decimal::Decimal::cast_from(self.number())))
            .abs();
            number.cast().with_uncertainty(uncertainty.cast())
        }
    }

    /// Returns the maximum of the two numbers.
    ///
    /// If the two are equal, returns `self` (relevant for the uncertainty).
    fn max(self, other: Self) -> Self {
        if other > self { other } else { self }
    }

    /// Returns the minimum of the two numbers.
    ///
    /// If the two are equal, returns `self` (relevant for the uncertainty).
    fn min(self, other: Self) -> Self {
        if other < self { other } else { self }
    }

    fn abs_sub(self, other: Self) -> Self {
        todo!()
    }

    fn hypot(self, other: Self) -> Self {
        todo!()
    }

    fn sin(self) -> Self {
        todo!()
    }

    fn cos(self) -> Self {
        todo!()
    }

    fn tan(self) -> Self {
        todo!()
    }

    fn asin(self) -> Self {
        todo!()
    }

    fn acos(self) -> Self {
        todo!()
    }

    fn atan(self) -> Self {
        todo!()
    }

    fn atan2(self, other: Self) -> Self {
        todo!()
    }

    fn sin_cos(self) -> (Self, Self) {
        todo!()
    }

    fn exp_m1(self) -> Self {
        todo!()
    }

    fn ln_1p(self) -> Self {
        todo!()
    }

    fn sinh(self) -> Self {
        todo!()
    }

    fn cosh(self) -> Self {
        todo!()
    }

    fn tanh(self) -> Self {
        todo!()
    }

    fn asinh(self) -> Self {
        todo!()
    }

    fn acosh(self) -> Self {
        todo!()
    }

    fn atanh(self) -> Self {
        todo!()
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        todo!()
    }
}
/*
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn natural_log() {
        let n1 = SciDecimal::new_with_uncertainty(20, 2, 0);
        let n2 = SciDecimal::new_with_uncertainty(30, 5, 0);
        let ratio = n1 / n2;
        let result = ratio.ln();
        assert_eq!(
            rust_decimal::Decimal::cast_from(result.uncertainty()).round_dp(5),
            rust_decimal_macros::dec!(0.194365063161).round_dp(5)
        );
    }

    #[test]
    fn log_base10() {
        let n1 = SciDecimal::new_with_uncertainty(20, 2, 0);
        let n2 = SciDecimal::new_with_uncertainty(30, 5, 0);
        let ratio = n1 / n2;
        let result = ratio.log10();
        assert_eq!(
            rust_decimal::Decimal::cast_from(result.uncertainty()).round_dp(5),
            rust_decimal_macros::dec!(0.08441167440582).round_dp(5)
        );
    }

    #[test]
    fn exponential() {
        let n1 = SciDecimal::new_with_uncertainty(20, 2, 0);
        let n2 = SciDecimal::new_with_uncertainty(30, 5, 0);
        let ratio = n1 / n2;
        let result = ratio.exp();
        assert_eq!(
            rust_decimal::Decimal::cast_from(result.uncertainty()).round_dp(5),
            rust_decimal_macros::dec!(0.25238096660761).round_dp(5)
        );
    }
}
*/
