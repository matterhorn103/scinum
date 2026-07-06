//! Implementation of the `num_traits::Float` trait for [`SciFloat`].

use std::num::FpCategory;

use num_traits::Float;
use std::f64::consts::PI;

use crate::{SciFloat, SciNum};

impl Float for SciFloat {
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
        todo!()
    }

    fn min_positive_value() -> Self {
        todo!()
    }

    fn max_value() -> Self {
        todo!()
    }

    #[inline]
    fn is_nan(self) -> bool {
        self.number.is_nan()
    }

    #[inline]
    fn is_infinite(self) -> bool {
        self.number.is_infinite()
    }

    #[inline]
    fn is_finite(self) -> bool {
        self.number.is_finite()
    }

    #[inline]
    fn is_normal(self) -> bool {
        self.number.is_normal()
    }

    #[inline]
    fn classify(self) -> FpCategory {
        self.number.classify()
    }

    fn floor(self) -> Self {
        todo!()
    }

    fn ceil(self) -> Self {
        todo!()
    }

    fn round(self) -> Self {
        todo!()
    }

    fn trunc(self) -> Self {
        todo!()
    }

    fn fract(self) -> Self {
        todo!()
    }

    fn abs(self) -> Self {
        Self {
            number: self.number.abs(),
            uncertainty: self.uncertainty,
        }
    }

    #[inline]
    fn signum(self) -> Self {
        self.number.signum().into()
    }

    #[inline]
    fn is_sign_positive(self) -> bool {
        self.number.is_sign_positive()
    }

    #[inline]
    fn is_sign_negative(self) -> bool {
        self.number.is_sign_negative()
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        todo!()
    }

    fn recip(self) -> Self {
        todo!()
    }

    /// Raise the `SciFloat` to an integer power.
    #[inline]
    fn powi(self, n: i32) -> Self {
        if !self.is_finite() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        let result = self.number.powi(n);
        if self.is_exact() {
            SciFloat::new(result)
        } else {
            let uncertainty = self.relative_uncertainty() * result * n as f64;
            SciFloat::new_with_uncertainty(result, uncertainty)
        }
    }

    fn powf(self, n: Self) -> Self {
        todo!()
    }

    #[inline]
    fn sqrt(self) -> Self {
        if !self.is_finite() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        let result = self.number.sqrt();
        if self.is_exact() {
            SciFloat::new(result)
        } else {
            let uncertainty = (self.relative_uncertainty() * result) / (2.0);
            SciFloat::new_with_uncertainty(result, uncertainty)
        }
    }

    fn cbrt(self) -> Self {
        if !self.is_finite() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        let result = self.number.cbrt();
        if self.is_exact() {
            SciFloat::new(result)
        } else {
            let uncertainty = (self.relative_uncertainty() * result) / (3.0);
            SciFloat::new_with_uncertainty(result, uncertainty)
        }
    }

    #[inline]
    fn exp(self) -> Self {
        if !self.is_finite() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        let result = self.number.exp();
        if self.is_exact() {
            SciFloat::new(result)
        } else {
            let uncertainty = result.abs() * self.uncertainty;
            SciFloat::new_with_uncertainty(result, uncertainty)
        }
    }

    fn exp2(self) -> Self {
        todo!()
    }

    fn ln(self) -> Self {
        if !self.is_finite() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        let result = self.number.ln();
        if self.is_exact() {
            SciFloat::new(result)
        } else {
            let uncertainty = self.relative_uncertainty();
            SciFloat::new_with_uncertainty(result, uncertainty)
        }
    }

    fn log(self, base: Self) -> Self {
        todo!()
    }

    fn log2(self) -> Self {
        todo!()
    }

    fn log10(self) -> Self {
        if !self.is_finite() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        let result = self.number.log10();
        if self.is_exact() {
            Self::new(result)
        } else {
            let uncertainty = (self.relative_uncertainty() / (10.0_f64).ln());
            Self::new_with_uncertainty(result, uncertainty)
        }
    }

    fn to_degrees(self) -> Self {
        if !self.is_finite() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        let result = self.number * (180.0 / PI);
        if self.is_exact() {
            Self::new(result)
        } else {
            let uncertainty = self.uncertainty * (180.0 / PI);
            Self::new_with_uncertainty(result, uncertainty)
        }
    }

    fn to_radians(self) -> Self {
        if !self.is_finite() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        let result = self.number * (PI / 180.0);
        if self.is_exact() {
            Self::new(result)
        } else {
            let uncertainty = self.uncertainty * (PI / 180.0);
            Self::new_with_uncertainty(result, uncertainty)
        }
    }

    fn max(self, other: Self) -> Self {
        match self > other {
            true => self,
            false => other,
        }
    }

    fn min(self, other: Self) -> Self {
        match self < other {
            true => self,
            false => other,
        }
    }

    fn abs_sub(self, other: Self) -> Self {
        todo!()
    }

    fn hypot(self, other: Self) -> Self {
        if !(self.is_finite() && other.is_finite()) {
            todo!("Special values are not yet handled correctly by this method!")
        }
        let result = self.number.hypot(other.number);
        if self.is_exact() {
            Self::new(result)
        } else {
            let uncertainty = (((self.number * self.uncertainty) / result).powi(2)
                + ((other.number * other.uncertainty) / result).powi(2))
            .sqrt();
            Self::new_with_uncertainty(result, uncertainty)
        }
    }

    fn sin(self) -> Self {
        if !self.is_finite() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        let result = self.number.sin();
        if self.is_exact() {
            Self::new(result)
        } else {
            let uncertainty = (self.number.cos() * self.uncertainty).abs();
            Self::new_with_uncertainty(result, uncertainty)
        }
    }

    fn cos(self) -> Self {
        if !self.is_finite() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        let result = self.number.cos();
        if self.is_exact() {
            Self::new(result)
        } else {
            let uncertainty = (self.number.sin() * self.uncertainty).abs();
            Self::new_with_uncertainty(result, uncertainty)
        }
    }

    fn tan(self) -> Self {
        if !self.is_finite() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        let result = self.number.tan();
        if self.is_exact() {
            Self::new(result)
        } else {
            let uncertainty = ((1_f64 / (self.number.cos().powi(2))) * self.uncertainty).abs();
            Self::new_with_uncertainty(result, uncertainty)
        }
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
        (self.sin(), self.cos())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn powi_exact() {
        let n = SciFloat::new(4.0);
        assert_eq!(n.powi(2), SciFloat::new(16.));
        assert_eq!(n.powi(3), SciFloat::new(64.));
        assert_eq!(n.powi(-1), SciFloat::new(0.25));
        assert_eq!(n.powi(-2), SciFloat::new(0.0625));
    }

    #[test]
    fn powi_with_uncertainty() {
        let n = SciFloat::new_with_uncertainty(20.0, 2.0);
        let result = n.powi(2);
        assert_eq!(result.number(), 400.0);
        assert_eq!(result.uncertainty(), 80.0);
    }

    #[test]
    fn natural_log() {
        let n1 = SciFloat::new_with_uncertainty(20.0, 2.0);
        let n2 = SciFloat::new_with_uncertainty(30.0, 5.0);
        let ratio = n1 / n2;
        let result = ratio.ln();
        assert_eq!(result.uncertainty(), 0.19436506316151);
    }

    #[test]
    fn log_base10() {
        let n1 = SciFloat::new_with_uncertainty(20.0, 2.0);
        let n2 = SciFloat::new_with_uncertainty(30.0, 5.0);
        let ratio = n1 / n2;
        let result = ratio.log10();
        assert_eq!(result.uncertainty(), 0.08441167440582079);
    }

    #[test]
    fn exponential() {
        let n1 = SciFloat::new_with_uncertainty(20.0, 2.0);
        let n2 = SciFloat::new_with_uncertainty(30.0, 5.0);
        let ratio = n1 / n2;
        let result = ratio.exp();
        assert_eq!(result.uncertainty(), 0.2523809666076101);
    }
}
