// SPDX-FileCopyrightText: 2025 Matthew Milner <matterhorn103@proton.me>
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    fmt::Display,
    num::{FpCategory, ParseFloatError},
    ops::{Add, Div, Mul, Neg, Rem, Sub},
    str::FromStr,
};

use num_traits::{Float, FloatConst, Inv, Num, One, Pow, Zero};

use crate::{RoundingMode, SciDecimal, SciNum, error::SciNumError};

/// A binary floating point number with an associated uncertainty.
///
/// Wraps the native `f64` type.
#[derive(Debug, Clone, Copy, serde_with::DeserializeFromStr, serde_with::SerializeDisplay)]
pub struct SciFloat {
    number: f64,
    uncertainty: f64,
}

// Constants that don't belong to specific traits
impl SciFloat {
    /// The lowest supported number.
    pub const MIN: SciFloat = SciFloat {
        number: f64::MIN,
        uncertainty: 0.0,
    };

    /// The highest supported number.
    pub const MAX: SciFloat = SciFloat {
        number: f64::MAX,
        uncertainty: 0.0,
    };

    /// The `SciFloat` representation of `NaN`, "not a number".
    pub const NAN: SciFloat = SciFloat {
        number: f64::NAN,
        uncertainty: 0.0,
    };

    /// The `SciFloat` representation of positive infinity.
    pub const INFINITY: SciFloat = SciFloat {
        number: f64::INFINITY,
        uncertainty: 0.0,
    };

    /// The `SciFloat` representation of negative infinity.
    pub const NEG_INFINITY: SciFloat = SciFloat {
        number: f64::NEG_INFINITY,
        uncertainty: 0.0,
    };

    /// The `SciFloat` representation of negative zero.
    pub const NEG_ZERO: SciFloat = SciFloat {
        number: -0.0,
        uncertainty: 0.0,
    };
}

impl Zero for SciFloat {
    #[inline]
    fn zero() -> Self {
        Self::ZERO
    }

    /// Returns true if the `SciFloat` is equal to zero, regardless of any
    /// uncertainty.
    fn is_zero(&self) -> bool {
        self.number.is_zero()
    }
}

impl One for SciFloat {
    #[inline]
    fn one() -> Self {
        Self::ONE
    }
}

// Instantiation
impl SciFloat {
    pub fn new(number: f64) -> Self {
        Self {
            number,
            uncertainty: 0.0,
        }
    }

    pub fn new_with_uncertainty(number: f64, uncertainty: f64) -> Self {
        Self {
            number,
            uncertainty,
        }
    }
}

// Precision, figures, and rounding
impl SciFloat {
    /// Removes significant figures from the significand to give a new `SciFloat`
    /// with the specified number.
    ///
    /// Equivalent to rounding towards zero.
    ///
    /// The uncertainty of the `SciFloat` is left unchanged.
    pub fn trunc_sf(mut self, sf: u8) -> Self {
        let scale = 10_f64.powi(sf as i32);
        let mut value = self.number;
        // If integer
        if value.fract() != 0.0 {
            self.number = (self.number * scale).trunc() / scale;
            self
        } else {
            let mut exponent: i32 = 0;
            while value.fract() != value {
                value /= 10.0;
                exponent += 1;
            }
            self.number = (value * scale).trunc() / 10_f64.powi(sf as i32 - exponent);
            self
        }
    }
}

impl SciNum for SciFloat {
    type Number = f64;

    const ZERO: Self = SciFloat {
        number: 0.0,
        uncertainty: 0.0,
    };

    const ONE: Self = SciFloat {
        number: 1.0,
        uncertainty: 0.0,
    };

    /// Returns the number as an `f64`.
    #[inline]
    fn number(&self) -> f64 {
        self.number
    }

    /// Returns the absolute uncertainty as an `f64`.
    ///
    /// The uncertainty is always positive.
    ///
    /// An infinity always has an uncertainty of (positive) infinity, and `NaN`
    /// always has an uncertainty of `NaN`.
    #[inline]
    fn uncertainty(&self) -> f64 {
        if self.is_nan() {
            f64::NAN
        } else if self.is_infinite() {
            f64::INFINITY
        } else {
            self.uncertainty.abs()
        }
    }

    /// Returns the relative uncertainty as an `f64`.
    ///
    /// The relative uncertainty is always positive.
    #[inline]
    fn relative_uncertainty(&self) -> f64 {
        self.uncertainty / self.number.abs()
    }

    /// Creates a new `SciFloat` with the same number but the provided
    /// uncertainty.
    #[inline]
    fn with_uncertainty(self, uncertainty: f64) -> Self {
        Self {
            number: self.number,
            uncertainty: uncertainty.abs(),
        }
    }

    /// Returns true if the `SciFloat` has an uncertainty of zero.
    #[inline]
    fn is_exact(&self) -> bool {
        if !self.is_finite() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        self.uncertainty == 0.0
    }

    fn round_precision(self, prec: i16, mode: RoundingMode) -> Self {
        todo!()
    }

    fn round_dp(self, dp: u16, mode: RoundingMode) -> Self {
        todo!()
    }

    fn round_sf(self, sf: u8, mode: RoundingMode) -> Self {
        todo!()
    }

    fn round_match_uncertainty(self, mode: RoundingMode) -> Self {
        todo!()
    }

    fn round_match_uncertainty_sf(self, sf: u8, mode: RoundingMode) -> Self {
        todo!()
    }

    fn round_uncertainty_precision(self, prec: i16, mode: RoundingMode) -> Self {
        todo!()
    }

    fn round_uncertainty_dp(self, dp: u16, mode: RoundingMode) -> Self {
        todo!()
    }

    fn round_uncertainty_sf(self, sf: u8, mode: RoundingMode) -> Self {
        todo!()
    }

    fn round_uncertainty_match_number(self, mode: RoundingMode) -> Self {
        todo!()
    }
}

impl Num for SciFloat {
    type FromStrRadixErr = <f64 as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(Self {
            number: f64::from_str_radix(str, radix)?,
            uncertainty: 0.0,
        })
    }
}

//impl Float for SciFloat {
#[allow(unused)]
impl SciFloat {
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
        let result = self.number * (180.0 / f64::PI());
        if self.is_exact() {
            Self::new(result)
        } else {
            let uncertainty = self.uncertainty * (180.0 / f64::PI());
            Self::new_with_uncertainty(result, uncertainty)
        }
    }

    fn to_radians(self) -> Self {
        if !self.is_finite() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        let result = self.number * (f64::PI() / 180.0);
        if self.is_exact() {
            Self::new(result)
        } else {
            let uncertainty = self.uncertainty * (f64::PI() / 180.0);
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

impl PartialEq for SciFloat {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.number == other.number
    }
}

impl Eq for SciFloat {}

impl PartialOrd for SciFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.number.partial_cmp(&other.number)
    }
}

impl Add for SciFloat {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let number = self.number + rhs.number;
        let uncertainty = (self.uncertainty.powi(2) + rhs.uncertainty.powi(2)).sqrt();
        Self {
            number,
            uncertainty,
        }
    }
}

impl Sub for SciFloat {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let number = self.number - rhs.number;
        let uncertainty = (self.uncertainty.powi(2) + rhs.uncertainty.powi(2)).sqrt();
        Self {
            number,
            uncertainty,
        }
    }
}

impl Mul for SciFloat {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let number = self.number * rhs.number;
        let uncertainty =
            (self.relative_uncertainty().powi(2) + rhs.relative_uncertainty().powi(2)).sqrt()
                * number.abs();
        Self {
            number,
            uncertainty,
        }
    }
}

impl Div for SciFloat {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        let number = self.number / rhs.number;
        let uncertainty =
            (self.relative_uncertainty().powi(2) + rhs.relative_uncertainty().powi(2)).sqrt()
                * number.abs();
        Self {
            number,
            uncertainty,
        }
    }
}

impl Rem for SciFloat {
    type Output = Self;

    /// Performs the `%` operation.
    ///
    /// NOTE: Uncertainty propagation is not implemented for this method,
    /// and the returned result is exact.
    fn rem(self, rhs: Self) -> Self {
        let number = self.number % rhs.number;
        // Don't calculate uncertainty as the remainder function is discontinuous,
        // making it tricky
        number.into()
    }
}

impl Pow<Self> for SciFloat {
    type Output = Self;

    /// Raise the `SciFloat` to a `SciFloat` power.
    /// Currently missing correlated uncertainties.
    fn pow(self, rhs: Self) -> Self {
        if !(self.is_finite() && rhs.is_finite()) {
            todo!("Special values are not yet handled correctly by this method!")
        }
        let result = self.number.pow(rhs.number);
        if self.is_exact() {
            SciFloat::new(result)
        } else {
            let uncertainty = result.abs()
                * ((self.relative_uncertainty() * rhs.number).powi(2)
                    + (self.number.ln() * rhs.uncertainty).powi(2))
                .sqrt();
            SciFloat::new_with_uncertainty(result, uncertainty)
        }
    }
}

impl Pow<Self> for &SciFloat {
    type Output = SciFloat;

    fn pow(self, rhs: Self) -> SciFloat {
        (*self).pow(*rhs)
    }
}

impl Neg for SciFloat {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            number: -self.number,
            ..self
        }
    }
}

impl Neg for &SciFloat {
    type Output = SciFloat;

    #[inline]
    fn neg(self) -> SciFloat {
        SciFloat {
            number: -self.number,
            ..*self
        }
    }
}

impl Inv for SciFloat {
    type Output = Self;

    #[inline]
    fn inv(self) -> Self {
        Self::ONE / self
    }
}

impl Inv for &SciFloat {
    type Output = SciFloat;

    #[inline]
    fn inv(self) -> SciFloat {
        SciFloat::ONE / *self
    }
}

impl Display for SciFloat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} +/- {}", self.number, self.uncertainty)
    }
}

impl FromStr for SciFloat {
    type Err = SciNumError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let number: Result<f64, ParseFloatError> = s.parse();
        match number {
            Ok(num) => Ok(SciFloat::new(num)),
            Err(_) => Err(SciNumError::Parse(s.into())),
        }
    }
}

impl From<f64> for SciFloat {
    /// Converts an `f64` into a `SciFloat`.
    fn from(n: f64) -> Self {
        Self {
            number: n,
            uncertainty: 0.0,
        }
    }
}

impl From<SciFloat> for f64 {
    #[inline]
    fn from(n: SciFloat) -> Self {
        n.number()
    }
}

impl From<f32> for SciFloat {
    /// Converts an `f32` into a `SciFloat`.
    fn from(n: f32) -> Self {
        Self {
            number: n.into(),
            uncertainty: 0.0,
        }
    }
}

/// TODO: tests
impl From<SciDecimal> for SciFloat {
    /// Converts a `SciDecimal` to a `SciFloat`.
    /// 
    /// `n` is first rounded to 15 significant figures using `SciDecimal.round_sf()`,
    /// which in some cases may give the result a slightly lower precision than
    /// would theoretically be representable.
    /// The rounding uses the `RoundingMode::HalfEven` strategy.
    /// 
    /// If the absolute value of `n` is larger than `f64::MAX`, the appropriate
    /// infinity will be returned.
    /// If the absolute value of `n` is smaller than `f64::MIN_POSITIVE`, positive
    /// zero will be returned.
    fn from(n: SciDecimal) -> Self {
        Self {
            number: n.number().into(),
            uncertainty: n.uncertainty().into(),
        }
    }
}

macro_rules! impl_from_int {
    ($T:ty) => {
        impl From<$T> for SciFloat {
            fn from(t: $T) -> Self {
                Self::new(t.into())
            }
        }
    };
}

impl_from_int!(i8);
impl_from_int!(i16);
impl_from_int!(i32);
impl_from_int!(u8);
impl_from_int!(u16);
impl_from_int!(u32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_from_float() {
        // Using new
        let n = SciFloat::new(30.0);
        assert_eq!(n.number(), 30.0);
        assert_eq!(n.uncertainty(), 0.0);
        // Using from
        let n = SciFloat::from(42.0);
        assert_eq!(n.number(), 42.0);
        assert_eq!(n.uncertainty(), 0.0);
    }

    #[test]
    fn new_from_int_with_uncertainty() {
        let n = SciFloat::new_with_uncertainty(20.0, 2.0);
        assert_eq!(n.number(), 20.0);
        assert_eq!(n.uncertainty(), 2.0);
    }

    #[test]
    fn new_large() {
        let _n = SciFloat::new(236e40);
    }

    #[test]
    fn new_small() {
        let _n = SciFloat::new(49e-76);
    }

    #[test]
    fn new_largest_significand() {
        let _n = SciFloat::new(f64::MAX);
    }

    #[test]
    fn new_largest_negative_significand() {
        let _n = SciFloat::new(-f64::MAX);
    }

    #[test]
    fn uncertainty() {
        let n = SciFloat::new_with_uncertainty(30.0, 5.0);
        assert_eq!(n.uncertainty(), 5.0);
    }

    #[test]
    fn relative_uncertainty() {
        let n = SciFloat::new_with_uncertainty(20.0, 2.0);
        assert_eq!(n.relative_uncertainty(), 1e-1);

        let n2 = SciFloat::new_with_uncertainty(500.0, 5.0);
        assert_eq!(n2.relative_uncertainty(), 1e-2);

        let n3 = SciFloat::new_with_uncertainty(1000.0, 15.0);
        assert_eq!(n3.relative_uncertainty(), 15e-3);
    }

    #[test]
    fn is_exact() {
        let n1 = SciFloat::new(45.1);
        let n2 = SciFloat::new_with_uncertainty(500.0, 5.0);
        assert!(n1.is_exact());
        assert!(!n2.is_exact());
    }

    #[test]
    fn eq() {
        // Basic case
        assert_eq!(SciFloat::new(3.0), SciFloat::new(3.0));
        // Not equal, basic case
        assert_ne!(SciFloat::new(3.0), SciFloat::new(4.0));
        // Both zero
        assert_eq!(SciFloat::new(0.0), SciFloat::new(0.0));
        // Both zero, one is negative zero
        assert_eq!(SciFloat::new(0.0), SciFloat::new(-0.0));
        // Opposite sign but same significand
        assert_ne!(SciFloat::new(3.0), SciFloat::new(-3.0));
        // Same value but different precision
        assert_eq!(SciFloat::new(200e3), SciFloat::new(2e5));
        // How is this different than the previous one?
        // Same value but different precision, small numbers
        //assert_eq!(SciFloat::new(200, 3), SciFloat::new(2, 5));
    }

    #[test]
    fn truncate() {
        // Positive
        let n = SciFloat::new(25.6949);
        assert_eq!(n.trunc_sf(2), SciFloat::new(25.69));
        assert_eq!(n.trunc_sf(3), SciFloat::new(25.694));
        // Negative
        let n = SciFloat::new(-3.794718);
        assert_eq!(n.trunc_sf(4), SciFloat::new(-3.7947));
        assert_eq!(n.trunc_sf(3), SciFloat::new(-3.794));
        // Integer
        let n = SciFloat::new(4327890.0);
        assert_eq!(n.trunc_sf(4), SciFloat::new(4.327e6));
        assert_eq!(n.trunc_sf(5), SciFloat::new(4.3278e6));
        // Smaller than 1
        let n = SciFloat::new(0.4327890);
        assert_eq!(n.trunc_sf(4), SciFloat::new(4.327e-1));
        assert_eq!(n.trunc_sf(5), SciFloat::new(4.3278e-1));
    }

    #[test]
    fn add_exact() {
        let n1 = SciFloat::new(40.0);
        let n2 = SciFloat::new(5.1);
        let result = n1 + n2;
        assert_eq!(result, SciFloat::new(45.1));
    }

    #[test]
    fn add_with_uncertainty() {
        let n1 = SciFloat::new_with_uncertainty(20.0, 2.0);
        let n2 = SciFloat::new_with_uncertainty(30.0, 5.0);
        let result = n1 + n2;
        assert_eq!(result.number(), 50.0);
        //assert_eq!(
        //    Decimal::try_from(result.uncertainty()).unwrap().round_dp(5),
        //    dec!(5.3851648071345).round_dp(5)
        //);
    }

    #[test]
    fn sub_exact() {
        let n1 = SciFloat::new(20.0);
        let n2 = SciFloat::new(30.0);
        assert_eq!(n1 - n2, SciFloat::new(-10.0));
    }

    #[test]
    fn sub_with_uncertainty() {
        let n1 = SciFloat::new_with_uncertainty(20.0, 2.0);
        let n2 = SciFloat::new_with_uncertainty(30.0, 5.0);
        let result = n1 - n2;
        assert_eq!(result, SciFloat::new(-10.0));
        assert_eq!(result.uncertainty(), 5.385164807134504);
    }

    #[test]
    fn mul_exact() {
        let n1 = SciFloat::new(20.0);
        let n2 = SciFloat::new(30.0);
        assert_eq!(n1 * n2, SciFloat::new(600.0));
    }

    #[test]
    fn mul_with_uncertainty() {
        let n1 = SciFloat::new_with_uncertainty(20.0, 2.0);
        let n2 = SciFloat::new_with_uncertainty(30.0, 5.0);
        let result = n1 * n2;
        assert_eq!(result.number(), 600.0);
        assert_eq!(result.uncertainty(), 116.61903789690601);
        let ft = SciFloat::from(0.3048);
        let square_ft = ft * ft;
        assert_eq!(square_ft, SciFloat::new(0.09290304));
    }

    #[test]
    fn div_exact() {
        // Non-recurring result with same exponent
        assert_eq!(
            SciFloat::new(60.0) / SciFloat::new(30.0),
            SciFloat::new(2.0),
        );
        // Non-recurring result with different exponent
        assert_eq!(
            SciFloat::new(30.0) / SciFloat::new(60.0),
            SciFloat::new(5e-1),
        );
        // Identical recurring results with different pairs of starting numbers
        assert_eq!(
            SciFloat::new(30.0) / SciFloat::new(60.0),
            SciFloat::new(3e6) / SciFloat::new(6e6),
        );
        // Recurring results
        assert_eq!(
            (SciFloat::new(1.0) / SciFloat::new(3.0)),
            SciFloat::new(3333333333333333e-16),
        );
        assert_eq!(
            (SciFloat::new(1.0) / SciFloat::new(9.0)),
            SciFloat::new(1111111111111111e-16),
        );
    }

    #[test]
    fn div_with_uncertainty() {
        let n1 = SciFloat::new_with_uncertainty(20.0, 2.0);
        let n2 = SciFloat::new_with_uncertainty(30.0, 5.0);
        let result = n1 / n2;
        dbg!(result);
        assert_eq!(result.number(), 0.6666666666666666);
        assert_eq!(result.uncertainty(), 0.129576708774340);
    }

    #[test]
    fn div_with_uncertainty_reversed() {
        let n1 = SciFloat::new_with_uncertainty(20.0, 2.0);
        let n2 = SciFloat::new_with_uncertainty(30.0, 5.0);
        let result = n2 / n1;
        assert_eq!(result, SciFloat::new(1.5));
        assert_eq!(result.uncertainty(), 0.291547594742265);
    }

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
    fn inv() {
        assert_eq!(SciFloat::new(4.0).inv(), SciFloat::new(25e-2));
        assert_eq!(SciFloat::new(5e-1).inv(), SciFloat::new(2.0));
    }

    #[test]
    fn neg() {
        let n_pos = SciFloat::new(4.0);
        let n_neg = n_pos.neg();
        assert_eq!(n_neg, SciFloat::new(-4.0));
        assert_eq!(n_neg.number(), -4.0);
        let n_roundtrip = n_neg.neg();
        assert_eq!(n_roundtrip, n_pos);
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

    //#[test]
    //fn debug() {
    //    let n = SciFloat::new_with_uncertainty(20, 2, 0);
    //    assert_eq!(format!("{n:?}"), "SciFloat { number: 20, uncertainty: 2 }");
    //}

    #[test]
    fn display() {
        // Numbers with up to five places either side of the decimal point should
        // be displayed using normal notation
        // Integers should display without any decimal point at all
        assert_eq!(SciFloat::new(20.0).to_string(), "20 +/- 0");
        assert_eq!(SciFloat::new(-20.0).to_string(), "-20 +/- 0");
        assert_eq!(SciFloat::new(99999.0).to_string(), "99999 +/- 0");
        assert_eq!(SciFloat::new(10000.0).to_string(), "10000 +/- 0");
        assert_eq!(SciFloat::new(1000.0).to_string(), "1000 +/- 0");
        assert_eq!(SciFloat::new(100.0).to_string(), "100 +/- 0");
        assert_eq!(SciFloat::new(10.0).to_string(), "10 +/- 0");
        assert_eq!(SciFloat::new(1.0).to_string(), "1 +/- 0");
        assert_eq!(SciFloat::new(0.1).to_string(), "0.1 +/- 0");
        assert_eq!(SciFloat::new(0.01).to_string(), "0.01 +/- 0");
        assert_eq!(SciFloat::new(0.001).to_string(), "0.001 +/- 0");
        assert_eq!(SciFloat::new(0.0001).to_string(), "0.0001 +/- 0");
        assert_eq!(SciFloat::new(0.00001).to_string(), "0.00001 +/- 0");
        assert_eq!(SciFloat::new(0.0325).to_string(), "0.0325 +/- 0");
        assert_eq!(SciFloat::new(-0.0325).to_string(), "-0.0325 +/- 0");
        assert_eq!(SciFloat::new(85.13).to_string(), "85.13 +/- 0");
        assert_eq!(SciFloat::new(81700.0).to_string(), "81700 +/- 0");

        assert_eq!(
            SciFloat::new_with_uncertainty(20.0, 2.0).to_string(),
            "20 +/- 2"
        );
        assert_eq!(
            SciFloat::new_with_uncertainty(10000.0, 15.0).to_string(),
            "10000 +/- 15"
        );
        assert_eq!(
            SciFloat::new_with_uncertainty(86.75309, 42.0).to_string(),
            "86.75309 +/- 42"
        );
        assert_eq!(
            SciFloat::new_with_uncertainty(-86.75309, 42.0).to_string(),
            "-86.75309 +/- 42"
        );
    }

    #[test]
    fn from_str() {
        // Integer
        assert_eq!(SciFloat::from_str("42").unwrap(), SciFloat::new(42.0));
        // Decimal
        assert_eq!(SciFloat::from_str("0.0859").unwrap(), SciFloat::new(859e-4));
        // Decimal without integral part before decimal point
        assert_eq!(SciFloat::from_str(".0859").unwrap(), SciFloat::new(859e-4));
        // Negative decimal
        assert_eq!(SciFloat::from_str("-3.12").unwrap(), SciFloat::new(-312e-2));
        // Scientific notation
        assert_eq!(SciFloat::from_str("1.5e8").unwrap(), SciFloat::new(15e7));
        // Scientific notation with negative exponent
        assert_eq!(SciFloat::from_str("2e-5").unwrap(), SciFloat::new(2e-5));
        // Negative number with positive exponent
        assert_eq!(
            SciFloat::from_str("-6.022e6").unwrap(),
            SciFloat::new(-6022e3)
        );
        // Large exponents
        assert_eq!(SciFloat::from_str("1.5e18").unwrap(), SciFloat::new(15e17));
        assert_eq!(
            SciFloat::from_str("-6.022e23").unwrap(),
            SciFloat::new(-6022e20)
        );
        // Capital E for exponent
        assert_eq!(SciFloat::from_str("1.5E8").unwrap(), SciFloat::new(15e7));
        // 16 significant figures must always be fine
        assert_eq!(
            SciFloat::from_str("0.5293040185492948").unwrap(),
            SciFloat::new(5293040185492948e-16)
        );
        // Make sure incorrectly formatted strings fail
        assert!(SciFloat::from_str("not a number").is_err());
        assert!(SciFloat::from_str("x.482").is_err());
        assert!(SciFloat::from_str("52.x").is_err());
        assert!(SciFloat::from_str("-2.42F-4").is_err());
    }
}
