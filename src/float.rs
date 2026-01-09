// SPDX-FileCopyrightText: 2025 Matthew Milner <matterhorn103@proton.me>
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    fmt::Display,
    num::ParseFloatError,
    ops::{Add, Div, Mul, Rem, Sub, Neg},
    str::FromStr,
};

use num_traits::{Inv, Pow, Num, One, Zero, FloatConst};

use crate::{SciDecimal, SciNum, error::SciNumError};

#[derive(Debug, Clone, Copy)]
pub struct SciFloat {
    number: f64,
    uncertainty: f64,
}


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

    /// Returns true if the `SciDecimal` has an uncertainty of zero.
    #[inline]
    pub fn is_exact(&self) -> bool {
        self.uncertainty == 0.0
    }

    pub fn truncate(mut self, sig_figs: u32) -> Self {
        // If integer
        let scale = 10_f64.powi(sig_figs as i32);
        let mut value = self.number();
        if value.fract() != 0.0 {
            self.number = (value * scale).trunc() / scale;
            return self;
        } else {
            let mut exponent: i32 = 0;
            while value.fract() != value {
                value = value / 10.0;
                exponent += 1;
            }
            self.number = (value * scale).trunc() / 10_f64.powi(sig_figs as i32 - exponent);
            self
        }
    }
}

impl SciNum for SciFloat {
    type Number = f64;

    #[inline]
    fn number(&self) -> f64 {
        self.number
    }

    #[inline]
    fn uncertainty(&self) -> f64 {
        self.uncertainty
    }

    #[inline]
    fn relative_uncertainty(&self) -> f64 {
        self.uncertainty / self.number.abs()
    }

    #[inline]
    fn with_uncertainty(self, uncertainty: f64) -> Self {
        Self {
            number: self.number,
            uncertainty,
        }
    }

    const ZERO: Self = SciFloat {
        number: 0.0,
        uncertainty: 0.0,
    };

    const ONE: Self = SciFloat {
        number: 1.0,
        uncertainty: 0.0,
    };
}

impl From<f64> for SciFloat {
    fn from(n: f64) -> Self {
        Self {
            number: n,
            uncertainty: 0.0,
        }
    }
}

impl From<f32> for SciFloat {
    fn from(n: f32) -> Self {
        Self {
            number: n.into(),
            uncertainty: 0.0,
        }
    }
}

impl TryFrom<SciDecimal> for SciFloat {
    type Error = ParseFloatError;

    fn try_from(n: SciDecimal) -> Result<Self, Self::Error> {
        let number: f64 = n.number().to_string().parse()?;
        let uncertainty: f64 = n.uncertainty().to_string().parse()?;
        Ok(Self {
            number,
            uncertainty,
        })
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

impl Num for SciFloat {
    type FromStrRadixErr = <f64 as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(Self {
            number: f64::from_str_radix(str, radix)?,
            uncertainty: 0.0,
        })
    }
}

impl Zero for SciFloat {
    fn zero() -> Self {
        Self {
            number: 0.0,
            uncertainty: 0.0,
        }
    }

    fn is_zero(&self) -> bool {
        self.number.is_zero()
    }
}

impl One for SciFloat {
    fn one() -> Self {
        Self { number: 1.0, uncertainty: 0.0 }
    }
}

// Methods that will belong to the Float trait if we implement it properly later
// impl Float for SciFloat {
impl SciFloat {
    //fn min_value() -> Self {
    //    todo!()
    //}

    //fn min_positive_value() -> Self {
    //    todo!()
    //}

    //fn epsilon() -> Self {
    //    todo!()
    //}

    //fn max_value() -> Self {
    //    todo!()
    //}

    //fn floor(self) -> Self {
    //    todo!()
    //}

    //fn ceil(self) -> Self {
    //    todo!()
    //}

    //fn round(self) -> Self {
    //    todo!()
    //}

    //fn trunc(self) -> Self {
    //    todo!()
    //}

    //fn fract(self) -> Self {
    //    todo!()
    //}

    pub fn abs(self) -> Self {
        Self {
            number: self.number.abs(),
            uncertainty: self.uncertainty
        }
    }

    //fn signum(self) -> Self {
    //    todo!()
    //}

    //fn is_sign_positive(self) -> bool {
    //    todo!()
    //}

    //fn is_sign_negative(self) -> bool {
    //    todo!()
    //}

    //fn mul_add(self, a: Self, b: Self) -> Self {
    //    todo!()
    //}

    //fn recip(self) -> Self {
    //    todo!()
    //}

    /// Raise the `SciFloat` to an integer power.
    #[inline]
    pub fn powi(self, n: i32) -> Self {
        let number = self.number().powi(n);

        if self.is_exact() {
            SciFloat::new(number)
        } else {
            let uncertainty = (self.uncertainty() * (n as f64) * number.abs()) / self.number();
            SciFloat::new_with_uncertainty(number, uncertainty)
        }
    }

    //fn powf(self, n: Self) -> Self {
    //    todo!()
    //}

    pub fn sqrt(self) -> Self {
        let number: f64 = self.number().sqrt();
        let uncertainty: f64 = (number * self.uncertainty()) / (2.0 * self.number());
        Self { number, uncertainty }
    }

    pub fn exp(self) -> Self {
        let number = f64::E().powf(self.number()).into();
        let uncertainty = number * self.uncertainty();
        Self { number, uncertainty }
    }

    //fn exp2(self) -> Self {
    //    todo!()
    //}

    pub fn ln(self) -> Self {
        let number = self.number().ln();
        if self.is_exact() {
            Self::from(number)
        } else {
            let uncertainty = self.relative_uncertainty().abs();
            Self::from(number).with_uncertainty(uncertainty.into())
        }
    }

    //fn log(self, base: Self) -> Self {
    //    todo!()
    //}

    //fn log2(self) -> Self {
    //    todo!()
    //}

    pub fn log10(self) -> Self {
        let number = self.number().log10();
        if self.is_exact() {
            Self::from(number)
        } else {
            let uncertainty = self.uncertainty() / ((10.0_f64).ln() * self.number()).abs();
            Self::new_with_uncertainty(number, uncertainty)
        }
    }

    pub fn to_degrees(self) -> Self {
        let number = self.number() * (180.0 / f64::PI());
        let uncertainty = self.uncertainty() * (180.0 / f64::PI());
        Self { number, uncertainty }
    }

    pub fn to_radians(self) -> Self {
        let number = self.number() * (f64::PI() / 180.0);
        let uncertainty = self.uncertainty() * (f64::PI() / 180.0);
        Self { number, uncertainty }
    }

    pub fn max(self, other: Self) -> Self {
        match self > other {
            true => self,
            false => other
        }
    }

    pub fn min(self, other: Self) -> Self {
        match self < other {
            true => self,
            false => other
        }
    }

    pub fn cbrt(self) -> Self {
        let number: f64 = self.number().cbrt();
        let uncertainty: f64 = (number * self.uncertainty()) / (3.0 * self.number());
        Self { number, uncertainty }
    }

    pub fn hypot(self, other: Self) -> Self {
        let number = (self.number().powi(2) + other.number().powi(2)).sqrt();
        let uncertainty = ((self.number() * self.uncertainty()).abs() + (other.number() * other.uncertainty()).abs()) / number;
        Self { number, uncertainty }
    }

    pub fn sin(self) -> Self {
        let number = self.number().sin();
        let uncertainty = (self.number().cos() * self.uncertainty()).abs();
        Self { number, uncertainty }
    }

    pub fn cos(self) -> Self {
        let number = self.number().cos();
        let uncertainty = (self.number().sin() * self.uncertainty()).abs();
        Self { number, uncertainty }
    }

    pub fn tan(self) -> Self {
        let number = self.number().tan();
        let uncertainty: f64 = ((1_f64 / (self.number().cos().powi(2))) * self.uncertainty()).abs();
        Self { number, uncertainty }
    }

    //fn asin(self) -> Self {
    //    todo!()
    //}

    //fn acos(self) -> Self {
    //    todo!()
    //}

    //fn atan(self) -> Self {
    //    todo!()
    //}

    //fn atan2(self, other: Self) -> Self {
    //    todo!()
    //}

    pub fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    //fn exp_m1(self) -> Self {
    //    todo!()
    //}

    //fn ln_1p(self) -> Self {
    //    todo!()
    //}

    //fn sinh(self) -> Self {
    //    todo!()
    //}

    //fn cosh(self) -> Self {
    //    todo!()
    //}

    //fn tanh(self) -> Self {
    //    todo!()
    //}

    //fn asinh(self) -> Self {
    //    todo!()
    //}

    //fn acosh(self) -> Self {
    //    todo!()
    //}

    //fn atanh(self) -> Self {
    //    todo!()
    //}
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
        let number: f64 = self.number().pow(rhs.number());
        let uncertainty: f64 = number
            * (
                (self.uncertainty() * (rhs.number() / self.number()))
                + (rhs.uncertainty() * self.number().ln())
            );

        Self { number, uncertainty }
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
        Self { number: -self.number, ..self }
    }
}

impl Neg for &SciFloat {
    type Output = SciFloat;

    #[inline]
    fn neg(self) -> SciFloat {
        SciFloat { number: -self.number, ..*self }
    }
}

impl Inv for SciFloat {
    type Output = Self;

    #[inline]
    fn inv(self) -> Self {
        Self::one() / self
    }
}

impl Inv for &SciFloat {
    type Output = SciFloat;

    #[inline]
    fn inv(self) -> SciFloat {
        SciFloat::one() / *self
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
        assert_eq!(n.truncate(2), SciFloat::new(25.69));
        assert_eq!(n.truncate(3), SciFloat::new(25.694));
        // Negative
        let n = SciFloat::new(-3.794718);
        assert_eq!(n.truncate(4), SciFloat::new(-3.7947));
        assert_eq!(n.truncate(3), SciFloat::new(-3.794));
        // Integer
        let n = SciFloat::new(4327890.0);
        assert_eq!(n.truncate(4), SciFloat::new(4.327e6));
        assert_eq!(n.truncate(5), SciFloat::new(4.3278e6));
        // Smaller than 1
        let n = SciFloat::new(0.4327890);
        assert_eq!(n.truncate(4), SciFloat::new(4.327e-1));
        assert_eq!(n.truncate(5), SciFloat::new(4.3278e-1));
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
        assert_eq!(
            result.uncertainty(),
            0.291547594742265
        );
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
