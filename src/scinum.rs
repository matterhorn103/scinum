// SPDX-FileCopyrightText: 2025 Matthew Milner <matterhorn103@proton.me>
// SPDX-License-Identifier: MIT

use std::{
    fmt::{self, Debug},
    ops::{Add, Div, Mul, Rem, Sub},
    str::FromStr,
};

use num_traits::{FromPrimitive, Inv, Num, One, Pow, Zero};
use regex::Regex;
use rust_decimal::{Decimal, MathematicalOps};

use crate::error::SciNumError;

/// A decimal floating point number with an associated uncertainty.
///
/// Represents a number of the form (_m_ ± _u_) × 10<sup><i>n</i></sup>.
///
/// The design is intended to allow excellent compatibility with other numeric
/// types and provide the precision of 64-bit formats while also propagating
/// uncertainties across arithmetic operations.
/// `SciNum` uses a 64-bit significand in binary integer format (providing
/// 18 decimal digits of precision) and a 16-bit signed exponent.
/// As such, all values covered by the IEEE 754-2008 `binary64` (i.e. `f64`)
/// and `decimal64` formats are representable.
///
/// Rounding, formatting, and parsing methods are provided with a view to
/// enabling typical scientific calculations.
#[derive(Copy, Clone, serde_with::DeserializeFromStr, serde_with::SerializeDisplay)]
pub struct SciNum {
    negative: bool,
    exponent: i16,
    uncertainty_scale: u8, // This will allow the uncertainty to have a different precision, but for the moment must always be 0
    uncertainty: u32,
    significand: u64,
}

impl SciNum {
    /// Creates an exact `SciNum` from parts corresponding to _m_ ×
    /// 10<sup><i>n</i></sup>.
    ///
    /// # Example
    ///
    /// ```
    /// # use scinum::SciNum;
    /// #
    /// let n = SciNum::new(251, -3);
    /// assert_eq!(n.to_string(), "0.251");
    /// ```
    pub fn new(number: i128, exponent: i16) -> Self {
        Self {
            negative: number.is_negative(),
            exponent,
            uncertainty_scale: 0,
            uncertainty: 0,
            significand: number.unsigned_abs() as u64,
        }
    }

    /// Creates a `SciNum` from parts corresponding to (_m_ ± _u_) ×
    /// 10<sup><i>n</i></sup>.
    ///
    /// This means the number of decimal places in the number and uncertainty
    /// will be the same in the created `SciNum`.
    ///
    /// # Example
    ///
    /// ```
    /// # use scinum::SciNum;
    /// #
    /// let n = SciNum::new_with_uncertainty(251, 3, -3);
    /// assert_eq!(n.to_string(), "0.251(3)");
    /// ```
    pub fn new_with_uncertainty(number: i128, uncertainty: u32, exponent: i16) -> Self {
        Self {
            negative: number.is_negative(),
            exponent,
            uncertainty_scale: 0,
            uncertainty,
            significand: number.unsigned_abs() as u64,
        }
    }

    /// Creates a `SciNum` from separate integer and fractional parts,
    /// corresponding to `ii.ffff(uu) × 10^nn`.
    ///
    /// # Example
    ///
    /// ```
    /// # use scinum::SciNum;
    /// #
    /// let n = SciNum::from_scientific_parts(2, 51, 3, -1);
    /// assert_eq!(n.to_string(), "0.251(3)");
    /// ```
    pub fn from_scientific_parts(
        integer: i16,
        fraction: u32,
        uncertainty: u32,
        exponent: i16,
    ) -> Self {
        let unsigned_integer: u32 = integer.unsigned_abs().into();
        let (significand, exponent): (u64, i16) = {
            if let Some(decimal_places_minus_one) = fraction.checked_ilog10() {
                let decimal_places = decimal_places_minus_one + 1;
                let significand = (unsigned_integer * 10_u32.pow(decimal_places)) + fraction;
                let exponent = exponent - (decimal_places as i16);
                (significand.into(), exponent)
            } else {
                (unsigned_integer.into(), exponent)
            }
        };
        Self {
            negative: integer.is_negative(),
            exponent,
            uncertainty_scale: 0,
            uncertainty,
            significand,
        }
    }

    /// Creates a new `SciNum` with the same number but the provided
    /// uncertainty.
    /// 
    /// # Panics
    /// 
    /// This function panics if the uncertainty has a significand greater than
    /// `u32::MAX` (i.e. more than ~9 decimal places).
    ///
    /// # Example
    ///
    /// ```
    /// # use scinum::SciNum;
    /// #
    /// let n = SciNum::new(251, -3).with_uncertainty(SciNum::new(3, -3));
    /// assert_eq!(n.to_string(), "0.251(3)");
    /// assert_eq!(n, SciNum::new_with_uncertainty(251, 3, -3));
    #[inline]
    pub fn with_uncertainty(mut self, uncertainty: Self) -> Self {
        let uncertainty_scale = self.exponent - uncertainty.exponent;
        if uncertainty_scale != 0 {
            todo!("Currently, an uncertainty must have the same precision as the number itself!")
        } else {
            self.uncertainty = uncertainty.significand.try_into().expect("The uncertainty may not have a significand greater than `u32::MAX`!");
        }
        self
    }

    /// Returns the number as an exact `SciNum` without its uncertainty.
    #[inline]
    pub fn number(&self) -> Self {
        Self {
            uncertainty_scale: 0,
            uncertainty: 0,
            ..*self
        }
    }

    /// Returns the absolute uncertainty as an exact `SciNum`.
    ///
    /// The uncertainty is always positive.
    #[inline]
    pub fn uncertainty(&self) -> Self {
        Self {
            negative: false,
            exponent: self.exponent,
            uncertainty_scale: 0,
            uncertainty: 0,
            significand: self.uncertainty.into(),
        }
    }

    /// Returns the relative uncertainty as an exact `SciNum`.
    ///
    /// The relative uncertainty is always positive.
    #[inline]
    pub(crate) fn relative_uncertainty(&self) -> Self {
        todo!();
        //self.uncertainty() / self.number().abs()
    }

    /// Returns the significand _m_ of the number when represented with _m_ as
    /// an integer.
    ///
    /// Corresponds to representation of the number as `mmmmm × 10^nn`.
    #[inline]
    pub fn significand_integral(&self) -> i128 {
        if self.negative {
            -(self.significand as i128)
        } else {
            self.significand as i128
        }
    }

    /// Returns the exponent _n_ of the number when represented with _m_ as an
    /// integer.
    ///
    /// Corresponds to representation of the number as `mmmmm × 10^nn`.
    #[inline]
    pub fn exponent_integral(&self) -> i16 {
        self.exponent
    }

    // Returns the significand _m_ of the number when represented with
    // normalized notation i.e. with 10 > _m_ >= 1.
    //
    // Corresponds to `iffff` when the number is notated as `i.ffff × 10^nn`.
    //#[inline]
    //pub fn significand_normalized(&self) -> i128 {
    //    let unsigned = (self.number_hi as i128) << 64
    //        | (self.number_mid as i128) << 32
    //        | self.number_lo as i128;
    //    if self.negative { -unsigned } else { unsigned }
    //}

    /// Returns the integer, leading zeros, fractional, uncertainty, and
    /// exponent parts of the of the number when represented with normalized
    /// notation i.e. with 10 > _m_ >= 1.
    ///
    /// Corresponds to `(ii, zz, fff, uu, nn)` when the number is notated as
    /// `ii.zzfff(uu) × 10^nn`.
    pub fn scientific_parts_normalized(&self) -> (i8, u8, Option<i128>, u32, i16) {
        todo!()
        //if self.is_zero() {
        //    return (0, None, 0);
        //}
        //let significand = self.significand_integral();
        //// Work out the number of places the decimal point needs to move to the left in
        //// the significand to get the correct representation
        //let (divisor, exponent) = if Decimal::try_from(self.number()).abs() < Decimal::ONE {
        //    // For small numbers decimal already provides us with the scale
        //    let shifted_places = significand.abs().ilog10() as i16;
        //    let divisor = 10_i128.pow(shifted_places as u32);
        //    let exponent = -(self.number_scale as i16) + shifted_places;
        //    (divisor, exponent)
        //} else {
        //    // For large integers Decimal's scale is 0 so we take the base 10 logarithm
        //    let shifted_places = (significand.abs().ilog10() as i16) + (self.number_scale as i16);
        //    let divisor = 10_i128.pow(shifted_places as u32);
        //    let exponent = self.exponent_integral() + shifted_places;
        //    (divisor, exponent)
        //};
        //let int_part = significand / divisor;
        //let frac_part = significand.abs() % divisor;
        //let exp_part = exponent;
        //
        //(int_part as i8, Some(frac_part), exp_part)
    }

    /// Returns the exponent _n_ of the number when represented with normalized
    /// notation i.e. with 10 > _m_ >= 1.
    ///
    /// Corresponds to `nn` when the number is notated as `ii.ffff(uu) × 10^nn`.
    #[inline]
    pub fn exponent_normalized(&self) -> i16 {
        todo!()
    }

    /// Returns the number of significant decimal digits in the significand.
    /// 0 is considered to have 0 significant figures.
    #[inline]
    pub fn sigfigs(&self) -> u32 {
        if let Some(log) = self.significand.checked_ilog10() {
            log + 1
        } else {
            0
        }
    }

    /// Returns the scale of the last significant place.
    ///
    /// For example:
    /// - 0.02 returns -2
    /// - 0.020 returns -3
    /// - 2 returns 0
    /// - 200 returns 2 or 1 or 0, depending on the precision of the number
    #[inline]
    pub fn precision(&self) -> i16 {
        self.exponent
    }

    /// Returns true if the `SciNum` has an uncertainty of zero.
    #[inline]
    pub fn is_exact(&self) -> bool {
        self.uncertainty == 0
    }

    /// Returns true if the sign bit is negative.
    /// Zero is considered positive.
    #[inline(always)]
    //#[must_use]
    pub const fn is_sign_negative(&self) -> bool {
        self.negative
    }

    /// Returns true if the sign bit is positive.
    /// Zero is also considered positive.
    #[inline(always)]
    //#[must_use]
    pub const fn is_sign_positive(&self) -> bool {
        !self.negative
    }

    /// Creates an exact `SciNum` from a float.
    ///
    /// Currently this goes via `Decimal::try_from_f64()`.
    pub fn from_f64(number: f64) -> Option<Self> {
        let dec = Decimal::from_f64(number)?;
        Some(dec.into())
    }

    pub fn add_with_correlation<T>(self, rhs: Self, correlation: T) -> Self
    where
        T: Into<Decimal>,
    {
        let number = Decimal::try_from(self.number()).unwrap() + Decimal::try_from(rhs.number()).unwrap();
        if self.is_exact() && rhs.is_exact() {
            Self::from(number)
        } else {
            let sigma_ab = correlation.into()
                * Decimal::try_from(self.uncertainty()).unwrap()
                * Decimal::try_from(rhs.uncertainty()).unwrap();
            let uncertainty = ((Decimal::try_from(self.uncertainty()).unwrap().powu(2))
                + (Decimal::try_from(rhs.uncertainty()).unwrap().powu(2))
                + (Decimal::TWO * sigma_ab))
                .sqrt()
                .unwrap();
            Self::from(number).with_uncertainty(uncertainty.into())
        }
    }

    pub fn sub_with_correlation<T>(self, rhs: Self, correlation: T) -> Self
    where
        T: Into<Decimal>,
    {
        let number = Decimal::try_from(self.number()).unwrap() - Decimal::try_from(rhs.number()).unwrap();
        if self.is_exact() && rhs.is_exact() {
            Self::from(number)
        } else {
            let sigma_ab = correlation.into()
                * Decimal::try_from(self.uncertainty()).unwrap()
                * Decimal::try_from(rhs.uncertainty()).unwrap();
            let uncertainty = ((Decimal::try_from(self.uncertainty()).unwrap().powu(2))
                + (Decimal::try_from(rhs.uncertainty()).unwrap().powu(2))
                - (Decimal::TWO * sigma_ab))
                .sqrt()
                .unwrap();
            Self::from(number).with_uncertainty(uncertainty.into())
        }
    }

    pub fn mul_with_correlation<T>(self, rhs: Self, correlation: T) -> Self
    where
        T: Into<Decimal>,
    {
        let number = Decimal::try_from(self.number()).unwrap() * Decimal::try_from(rhs.number()).unwrap();
        if self.is_exact() && rhs.is_exact() {
            Self::from(number)
        } else {
            let sigma_ab = correlation.into()
                * Decimal::try_from(self.uncertainty()).unwrap()
                * Decimal::try_from(rhs.uncertainty()).unwrap();
            let uncertainty = ((Decimal::try_from(self.relative_uncertainty()).unwrap().powu(2))
                + (Decimal::try_from(rhs.relative_uncertainty()).unwrap().powu(2))
                + (Decimal::TWO * sigma_ab / number))
                .sqrt()
                .unwrap()
                * number.abs();
            Self::from(number).with_uncertainty(uncertainty.into())
        }
    }

    pub fn div_with_correlation<T>(self, rhs: Self, correlation: T) -> Self
    where
        T: Into<Decimal>,
    {
        let number = Decimal::try_from(self.number()).unwrap() / Decimal::try_from(rhs.number()).unwrap();
        if self.is_exact() && rhs.is_exact() {
           Self::from(number)
        } else {
            let sigma_ab = correlation.into()
                * Decimal::try_from(self.uncertainty()).unwrap()
                * Decimal::try_from(rhs.uncertainty()).unwrap();
            let uncertainty = ((Decimal::try_from(self.relative_uncertainty()).unwrap().powu(2))
                + (Decimal::try_from(rhs.relative_uncertainty()).unwrap().powu(2))
                - (Decimal::TWO * sigma_ab / number))
                .sqrt()
                .unwrap()
                * number.abs();
            Self::from(number).with_uncertainty(uncertainty.into())
        }
    }

    #[inline]
    pub fn powi(self, rhs: i64) -> Self {
        self.powd(rhs.into())
    }

    #[inline]
    pub fn powd(self, rhs: Decimal) -> Self {
        self.pow_with_correlation(rhs.into(), Decimal::ZERO)
    }

    #[inline]
    pub fn powf(self, rhs: f64) -> Self {
        let rhs = Self::from_f64(rhs).unwrap();
        self.pow_with_correlation(rhs, Decimal::ZERO)
    }

    //#[inline]
    //pub fn powfrac(self, rhs: Frac) -> Self {
    //    let n: Decimal = (*rhs.numer()).into();
    //    let d: Decimal = (*rhs.denom()).into();
    //    let rhs = n / d;
    //    self.powd(rhs)
    //}

    pub fn pow_with_correlation<T>(self, rhs: Self, correlation: T) -> Self
    where
        T: Into<Decimal>,
    {
        let number = Decimal::try_from(self.number()).unwrap().powd(Decimal::try_from(rhs.number()).unwrap());
        if self.is_exact() && rhs.is_exact() {
            Self::from(number)
        } else {
            let sigma_ab = correlation.into()
                * Decimal::try_from(self.uncertainty()).unwrap()
                * Decimal::try_from(rhs.uncertainty()).unwrap();
            let uncertainty = ((Decimal::try_from(self.relative_uncertainty()).unwrap() * Decimal::try_from(rhs.number()).unwrap()).powu(2)
                + (Decimal::try_from(self.number()).unwrap().ln() * Decimal::try_from(rhs.uncertainty()).unwrap()).powu(2)
                + (Decimal::TWO
                    * ((Decimal::try_from(self.number()).unwrap().ln() * Decimal::try_from(rhs.number()).unwrap())
                        / Decimal::try_from(self.number()).unwrap())
                    * sigma_ab))
                .sqrt()
                .unwrap()
                * number.abs();
            Self::from(number).with_uncertainty(uncertainty.into())
        }
    }

    pub fn ln(self) -> Self {
        let number = Decimal::try_from(self.number()).unwrap().ln();
        if self.is_exact() {
            Self::from(number)
        } else {
            let uncertainty = Decimal::try_from(self.relative_uncertainty()).unwrap().abs();
            Self::from(number).with_uncertainty(uncertainty.into())
        }
    }

    pub fn log10(self) -> Self {
        let number = Decimal::try_from(self.number()).unwrap().log10();
        if self.is_exact() {
            Self::from(number)
        } else {
            let uncertainty = (Decimal::try_from(self.uncertainty()).unwrap()
                / (Decimal::TEN.ln() * Decimal::try_from(self.number()).unwrap()))
            .abs();
            Self::from(number).with_uncertainty(uncertainty.into())
        }
    }

    pub fn exp(self) -> Self {
        let number = Decimal::try_from(self.number()).unwrap().exp();
        if self.is_exact() {
            Self::from(number)
        } else {
            let uncertainty = number.abs() * Decimal::try_from(self.uncertainty()).unwrap();
            Self::from(number).with_uncertainty(uncertainty.into())
        }
    }
}

impl Num for SciNum {
    type FromStrRadixErr = rust_decimal::Error;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, rust_decimal::Error> {
        // For now, just make use of the Decimal implementation
        let dec = Decimal::from_str_radix(str, radix)?;
        Ok(Self::from(dec))
    }
}

impl Zero for SciNum {
    #[inline]
    fn zero() -> Self {
        Self::ZERO
    }

    /// Returns true if the `SciNum` is equal to zero, regardless of any
    /// uncertainty.
    #[inline]
    fn is_zero(&self) -> bool {
        self.significand == 0
    }
}

impl One for SciNum {
    #[inline]
    fn one() -> Self {
        Self::ONE
    }
}

impl From<Decimal> for SciNum {
    #[inline]
    /// Converts a `rust_decimal::Decimal` to a `SciNum`.
    /// 
    /// A silent loss of precision will occur if the `Decimal` has more than 18
    /// significant figures.
    /// If this is the case, `n` is first rounded to 18 decimal places using
    /// `Decimal.rescale()`; the rounding thus follows the
    /// `rust_decimal::RoundingStrategy::MidpointAwayFromZero` strategy.
    fn from(mut n: Decimal) -> Self {
        if n.scale() > 18 {
            n.rescale(18);
        }
        // `n.hi` should now always be 0 and the significand should fit into a `u64`
        let mantissa = n.mantissa();
        // `n.scale()` is max 28 anyway, should be max 18 at this point
        Self::new(mantissa, -(n.scale() as i16))
    }
}

impl TryFrom<SciNum> for Decimal {
    type Error = rust_decimal::Error;

    /// Attempts to convert a `SciNum` into a `rust_decimal::Decimal`.
    /// 
    /// Fails if `n` has a positive exponent or an exponent lower than −28.
    fn try_from(n: SciNum) -> Result<Decimal, rust_decimal::Error> {
        if n.exponent.is_positive() {
            Err(rust_decimal::Error::ConversionTo("Decimal".to_string()))
        } else {
            Decimal::try_from_i128_with_scale(
                n.significand_integral(),
                n.exponent.unsigned_abs().into(),
            )
        }
    }
}

macro_rules! impl_from_int {
    ($T:ty) => {
        impl From<$T> for SciNum {
            fn from(t: $T) -> Self {
                Self::new(t.into(), 0)
            }
        }
    };
}

impl_from_int!(i8);
impl_from_int!(i16);
impl_from_int!(i32);
impl_from_int!(i64);
impl_from_int!(u8);
impl_from_int!(u16);
impl_from_int!(u32);
impl_from_int!(u64);

impl PartialEq for SciNum {
    fn eq(&self, other: &Self) -> bool {
        Decimal::try_from(*self).unwrap() == Decimal::try_from(*other).unwrap()
    }
}

impl Eq for SciNum {}

impl PartialOrd for SciNum {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SciNum {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        Decimal::try_from(*self).unwrap().cmp(&Decimal::try_from(*other).unwrap())
    }
}

impl Add for SciNum {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        self.add_with_correlation(rhs, Decimal::ZERO)
    }
}

impl Add for &SciNum {
    type Output = SciNum;

    fn add(self, rhs: Self) -> SciNum {
        self.add_with_correlation(*rhs, Decimal::ZERO)
    }
}

impl Sub for SciNum {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self.sub_with_correlation(rhs, Decimal::ZERO)
    }
}

impl Sub for &SciNum {
    type Output = SciNum;

    fn sub(self, rhs: Self) -> SciNum {
        self.sub_with_correlation(*rhs, Decimal::ZERO)
    }
}

impl Mul for SciNum {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        self.mul_with_correlation(rhs, Decimal::ZERO)
    }
}

impl Mul for &SciNum {
    type Output = SciNum;

    fn mul(self, rhs: Self) -> SciNum {
        self.mul_with_correlation(*rhs, Decimal::ZERO)
    }
}

impl Div for SciNum {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        self.div_with_correlation(rhs, Decimal::ZERO)
    }
}

impl Div for &SciNum {
    type Output = SciNum;

    fn div(self, rhs: Self) -> SciNum {
        self.div_with_correlation(*rhs, Decimal::ZERO)
    }
}

impl Rem for SciNum {
    type Output = Self;

    /// Performs the `%` operation.
    ///
    /// WARNING: Uncertainty propagation is not yet implemented for this method,
    /// and the returned result will be exact.
    fn rem(self, rhs: Self) -> Self {
        let number = Decimal::try_from(self.number()).unwrap() % Decimal::try_from(rhs.number()).unwrap();
        number.into()
    }
}

impl Rem for &SciNum {
    type Output = SciNum;

    /// Performs the `%` operation.
    ///
    /// WARNING: Uncertainty propagation is not yet implemented for this method,
    /// and the returned result will be exact.
    fn rem(self, rhs: Self) -> SciNum {
        let number = Decimal::try_from(self.number()).unwrap() % Decimal::try_from(rhs.number()).unwrap();
        number.into()
    }
}

impl Pow<Self> for SciNum {
    type Output = Self;

    fn pow(self, rhs: Self) -> Self {
        self.pow_with_correlation(rhs, Decimal::ZERO)
    }
}

impl Pow<Self> for &SciNum {
    type Output = SciNum;

    fn pow(self, rhs: Self) -> SciNum {
        self.pow_with_correlation(*rhs, Decimal::ZERO)
    }
}

impl Inv for SciNum {
    type Output = Self;

    #[inline]
    fn inv(self) -> Self {
        Self::ONE / self
    }
}

impl Inv for &SciNum {
    type Output = SciNum;

    #[inline]
    fn inv(self) -> SciNum {
        SciNum::ONE / *self
    }
}

macro_rules! impl_arithmetic_int {
    ($t:ty) => {
        impl Add<$t> for SciNum {
            type Output = SciNum;

            fn add(self, rhs: $t) -> SciNum {
                self.add_with_correlation(rhs.into(), Decimal::ZERO)
            }
        }

        impl Add<SciNum> for $t {
            type Output = SciNum;

            fn add(self, rhs: SciNum) -> SciNum {
                let num: SciNum = self.into();
                num.add_with_correlation(rhs, Decimal::ZERO)
            }
        }

        impl Sub<$t> for SciNum {
            type Output = Self;

            fn sub(self, rhs: $t) -> SciNum {
                self.sub_with_correlation(rhs.into(), Decimal::ZERO)
            }
        }

        impl Sub<SciNum> for $t {
            type Output = SciNum;

            fn sub(self, rhs: SciNum) -> SciNum {
                let num: SciNum = self.into();
                num.sub_with_correlation(rhs, Decimal::ZERO)
            }
        }

        impl Mul<$t> for SciNum {
            type Output = Self;

            fn mul(self, rhs: $t) -> SciNum {
                self.mul_with_correlation(rhs.into(), Decimal::ZERO)
            }
        }

        impl Mul<SciNum> for $t {
            type Output = SciNum;

            fn mul(self, rhs: SciNum) -> SciNum {
                let num: SciNum = self.into();
                num.mul_with_correlation(rhs, Decimal::ZERO)
            }
        }

        impl Div<$t> for SciNum {
            type Output = Self;

            fn div(self, rhs: $t) -> SciNum {
                self.div_with_correlation(rhs.into(), Decimal::ZERO)
            }
        }

        impl Div<SciNum> for $t {
            type Output = SciNum;

            fn div(self, rhs: SciNum) -> SciNum {
                let num: SciNum = self.into();
                num.div_with_correlation(rhs, Decimal::ZERO)
            }
        }
    };
}

impl_arithmetic_int!(i8);
impl_arithmetic_int!(i16);
impl_arithmetic_int!(i32);
impl_arithmetic_int!(i64);
impl_arithmetic_int!(u8);
impl_arithmetic_int!(u16);
impl_arithmetic_int!(u32);
impl_arithmetic_int!(u64);

impl Debug for SciNum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SciNum")
            .field("number", &Decimal::try_from(self.number()))
            .field("uncertainty", &Decimal::try_from(self.uncertainty()))
            .finish()
    }
}

impl fmt::Display for SciNum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let significand = self.significand;
        let uncertainty = if self.is_exact() {
            String::new()
        } else {
            format!("({})", self.uncertainty)
        };
        // Display up to five places normally
        // If the number has more than five places,
        // or insignificant zeros before the decimal point,
        // display in scientific notation
        if self.sigfigs() <= 5 && self.precision() <= 0 && self.precision() >= -5 {
            if self.precision() == 0 {
                write!(f, "{significand}{uncertainty}")
            } else {
                // 3.25e-2 is (325, -4), should be formatted as 0.0325
                let leading_zeros = "0".repeat((u32::from(self.precision().unsigned_abs()) - self.sigfigs()).try_into().unwrap());
                write!(f, "0.{leading_zeros}{significand}{uncertainty}")
            }
        // Otherwise, use scientific notation
        } else {
            let (int, zeros, frac, _, exp) = self.scientific_parts_normalized();
            let leading_zeros = "0".repeat(zeros.into());
            // Fractional part might not have any places at all (e.g. 2e6)
            if let Some(frac) = frac {
                write!(f, "{int}.{leading_zeros}{frac}{uncertainty}e{exp}")
            } else {
                write!(f, "{int}{uncertainty}e{exp}")
            }
        }
    }
}

impl FromStr for SciNum {
    type Err = SciNumError;

    /// Parses a string and attempts to create a corresponding `SciNum`.
    /// 
    /// Does not currently support uncertainties.
    /// 
    /// For now goes via `rust_decimal::Decimal::from_str()`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        //let re = Regex::new(r"^(-?\d+(?:[.,]\d+)?)(?:\((\d+)\))?(?:[eE]([+-]?\d+))?$").unwrap();
        let re = Regex::new(r"^(-)?(\d+)(?:[.,](\d+))?(?:\((\d+)\))?(?:[eE]([+-]?\d+))?$").unwrap();
        let caps = re.captures(s).ok_or(SciNumError::Parse(s.into()))?;
        // Example given with "6.971e-7"
        let negative = caps.get(1).is_some(); // false
        let mut significand_str = String::new();
        let int = caps.get(2).ok_or(SciNumError::Parse(s.into()))?.as_str(); // "6"
        significand_str.push_str(int);
        let frac = caps.get(3).map_or("", |m| m.as_str()); // "971"
        significand_str.push_str(frac);
        let significand = u64::from_str(&significand_str).map_err(|_e| SciNumError::Parse(s.into()))?; // "6971"
        let frac_places = frac.len(); // 3
        let uncertainty = caps.get(4).map_or(Ok(0), |m| u32::from_str(m.as_str())).map_err(|_e| SciNumError::Parse(s.into()))?; // 0
        let exponent = caps.get(5).map_or(Ok(0), |m| i16::from_str(m.as_str())).map_err(|_e| SciNumError::Parse(s.into()))?; // -7
        // "6.971e-7" should be represented as (6971, -10)
        Ok(Self {
            negative,
            exponent: exponent - (frac_places as i16),
            uncertainty_scale: 0,
            uncertainty,
            significand,
        })
    }
}

#[allow(unused_macros)]
macro_rules! sci {
    ($s:expr) => {
        SciNum::from_str(stringify!($s)).unwrap()
    };
}

impl SciNum {
    /// A constant representing 0.
    pub const ZERO: SciNum = SciNum {
        negative: false,
        exponent: 0,
        uncertainty_scale: 0,
        uncertainty: 0,
        significand: 0,
    };

    /// A constant representing 1.
    pub const ONE: SciNum = SciNum {
        negative: false,
        exponent: 0,
        uncertainty_scale: 0,
        uncertainty: 0,
        significand: 1,
    };

    /// The largest supported number.
    ///
    /// Exponent is limited to 0 until the arithmetic is reimplemented to not go
    /// via `Decimal`.
    pub const MAX: SciNum = SciNum {
        negative: false,
        exponent: 0,
        uncertainty_scale: 0,
        uncertainty: 0,
        significand: u64::MAX,
    };

    /// The smallest supported number.
    ///
    /// Exponent is limited to 0 until the arithmetic is reimplemented to not go
    /// via `Decimal`.
    pub const MIN: SciNum = SciNum {
        negative: true,
        exponent: 0,
        uncertainty_scale: 0,
        uncertainty: 0,
        significand: u64::MAX,
    };
}

// Constants taken from rust_decimal
#[allow(dead_code)]
pub mod dec {
    // Sign mask for the flags field. A value of zero in this bit indicates a
    // positive Decimal value, and a value of one in this bit indicates a
    // negative Decimal value.
    pub(crate) const SIGN_MASK: u32 = 0x8000_0000;
    pub(crate) const UNSIGN_MASK: u32 = 0x4FFF_FFFF;

    // Scale mask for the flags field. This byte in the flags field contains
    // the power of 10 to divide the Decimal value by. The scale byte must
    // contain a value between 0 and 28 inclusive.
    pub const SCALE_MASK: u32 = 0x00FF_0000;
    pub const U8_MASK: u32 = 0x0000_00FF;
    pub const U32_MASK: u64 = u32::MAX as _;

    // Number of bits scale is shifted by.
    pub const SCALE_SHIFT: u32 = 16;
    // Number of bits sign is shifted by.
    pub const SIGN_SHIFT: u32 = 31;
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    
    #[test]
    fn new_from_int() {
        // Using new
        let n = SciNum::new(30, 0);
        assert_eq!(n.number(), SciNum::new(30, 0));
        assert_eq!(n.uncertainty(), SciNum::new(0, 0));
        // Using from
        let n = SciNum::from(42);
        assert_eq!(n.number(), SciNum::new(42, 0));
        assert_eq!(n.uncertainty(), SciNum::new(0, 0));
    }

    #[test]
    fn new_from_int_with_uncertainty() {
        let n = SciNum::new_with_uncertainty(20, 2, 0);
        assert_eq!(n.number(), SciNum::from(20));
        assert_eq!(n.uncertainty(), SciNum::new(2, 0));
    }

    #[test]
    fn new_from_dec() {
        let n = SciNum::from(dec!(20));
        assert_eq!(n.number(), SciNum::new(20, 0));
        assert_eq!(n.number(), SciNum::from(dec!(20)));
        assert_eq!(n.uncertainty(), SciNum::new(0, 0));
        assert_eq!(n.uncertainty(), SciNum::from(dec!(0)));
    }

    #[test]
    fn from_scientific_parts() {
        let n1 = SciNum::from_scientific_parts(67, 2, 0, 0); // 67.2
        assert_eq!(n1.to_string(), "67.2");
        assert_eq!(n1, SciNum::new(670, -1));

        // Should always get at least one decimal place using this method
        let n2 = SciNum::from_scientific_parts(672, 0, 0, 0); // 672.0
        assert_eq!(n2.to_string(), "672.0");
        assert_eq!(n2, SciNum::new(6720, -1));

        let n3 = SciNum::from_scientific_parts(2, 36, 0, 5);
        assert_eq!(n3.to_string(), "2.36e5");
        assert_eq!(n3, SciNum::from(dec!(2.36e5)));

        let n4 = SciNum::from_scientific_parts(23, 61, 0, -7);
        assert_eq!(n4.to_string(), "2.361e-6");
        assert_eq!(n4, SciNum::from(dec!(2.361e-6)));
    }

    #[test]
    #[should_panic] // For now, not supported yet
    fn exact_from_scientific_parts_large() {
        let _n = SciNum::new(236, 40);
    }

    #[test]
    #[should_panic] // For now, not supported
    fn exact_from_scientific_parts_small() {
        let _n = SciNum::new(49, -76);
    }

    #[test]
    fn num_dec() {
        let n = SciNum::new_with_uncertainty(20, 2, 0);
        assert_eq!(n.number_dec(), dec!(20));
    }

    #[test]
    fn uncert_dec() {
        let n = SciNum::new_with_uncertainty(30, 5, 0);
        assert_eq!(n.uncertainty_dec(), dec!(5));
    }

    #[test]
    fn relative_uncertainty() {
        let n = SciNum::new_with_uncertainty(20, 2, 0);
        assert_eq!(n.relative_uncertainty_dec(), dec!(0.1));

        let n2 = SciNum::new_with_uncertainty(500, 5, 0);
        assert_eq!(n2.relative_uncertainty_dec(), dec!(0.01));

        let n3 = SciNum::new_with_uncertainty(1000, 15, 0);
        assert_eq!(n3.relative_uncertainty_dec(), dec!(0.015));
    }

    #[test]
    fn sigfigs() {
        let n = SciNum::from_scientific_parts(123, 45, 0, 0);
        assert_eq!(n.sigfigs(), 5);

        let n2 = SciNum::new(dec!(0.00123));
        assert_eq!(n2.sigfigs(), 3);

        let n3 = SciNum::new(dec!(1234));
        assert_eq!(n3.sigfigs(), 4);
    }

    #[test]
    fn sigfigs_trailing_zeros() {
        let n = SciNum::new(dec!(123.4500));
        assert_eq!(n.sigfigs(), 7);

        let n2 = SciNum::new(dec!(0.001230));
        assert_eq!(n2.sigfigs(), 4);

        let n3 = SciNum::new(dec!(1230));
        assert_eq!(n3.sigfigs(), 4);
    }

    #[test]
    fn precision() {
        assert_eq!(SciNum::new(dec!(0.02)).precision(), -2);
        assert_eq!(SciNum::new(dec!(0.020)).precision(), -3);
        assert_eq!(SciNum::new(Decimal::TWO).precision(), 0);
        //assert_eq!(SciNum::new(dec!(2e3)).precision(), 3); // Fails for
        // now
    }

    #[test]
    fn is_exact() {
        let n1 = SciNum::new(dec!(45.1));
        let n2 = SciNum::new_with_uncertainty(500, 5);
        assert!(n1.is_exact());
        assert!(!n2.is_exact());
    }

    #[test]
    fn addition_fn_exact() {
        let n1 = SciNum::new(40);
        let n2 = SciNum::new(dec!(5.1));
        let result = n1.add_with_correlation(n2, 0);
        assert_eq!(result.number_dec(), dec!(45.1));
    }

    #[test]
    fn addition_fn() {
        let n1 = SciNum::new_with_uncertainty(20, 2);
        let n2 = SciNum::new_with_uncertainty(30, 5);
        let result = n1.add_with_correlation(n2, 0);
        assert_eq!(result.number_dec(), dec!(50));
        assert_eq!(
            result.uncertainty_dec().round_dp(5),
            dec!(5.3851648071345).round_dp(5)
        );
    }

    #[test]
    fn addition_op() {
        let n1 = SciNum::new_with_uncertainty(20, 2);
        let n2 = SciNum::new_with_uncertainty(30, 5);
        let result = n1 + n2;
        assert_eq!(result.number_dec(), dec!(50));
        assert_eq!(
            result.uncertainty_dec().round_dp(5),
            dec!(5.3851648071345).round_dp(5)
        );
    }

    #[test]
    fn addition_with_int() {
        let n1 = SciNum::new_with_uncertainty(20, 0);
        let n2 = 30;
        let result: SciNum = n1 + n2;
        assert_eq!(result.number_dec(), dec!(50));
    }

    #[test]
    fn subtraction() {
        let n1 = SciNum::new_with_uncertainty(20, 2);
        let n2 = SciNum::new_with_uncertainty(30, 5);
        let result = n1 - n2;
        assert_eq!(result.number_dec(), dec!(-10));
        assert_eq!(
            result.uncertainty_dec().round_dp(5),
            dec!(5.3851648071345).round_dp(5)
        );
    }

    #[test]
    fn subtraction_with_int() {
        let n1 = SciNum::new_with_uncertainty(20, 0);
        let n2 = 30;
        let result: SciNum = n1 - n2;
        assert_eq!(result.number_dec(), dec!(-10));
    }

    #[test]
    fn multiplication() {
        let n1 = SciNum::new_with_uncertainty(20, 2);
        let n2 = SciNum::new_with_uncertainty(30, 5);
        let result = n1 * n2;
        assert_eq!(result.number_dec(), dec!(600));
        assert_eq!(
            result.uncertainty_dec().round_dp(5),
            dec!(116.619037896906).round_dp(5)
        );
        let ft = SciNum::new(dec!(0.3048));
        let square_ft = ft * ft;
        assert_eq!(square_ft.number_dec(), dec!(0.09290304));
    }

    #[test]
    fn multiplication_with_int() {
        let n1 = SciNum::new_with_uncertainty(20, 0);
        let n2 = 30;
        let result: SciNum = n1 * n2;
        assert_eq!(result.number_dec(), dec!(600));
    }

    #[test]
    fn division() {
        let n1 = SciNum::new_with_uncertainty(20, 2);
        let n2 = SciNum::new_with_uncertainty(30, 5);
        let result = n1 / n2;
        assert_eq!(
            result.number_dec().round_dp(10),
            dec!(0.6666666667).round_dp(10)
        );
        assert_eq!(
            result.uncertainty_dec().round_dp(5),
            dec!(0.129576708774340).round_dp(5)
        );
    }

    #[test]
    fn division_with_int() {
        let n1 = SciNum::new_with_uncertainty(60, 0);
        let n2 = 30;
        let result: SciNum = n1 / n2;
        assert_eq!(result.number_dec(), Decimal::TWO);
    }

    #[test]
    fn division_reversed() {
        let n1 = SciNum::new_with_uncertainty(20, 2);
        let n2 = SciNum::new_with_uncertainty(30, 5);
        let result = n2 / n1;
        assert_eq!(result.number_dec(), dec!(1.5));
        assert_eq!(
            result.uncertainty_dec().round_dp(5),
            dec!(0.2915475947422).round_dp(5)
        );
    }

    #[test]
    fn exponentiation() {
        let n1 = SciNum::new_with_uncertainty(20, 2);

        let result = n1.powd(Decimal::TWO);
        assert_eq!(result.number_dec(), dec!(400));
        assert_eq!(result.uncertainty_dec(), dec!(80));

        let result = n1.powi(2);
        assert_eq!(result.number_dec(), dec!(400));
        assert_eq!(result.uncertainty_dec(), dec!(80));
    }

    #[test]
    fn natural_log() {
        let n1 = SciNum::new_with_uncertainty(20, 2);
        let n2 = SciNum::new_with_uncertainty(30, 5);
        let ratio = n1 / n2;
        let result = ratio.ln();
        assert_eq!(
            result.uncertainty_dec().round_dp(5),
            dec!(0.194365063161).round_dp(5)
        );
    }

    #[test]
    fn log_base10() {
        let n1 = SciNum::new_with_uncertainty(20, 2);
        let n2 = SciNum::new_with_uncertainty(30, 5);
        let ratio = n1 / n2;
        let result = ratio.log10();
        assert_eq!(
            result.uncertainty_dec().round_dp(5),
            dec!(0.08441167440582).round_dp(5)
        );
    }

    #[test]
    fn exponential() {
        let n1 = SciNum::new_with_uncertainty(20, 2);
        let n2 = SciNum::new_with_uncertainty(30, 5);
        let ratio = n1 / n2;
        let result = ratio.exp();
        assert_eq!(
            result.uncertainty_dec().round_dp(5),
            dec!(0.25238096660761).round_dp(5)
        );
    }

    #[test]
    fn debug() {
        let n = SciNum::new_with_uncertainty(20, 2);
        assert_eq!(format!("{n:?}"), "SciNum { number: 20, uncertainty: 2 }");
    }

    #[test]
    fn display() {
        // Small integers display normally
        assert_eq!(SciNum::new(20).to_string(), "20");
        // Up to 5 places displays normally
        assert_eq!(SciNum::new(99999).to_string(), "99999");
        assert_eq!(SciNum::new(dec!(0.00001)).to_string(), "0.00001");
        // Above 6 places uses scientific notation
        assert_eq!(SciNum::new(1295891).to_string(), "1.295891e6");
        assert_eq!(SciNum::new(dec!(0.000000432)).to_string(), "4.32e-7");
        // Explicit zeros should be treated as significant
        assert_eq!(SciNum::new(1295800).to_string(), "1.295800e6");
        // Here they shouldn't be but the problem is that Decimal does treat them as
        // significant... assert_eq!(SciNum::new(dec!(1.2958e6)).
        // to_string(), "1.2958e6"); Check uncertainty formatting
        assert_eq!(SciNum::new_with_uncertainty(20, 2).to_string(), "20±2");
        // TODO: More uncertainty display tests
    }

    #[test]
    fn from_str() {
        // Integer
        assert_eq!(SciNum::from_str("42").unwrap(), SciNum::new(dec!(42)));
        // Negative float
        assert_eq!(
            SciNum::from_str("-3.14").unwrap(),
            SciNum::new(dec!(-3.14))
        );
        // Scientific notation
        assert_eq!(
            SciNum::from_str("1.5e8").unwrap(),
            SciNum::new(dec!(1.5e8))
        );
        // TODO large exponent fails with overflow error
        //assert_eq!(SciNum::from_str("1.5e10").unwrap(),
        // SciNum::new(dec!(1.5e10))); Scientific notation with negative
        // exponent
        assert_eq!(
            SciNum::from_str("2e-5").unwrap(),
            SciNum::new(dec!(2e-5))
        );
        // Negative number with positive exponent
        assert_eq!(
            SciNum::from_str("-6.022e6").unwrap(),
            SciNum::new(dec!(-6.022e6))
        );
        // TODO large exponent fails with overflow error
        //assert_eq!(SciNum::from_str("-6.022e23").unwrap(),
        // SciNum::new(dec!(-6.022e23))); Capital E for exponent
        assert_eq!(
            SciNum::from_str("1.5E8").unwrap(),
            SciNum::new(dec!(1.5E8))
        );
        // Make sure incorrectly formatted string fails
        assert!(SciNum::from_str("not a number").is_err());
    }

    #[test]
    fn sci_macro() {
        // Integer
        assert_eq!(sci!(42), SciNum::new(42, 0));
        // Negative float
        assert_eq!(sci!(-3.14), SciNum::from_scientific_parts(-3, 14, 0, 0));
        // Scientific notation
        assert_eq!(sci!(1.5e8), SciNum::new(15, 7));
        // Scientific notation with large exponent
        assert_eq!(sci!(1.5e10), SciNum::new(15, 9));
        // Scientific notation with negative exponent
        assert_eq!(sci!(2e-5), SciNum::new(2, -5));
        // Negative number with positive exponent
        assert_eq!(sci!(-6.022e6), SciNum::new(-6022, 3));
        // Negative number with large exponent
        assert_eq!(sci!(-6.022e23), SciNum::new(-6022, 20));
        // Capital E for exponent
        assert_eq!(sci!(1.5E8), SciNum::new(15, 7));
    }
}
