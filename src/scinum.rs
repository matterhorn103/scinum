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
use rust_decimal_macros::dec;

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
    uncertainty_scale: u8,
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
    /// # Example
    ///
    /// ```
    /// # use scinum::SciNum;
    /// #
    /// let n = SciNum::new(251, -3).with_uncertainty(3);
    /// assert_eq!(n.to_string(), "0.251(3)");
    /// assert_eq!(n, SciNum::new_with_uncertainty(251, 3, -3));
    #[inline]
    pub fn with_uncertainty(mut self, uncertainty: u32) -> Self {
        self.uncertainty = uncertainty;
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

    /// Returns a tuple of the integer, fractional, uncertainty, and exponent
    /// parts of the of the number when represented with normalized
    /// notation i.e. with 10 > _m_ >= 1.
    ///
    /// Corresponds to `(ii, ffff, uu, nn)` when the number is notated as
    /// `ii.ffff(uu) × 10^nn`.
    pub fn scientific_parts_normalized(&self) -> (i8, Option<i128>, u32, i16) {
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
        let sigma_ab = correlation.into()
            * Decimal::try_from(self.uncertainty())
            * Decimal::try_from(rhs.uncertainty());
        let number = Decimal::try_from(self.number()) + Decimal::try_from(rhs.number());
        let uncertainty = if self.is_exact() && rhs.is_exact() {
            Decimal::ZERO
        } else {
            ((Decimal::try_from(self.uncertainty()).powu(2))
                + (Decimal::try_from(rhs.uncertainty()).powu(2))
                + (dec!(2) * sigma_ab))
                .sqrt()
                .unwrap()
        };
        Self::new_with_uncertainty(number, uncertainty)
    }

    pub fn sub_with_correlation<T>(self, rhs: Self, correlation: T) -> Self
    where
        T: Into<Decimal>,
    {
        let sigma_ab = correlation.into()
            * Decimal::try_from(self.uncertainty()).unwrap()
            * Decimal::try_from(rhs.uncertainty()).unwrap();
        let number = Decimal::try_from(self.number()).unwrap() - Decimal::try_from(rhs.number()).unwrap();
        let uncertainty = if self.is_exact() && rhs.is_exact() {
            Decimal::ZERO
        } else {
            ((Decimal::try_from(self.uncertainty()).unwrap().powu(2))
                + (Decimal::try_from(rhs.uncertainty()).unwrap().powu(2))
                - (dec!(2) * sigma_ab))
                .sqrt()
                .unwrap()
        };
        Self::new_with_uncertainty(number, uncertainty)
    }

    pub fn mul_with_correlation<T>(self, rhs: Self, correlation: T) -> Self
    where
        T: Into<Decimal>,
    {
        let sigma_ab = correlation.into()
            * Decimal::try_from(self.uncertainty()).unwrap()
            * Decimal::try_from(rhs.uncertainty()).unwrap();
        let number = Decimal::try_from(self.number()).unwrap() * Decimal::try_from(rhs.number()).unwrap();
        let uncertainty = if self.is_exact() && rhs.is_exact() {
            Decimal::ZERO
        } else {
            ((Decimal::try_from(self.relative_uncertainty()).unwrap().powu(2))
                + (Decimal::try_from(rhs.relative_uncertainty()).unwrap().powu(2))
                + (dec!(2) * sigma_ab / number))
                .sqrt()
                .unwrap()
                * number.abs()
        };
        Self::new_with_uncertainty(number, uncertainty)
    }

    pub fn div_with_correlation<T>(self, rhs: Self, correlation: T) -> Self
    where
        T: Into<Decimal>,
    {
        let sigma_ab = correlation.into()
            * Decimal::try_from(self.uncertainty()).unwrap()
            * Decimal::try_from(rhs.uncertainty()).unwrap();
        let number = Decimal::try_from(self.number()).unwrap() / Decimal::try_from(rhs.number()).unwrap();
        let uncertainty = if self.is_exact() && rhs.is_exact() {
            Decimal::ZERO
        } else {
            ((Decimal::try_from(self.relative_uncertainty()).unwrap().powu(2))
                + (Decimal::try_from(rhs.relative_uncertainty()).unwrap().powu(2))
                - (dec!(2) * sigma_ab / number))
                .sqrt()
                .unwrap()
                * number.abs()
        };
        Self::new_with_uncertainty(number, uncertainty)
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
        let sigma_ab = correlation.into()
            * Decimal::try_from(self.uncertainty()).unwrap()
            * Decimal::try_from(rhs.uncertainty()).unwrap();
        let number = Decimal::try_from(self.number()).unwrap().powd(Decimal::try_from(rhs.number()).unwrap());
        let uncertainty = if self.is_exact() && rhs.is_exact() {
            Decimal::ZERO
        } else {
            ((Decimal::try_from(self.relative_uncertainty()).unwrap() * Decimal::try_from(rhs.number()).unwrap()).powu(2)
                + (Decimal::try_from(self.number()).unwrap().ln() * Decimal::try_from(rhs.uncertainty()).unwrap()).powu(2)
                + (dec!(2)
                    * ((Decimal::try_from(self.number()).unwrap().ln() * Decimal::try_from(rhs.number()).unwrap())
                        / Decimal::try_from(self.number()).unwrap())
                    * sigma_ab))
                .sqrt()
                .unwrap()
                * number.abs()
        };
        Self::new_with_uncertainty(number, uncertainty)
    }

    pub fn ln(self) -> Self {
        let number = Decimal::try_from(self.number()).unwrap().ln();
        let uncertainty = Decimal::try_from(self.relative_uncertainty()).unwrap().abs();
        Self::new_with_uncertainty(number, uncertainty)
    }

    pub fn log10(self) -> Self {
        let number = Decimal::try_from(self.number()).unwrap().log10();
        let uncertainty = (Decimal::try_from(self.uncertainty()).unwrap()
            / (Decimal::TEN.ln() * Decimal::try_from(self.number()).unwrap()))
        .abs();
        Self::new_with_uncertainty(number, uncertainty)
    }

    pub fn exp(self) -> Self {
        let number = Decimal::try_from(self.number()).unwrap().exp();
        let uncertainty = number.abs() * Decimal::try_from(self.uncertainty()).unwrap();
        Self::new_with_uncertainty(number, uncertainty)
    }
}

impl Num for SciNum {
    type FromStrRadixErr = rust_decimal::Error;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, rust_decimal::Error> {
        // For now, just make use of the Decimal implementation
        let dec = Decimal::try_from_str_radix(str, radix)?;
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
    fn from(n: Decimal) -> Self {
        let n = n.unpack();
        // TODO: Handle gracefully when precision is too high
        // (should just drop the excess)
        let significand = if n.hi != 0 {
            todo!()
        } else {
            (n.mid as u64) << 32 | n.lo as u64
        };
        Self {
            negative: n.negative,
            exponent: -(n.scale as i16), // Scale is max 28 so this is fine
            uncertainty_scale: 0,
            uncertainty: 0,
            significand,
        }
    }
}

impl TryFrom<SciNum> for Decimal {
    type Error = SciNumError;

    fn try_from(n: SciNum) -> Result<Self, Self::Error> {
        if n.exponent.is_positive() || (u32::from(n.exponent.unsigned_abs()) > Decimal::MAX_SCALE) {
            Err(SciNumError::Cast("Decimal".to_string()))
        } else {
            Ok(Decimal::try_from_i128_with_scale(
                n.significand_integral(),
                n.exponent.unsigned_abs().into(),
            ))
        }
    }
}

//impl FromPrimitive for Number {
//    fn from_i64(n: i64) -> Option<Self> {
//        Some(Self {
//            number: n.into(),
//            uncertainty: Decimal::ZERO,
//        })
//    }
//
//    fn from_u64(n: u64) -> Option<Self> {
//        Some(Self {
//            number: n.into(),
//            uncertainty: Decimal::ZERO,
//        })
//    }
//}

macro_rules! impl_from {
    ($T:ty) => {
        impl From<$T> for SciNum {
            fn from(t: $T) -> Self {
                Self::new_exact(t)
            }
        }
    };
}

impl_from!(i8);
impl_from!(i16);
impl_from!(i32);
impl_from!(i64);
impl_from!(i128);
impl_from!(isize);
impl_from!(u8);
impl_from!(u16);
impl_from!(u32);
impl_from!(u64);
impl_from!(u128);
impl_from!(usize);

impl PartialEq for SciNum {
    fn eq(&self, other: &Self) -> bool {
        Decimal::try_from(self.number()) == other.number_dec()
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
        Decimal::try_from(self.number()).cmp(&other.number_dec())
    }
}

macro_rules! impl_comparisons {
    ($t:ty) => {
        impl PartialEq<$t> for SciNum {
            fn eq(&self, other: &$t) -> bool {
                Decimal::try_from(self.number()) == Decimal::try_from(*other)
            }
        }

        impl PartialOrd<$t> for SciNum {
            fn partial_cmp(&self, other: &$t) -> Option<std::cmp::Ordering> {
                Decimal::try_from(self.number()).partial_cmp(&Decimal::try_from(*other))
            }
        }
    };
}

impl_comparisons!(i8);
impl_comparisons!(i16);
impl_comparisons!(i32);
impl_comparisons!(i64);
impl_comparisons!(i128);
impl_comparisons!(isize);
impl_comparisons!(u8);
impl_comparisons!(u16);
impl_comparisons!(u32);
impl_comparisons!(u64);
impl_comparisons!(u128);
impl_comparisons!(usize);
impl_comparisons!(Decimal);

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

macro_rules! impl_arithmetic {
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

impl_arithmetic!(i8);
impl_arithmetic!(i16);
impl_arithmetic!(i32);
impl_arithmetic!(i64);
impl_arithmetic!(i128);
impl_arithmetic!(isize);
impl_arithmetic!(u8);
impl_arithmetic!(u16);
impl_arithmetic!(u32);
impl_arithmetic!(u64);
impl_arithmetic!(u128);
impl_arithmetic!(usize);

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
        if self.is_exact() {
            // Up to five decimal places, display normally
            if (self < &dec!(1e6)) && (self >= &dec!(1e-5)) {
                write!(f, "{}", Decimal::try_from(self.number()))
            // Otherwise, use scientific notation
            } else {
                let (int, frac, exp) = self.scientific_parts_normalized();
                // Fractional part might not have any places at all (e.g. 2e6)
                let frac_string = match frac {
                    Some(n) => n.to_string(),
                    None => String::new(),
                };
                write!(f, "{int}.{frac_string}e{exp}")
            }
        } else {
            // TODO Need to add support for ASCII +/-
            // Default representation should be parentheses notation in future though
            write!(f, "{}±{}", self.number(), self.uncertainty())
        }
    }
}

impl FromStr for SciNum {
    type Err = SciNumError;

    /// Parses a string and attempts to create a corresponding `SciNum`.
    /// Does not currently support uncertainties.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let re = Regex::new(r"^(-?\d+(?:[.,]\d+)?)(?:[eE]([+-]?\d+))?$").unwrap();
        let caps = re.captures(s).ok_or(SciNumError::Parse(s.into()))?;
        let number_str = caps.get(1).ok_or(SciNumError::Parse(s.into()))?.as_str();
        let number = Decimal::try_from_str(number_str).map_err(|_e| SciNumError::Parse(s.into()))?;
        let exponent_str = caps.get(2).map(|m| m.as_str()).unwrap_or("0");
        let exponent = i16::from_str(exponent_str).map_err(|_e| SciNumError::Parse(s.into()))?;
        Ok(Self::new(number, exponent))
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
        let n = SciNum::new_with_uncertainty(20, 2, 0);
        assert_eq!(n.number(), SciNum::new(20, 0));
        assert_eq!(n.uncertainty(), SciNum::new(2, 0));
    }

    #[test]
    fn new_exact_from_int() {
        let n = SciNum::new(30, 0);
        assert_eq!(n.number(), SciNum::new(30, 0));
        assert_eq!(n.uncertainty(), SciNum::new(0, 0));
    }

    #[test]
    fn new_from_dec() {
        let n = SciNum::new_with_uncertainty(dec!(30), dec!(5));
        assert_eq!(n.number(), SciNum::new_with_uncertainty(dec!(30), dec!(0)));
        assert_eq!(
            n.uncertainty(),
            SciNum::new_with_uncertainty(dec!(5), dec!(0))
        );
    }

    #[test]
    fn new_exact_from_dec() {
        let n = SciNum::new(dec!(20));
        assert_eq!(n.number(), SciNum::new(dec!(20)));
        assert_eq!(n.uncertainty(), SciNum::new(dec!(0)));
    }

    #[test]
    fn exact_from_scientific_parts() {
        let n = SciNum::new(67, 0);
        assert_eq!(n, SciNum::new(dec!(67)));
        let n2 = SciNum::new(236, 3);
        assert_eq!(n2, SciNum::new(dec!(2.36e5)));
        let n3 = SciNum::new(236, -6);
        assert_eq!(n3, SciNum::new(dec!(2.36e-4)));
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
        assert_eq!(SciNum::new(dec!(2)).precision(), 0);
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
        assert_eq!(result.number_dec(), dec!(2));
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

        let result = n1.powd(dec!(2));
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
        assert_eq!(sci!(42), SciNum::new(dec!(42)));
        // Negative float
        assert_eq!(sci!(-3.14), SciNum::new(dec!(-3.14)));
        // Scientific notation
        assert_eq!(sci!(1.5e8), SciNum::new(dec!(1.5e8)));
        // TODO large exponent fails with overflow error
        //assert_eq!(sci!(1.5e10), SciNum::new(dec!(1.5e10)));
        // Scientific notation with negative exponent
        assert_eq!(sci!(2e-5), SciNum::new(dec!(2e-5)));
        // Negative number with positive exponent
        assert_eq!(sci!(-6.022e6), SciNum::new(dec!(-6.022e6)));
        // TODO large exponent fails with overflow error
        //assert_eq!(sci!(-6.022e23), SciNum::new(dec!(-6.022e23)));
        // Capital E for exponent
        assert_eq!(sci!(1.5E8), SciNum::new(dec!(1.5E8)));
    }
}
