// SPDX-FileCopyrightText: 2025 Matthew Milner <matterhorn103@proton.me>
// SPDX-License-Identifier: MIT

use std::{
    fmt::{self, Debug},
    ops::{Add, Div, Mul, Rem, Sub},
    str::FromStr,
};

use num_traits::{FromPrimitive, Inv, Num, One, Pow, Zero, real::Real};
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
/// 16 decimal digits of precision) and an exponent from −16383 to 16384 
/// (represented as a 15-bit biased unsigned integer).
/// As such, all values covered by the IEEE 754-2008 `binary64` (i.e. `f64`)
/// and `decimal64` formats are representable.
///
/// Rounding, formatting, and parsing methods are provided with a view to
/// enabling typical scientific calculations.
#[derive(Copy, Clone, Debug, serde_with::DeserializeFromStr, serde_with::SerializeDisplay)]
pub struct SciNum {
    uncertainty: u32,
    uncertainty_scale: u8, // This will allow the uncertainty to have a different precision, but for the moment must always be 0
    // Have the uncertainty come first so that the bits used for comparisons are
    // the 80/81 least significant bits, just like in IEEE floating point formats
    negative: bool,
    exponent: u16, // Biased 15-bit exponent with bias b = 2^14 - 1 = 16383
    significand: u64,
}

// Only use 15 of the 16 bits so that we can fit the sign and exponent into 16
// bits later if so desired
const UEXPONENT_BIAS: u16 = 16383;
const UMIN_EXPONENT: u16 = 0;
const UMAX_EXPONENT: u16 = 32767;
const EXPONENT_BIAS: i16 = 16383;
const MIN_EXPONENT: i16 = -16383;
const MAX_EXPONENT: i16 = 16384;

impl SciNum {
    /// Creates an exact `SciNum` from parts corresponding to _m_ ×
    /// 10<sup><i>n</i></sup>.
    /// 
    /// # Panics
    /// 
    /// This function panics if `exponent` is outside of the range −16383 to 16384.
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
        if !(MIN_EXPONENT..=MAX_EXPONENT).contains(&exponent) {
            panic!()
        }
        Self {
            uncertainty: 0,
            uncertainty_scale: 0,
            negative: number.is_negative(),
            exponent: (exponent + EXPONENT_BIAS) as u16,
            significand: number.unsigned_abs() as u64,
        }
    }

    /// Creates a `SciNum` from parts corresponding to (_m_ ± _u_) ×
    /// 10<sup><i>n</i></sup>.
    ///
    /// This means the number of decimal places in the number and uncertainty
    /// will be the same in the created `SciNum`.
    /// 
    /// # Panics
    /// 
    /// This function panics if `exponent` is outside of the range −16383 to 16384.
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
        if !(MIN_EXPONENT..=MAX_EXPONENT).contains(&exponent) {
            panic!()
        }
        Self {
            uncertainty,
            uncertainty_scale: 0,
            negative: number.is_negative(),
            exponent: (exponent + EXPONENT_BIAS) as u16,
            significand: number.unsigned_abs() as u64,
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
        self.uncertainty_scale = (self.exponent - uncertainty.exponent).try_into().expect("Difference in precision of number and uncertainty should never be this large!");
        self.uncertainty = uncertainty.significand.try_into().expect("The uncertainty may not have a significand greater than `u32::MAX`!");
        self
    }
    
    /// Creates a `SciNum` from separate parts of a representation of the number in
    /// scientific notation.
    /// 
    /// The arguments should correspond to `(ii, z, fff, uu, nn)` when the number is
    /// notated as `ii.{zeros}fff(uu) × 10^nn`, where `z` is the number of leading
    /// zeros in the fractional part.
    /// 
    /// Trailing zeros in `fraction` are treated as significant, but leading zeros
    /// are not. If `fraction` is simply `0`, it is then also treated as
    /// insignificant. Passing `0` for both `zeros` and `fraction` therefore
    /// creates a `SciNum` with a significand equal to `integer`.
    /// 
    /// To create a number with only significant zeros in the fractional part (such
    /// as `2.0`), pass `0` for `fraction` and specify the appropriate number of
    /// zeros as `zeros`.
    /// 
    /// # Panics
    /// 
    /// This function panics if the overall significand does not fit into `u64`,
    /// or if `exponent` is outside of the range −16383 to 16384.
    ///
    /// # Example
    ///
    /// ```
    /// # use scinum::SciNum;
    /// #
    /// let n = SciNum::from_scientific_parts(2, 0, 51, 0, 0);
    /// assert_eq!(n.to_string(), "2.51");
    /// let n = SciNum::from_scientific_parts(2, 1, 51, 0, 0);
    /// assert_eq!(n.to_string(), "2.051");
    /// let n = SciNum::from_scientific_parts(2, 0, 51, 3, 0);
    /// assert_eq!(n.to_string(), "2.51(3)");
    /// let n = SciNum::from_scientific_parts(2, 0, 51, 3, -1);
    /// assert_eq!(n.to_string(), "0.251(3)");
    /// let n = SciNum::from_scientific_parts(2, 2, 0, 3, -2);
    /// assert_eq!(n.to_string(), "0.0200(3)");
    /// ```
    pub fn from_scientific_parts(
        integer: i8,
        zeros: u8,
        fraction: u64,
        uncertainty: u32,
        exponent: i16,
    ) -> Self {
        if !(MIN_EXPONENT..=MAX_EXPONENT).contains(&exponent) {
            panic!()
        }
        let unsigned_integer: u64 = integer.unsigned_abs().into();
        let (significand, exponent) = {
            if fraction != 0 {
                let decimal_places = fraction.ilog10() + 1;
                let significand = (unsigned_integer * 10_u64.pow(decimal_places + zeros as u32)) + fraction;
                let exponent = exponent - (decimal_places as i16);
                (significand, exponent)
            } else {
                (unsigned_integer, exponent)
            }
        };
        Self {
            uncertainty,
            uncertainty_scale: 0,
            negative: integer.is_negative(),
            exponent: (exponent + EXPONENT_BIAS) as u16,
            significand,
        }
    }

    /// Returns the integer part, number of fractional leading zeros,
    /// fractional part, uncertainty, and exponent of the number when represented
    /// with normalized notation i.e. with 10 > _m_ >= 1.
    ///
    /// Corresponds to `(ii, z, fff, uu, nn)` when the number is notated as
    /// `ii.{zeros}fff(uu) × 10^nn`, where `z` is the number of leading zeros
    /// in the fractional part.
    pub fn scientific_parts(&self) -> (i8, u8, u64, u32, i16) {
        if self.is_zero() {
            return (0, 0, 0, 0, 0)
        };
        let figs = self.sigfigs() as u32;
        let int_unsigned = self.significand / 10_u64.pow(figs - 1); // First digit
        let int = if self.negative {-(int_unsigned as i8)} else { int_unsigned as i8 };
        let frac = self.significand % 10_u64.pow(figs - 1);
        // Work out how many zeros have been dropped, if any
        let figs_in_frac = frac.checked_ilog10().map_or(0, |x| x + 1);
        let zeros = (figs - 1 - figs_in_frac) as u8; // 1 is for integer digit
        let uncert = self.uncertainty;
        let exp = self.exponent() + (figs as i16 - 1);
        // For example:
        // 1.23e2 = 123 is stored as (123, 0)       =>  2 =  0 + (3 - 1)
        // 4.5e6 = 4_500_000 is stored as (45, 5)   =>  6 =  5 + (2 - 1)
        // 4.5e-3 = 0.0045 is stored as (45, -4)    => -3 = -4 + (2 - 1)
        // 4.51e-3 = 0.00451 is stored as (451, -5) => -3 = -5 + (3 - 1)
        // 4.50e-3 = 0.00450 is stored as (450, -5) => -3 = -5 + (3 - 1)
        (int, zeros, frac, uncert, exp)
    }

    /// Returns the number as an exact `SciNum` without its uncertainty.
    #[inline]
    pub fn number(&self) -> Self {
        Self {
            uncertainty: 0,
            uncertainty_scale: 0,
            ..*self
        }
    }

    /// Returns the absolute uncertainty as an exact `SciNum`.
    ///
    /// The uncertainty is always positive.
    #[inline]
    pub fn uncertainty(&self) -> Self {
        Self {
            uncertainty: 0,
            uncertainty_scale: 0,
            negative: false,
            exponent: self.exponent,
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
    pub fn exponent(&self) -> i16 {
        self.exponent as i16 - EXPONENT_BIAS
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
    pub fn sigfigs(&self) -> u8 {
        if let Some(log) = self.significand.checked_ilog10() {
            log as u8 + 1
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
        self.exponent()
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
    pub fn powd(self, rhs: Decimal) -> Self {
        self.pow_with_correlation(rhs.into(), Decimal::ZERO)
    }

    #[inline]
    pub fn powfloat(self, rhs: f64) -> Self {
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
}

// Rounding functions
impl SciNum {
    /// Removes significant figures from the significand until the desired number
    /// is reached.
    /// 
    /// Equivalent to rounding towards zero.
    /// 
    /// The uncertainty of the `SciNum` is left unchanged.
    /// 
    /// # Panics
    /// 
    /// This function panics if the `SciNum` already has fewer significant figures
    /// than the requested number.
    pub fn truncate_sf(mut self, sf: u8) -> Self {
        if self.sigfigs() < sf { panic!() };
        while self.sigfigs() > sf {
            self.significand /= 10;
            // Exponent is now too small
            self.exponent += 1;
            // Uncertainty is now too large
            self.uncertainty_scale += 1;
        };
        self
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

// Methods that will belong to the Real trait if we implement it properly later
// impl Real for SciNum {
impl SciNum {
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

    fn abs(self) -> Self {
        Self {
            negative: false,
            ..self
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

    #[inline]
    pub fn powi(self, rhs: i32) -> Self {
        self.powd(rhs.into())
    }

    //fn powf(self, n: Self) -> Self {
    //    todo!()
    //}

    fn sqrt(self) -> Self {
        if self.is_sign_negative() { panic!() };
        self.powi(-2)
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

    //fn exp2(self) -> Self {
    //    todo!()
    //}

    pub fn ln(self) -> Self {
        let number = Decimal::try_from(self.number()).unwrap().ln();
        if self.is_exact() {
            Self::from(number)
        } else {
            let uncertainty = Decimal::try_from(self.relative_uncertainty()).unwrap().abs();
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

    //fn to_degrees(self) -> Self {
    //    todo!()
    //}

    //fn to_radians(self) -> Self {
    //    todo!()
    //}

    //fn max(self, other: Self) -> Self {
    //    todo!()
    //}

    //fn min(self, other: Self) -> Self {
    //    todo!()
    //}

    //fn abs_sub(self, other: Self) -> Self {
    //    todo!()
    //}

    //fn cbrt(self) -> Self {
    //    todo!()
    //}

    //fn hypot(self, other: Self) -> Self {
    //    todo!()
    //}

    //fn sin(self) -> Self {
    //    todo!()
    //}

    //fn cos(self) -> Self {
    //    todo!()
    //}

    //fn tan(self) -> Self {
    //    todo!()
    //}

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

    //fn sin_cos(self) -> (Self, Self) {
    //    todo!()
    //}

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
        if n.exponent > UEXPONENT_BIAS {
            Err(rust_decimal::Error::ConversionTo("Decimal".to_string()))
        } else {
            Decimal::try_from_i128_with_scale(
                n.significand_integral(),
                n.exponent().unsigned_abs().into(),
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
        if self.is_zero() {
            other.is_zero()
        } else if self.is_sign_negative() != other.is_sign_negative() {
            false
        } else if self.exponent == other.exponent {
            self.significand == other.significand
        } else if self.significand.is_multiple_of(other.significand) {
            let factor = self.significand / other.significand;
            if factor.is_multiple_of(10) {
                let order_diff = factor.ilog10();
                self.exponent + order_diff as u16 == other.exponent
            } else {
                false
            }
        } else if other.significand.is_multiple_of(self.significand) {
            let factor = other.significand / self.significand;
            if factor.is_multiple_of(10) {
                let order_diff = factor.ilog10();
                other.exponent + order_diff as u16 == self.exponent
            } else {
                false
            }
        } else {
            false
        }
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
        let number = Decimal::try_from(self.number()).unwrap() + Decimal::try_from(rhs.number()).unwrap();
        if self.is_exact() && rhs.is_exact() {
            Self::from(number)
        } else {
            let uncertainty = ((Decimal::try_from(self.uncertainty()).unwrap().powu(2))
                + (Decimal::try_from(rhs.uncertainty()).unwrap().powu(2)))
                .sqrt()
                .unwrap();
            Self::from(number).with_uncertainty(uncertainty.into())
        }
    }
}

impl Add for &SciNum {
    type Output = SciNum;

    fn add(self, rhs: Self) -> SciNum {
        *self + *rhs
    }
}

impl Sub for SciNum {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let number = Decimal::try_from(self.number()).unwrap() - Decimal::try_from(rhs.number()).unwrap();
        if self.is_exact() && rhs.is_exact() {
            Self::from(number)
        } else {
            let uncertainty = ((Decimal::try_from(self.uncertainty()).unwrap().powu(2))
                + (Decimal::try_from(rhs.uncertainty()).unwrap().powu(2)))
                .sqrt()
                .unwrap();
            Self::from(number).with_uncertainty(uncertainty.into())
        }
    }
}

impl Sub for &SciNum {
    type Output = SciNum;

    fn sub(self, rhs: Self) -> SciNum {
        *self - *rhs
    }
}

impl Mul for SciNum {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let negative = self.negative ^ rhs.negative;
        let (significand, exponent) = match self.significand.checked_mul(rhs.significand) {
            Some(s) => (s, self.exponent + rhs.exponent),
            None => {
                // Significand multiplication results in overflow, so convert to u128,
                // do mul (which won't ever overflow), then round
                let mut too_wide = (self.significand as u128) * (rhs.significand as u128);
                let mut e = self.exponent + rhs.exponent;
                let s: u64 = loop {
                    match u64::try_from(too_wide) {
                        Err(_) => {
                            // Still too wide so divide by 10
                            // In future we should round; for now, just truncate
                            too_wide /= 10;
                            e += 1;
                            continue;
                        }
                        // We have reduced the precision of the significand enough that it
                        // into a u64 again
                        Ok(narrow_enough) => break narrow_enough,
                    }
                };
                (s, e)
            },
        };
        let number = Self { negative, exponent, uncertainty_scale: 0, uncertainty: 0, significand };
        if self.is_exact() && rhs.is_exact() {
            number
        } else {
            let uncertainty = (self.relative_uncertainty().pow(2.into())
                + rhs.relative_uncertainty().pow(2.into()))
                .sqrt()
                * number.abs();
            number.with_uncertainty(uncertainty)
        }
    }
}

impl Mul for &SciNum {
    type Output = SciNum;

    fn mul(self, rhs: Self) -> SciNum {
        *self * *rhs
    }
}

impl Div for SciNum {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        let number = Decimal::try_from(self.number()).unwrap() / Decimal::try_from(rhs.number()).unwrap();
        if self.is_exact() && rhs.is_exact() {
           Self::from(number)
        } else {
            let uncertainty = ((Decimal::try_from(self.relative_uncertainty()).unwrap().powu(2))
                + (Decimal::try_from(rhs.relative_uncertainty()).unwrap().powu(2)))
                .sqrt()
                .unwrap()
                * number.abs();
            Self::from(number).with_uncertainty(uncertainty.into())
        }
    }
}

impl Div for &SciNum {
    type Output = SciNum;

    fn div(self, rhs: Self) -> SciNum {
        *self / *rhs
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
                self + SciNum::from(rhs)
            }
        }

        impl Add<SciNum> for $t {
            type Output = SciNum;

            fn add(self, rhs: SciNum) -> SciNum {
                SciNum::from(self) + rhs
            }
        }

        impl Sub<$t> for SciNum {
            type Output = Self;

            fn sub(self, rhs: $t) -> SciNum {
                self - SciNum::from(rhs)
            }
        }

        impl Sub<SciNum> for $t {
            type Output = SciNum;

            fn sub(self, rhs: SciNum) -> SciNum {
                SciNum::from(self) - rhs
            }
        }

        impl Mul<$t> for SciNum {
            type Output = Self;

            fn mul(self, rhs: $t) -> SciNum {
                self * SciNum::from(rhs)
            }
        }

        impl Mul<SciNum> for $t {
            type Output = SciNum;

            fn mul(self, rhs: SciNum) -> SciNum {
                SciNum::from(self) * rhs
            }
        }

        impl Div<$t> for SciNum {
            type Output = Self;

            fn div(self, rhs: $t) -> SciNum {
                self / SciNum::from(rhs)
            }
        }

        impl Div<SciNum> for $t {
            type Output = SciNum;

            fn div(self, rhs: SciNum) -> SciNum {
                SciNum::from(self) / rhs
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

//impl Debug for SciNum {
//    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//        f.debug_struct("SciNum")
//            .field("number", &self.number())
//            .field("uncertainty", &self.uncertainty())
//            .finish()
//    }
//}

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
                let zeros = "0".repeat((self.precision().unsigned_abs() - self.sigfigs() as u16).into());
                write!(f, "0.{zeros}{significand}{uncertainty}")
            }
        // Otherwise, use scientific notation
        } else {
            dbg!(&self);
            let (int, zeros, frac, _, exp) = self.scientific_parts();
            dbg!(exp);
            let zeros = "0".repeat(zeros.into());
            // Fractional part might not have any places at all (e.g. 2e6)
            if frac == 0 {
                write!(f, "{int}{uncertainty}e{exp}")
            } else {
                write!(f, "{int}.{zeros}{frac}{uncertainty}e{exp}")
            }
        }
    }
}

impl FromStr for SciNum {
    type Err = SciNumError;

    /// Parses a string and attempts to create a corresponding `SciNum`.
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
            uncertainty,
            uncertainty_scale: 0,
            negative,
            exponent: (exponent + EXPONENT_BIAS) as u16 - frac_places as u16,
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
        exponent: UEXPONENT_BIAS,
        uncertainty_scale: 0,
        uncertainty: 0,
        significand: 0,
    };

    /// A constant representing 1.
    pub const ONE: SciNum = SciNum {
        negative: false,
        exponent: UEXPONENT_BIAS,
        uncertainty_scale: 0,
        uncertainty: 0,
        significand: 1,
    };

    /// The highest supported number.
    pub const MAX: SciNum = SciNum {
        negative: false,
        exponent: UMAX_EXPONENT,
        uncertainty_scale: 0,
        uncertainty: 0,
        significand: u64::MAX,
    };

    /// The lowest supported number.
    pub const MIN: SciNum = SciNum {
        negative: true,
        exponent: UMAX_EXPONENT,
        uncertainty_scale: 0,
        uncertainty: 0,
        significand: u64::MAX,
    };
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
        let n1 = SciNum::from_scientific_parts(67, 0, 2, 0, 0); // 67.2
        assert_eq!(n1.to_string(), "67.2");
        assert_eq!(n1, SciNum::new(670, -1));

        let n2 = SciNum::from_scientific_parts(67, 1, 0, 0, 0); // 67.0
        assert_eq!(n2.to_string(), "67.0");
        assert_eq!(n2, SciNum::new(670, -1));

        let n3 = SciNum::from_scientific_parts(2, 0, 36, 0, 5);
        assert_eq!(n3.to_string(), "2.36e5");
        assert_eq!(n3, SciNum::from(dec!(2.36e5)));

        let n4 = SciNum::from_scientific_parts(23, 0, 61, 0, -7);
        assert_eq!(n4.to_string(), "2.361e-6");
        assert_eq!(n4, SciNum::from(dec!(2.361e-6)));
    }

    #[test]
    fn new_large() {
        let _n = SciNum::new(236, 40);
    }

    #[test]
    fn new_small() {
        let _n = SciNum::new(49, -76);
    }

    #[test]
    fn into_decimal() {
        let n = SciNum::new_with_uncertainty(20, 2, 0);
        assert_eq!(Decimal::try_from(n).unwrap(), dec!(20));
    }

    #[test]
    #[should_panic]
    fn into_decimal_fails() {
        let n = SciNum::new_with_uncertainty(20, 2, 40);
        let _d = Decimal::try_from(n).unwrap();
    }

    #[test]
    fn uncertainty() {
        let n = SciNum::new_with_uncertainty(30, 5, 0);
        assert_eq!(n.uncertainty(), SciNum::new(5, 0));
    }

    #[test]
    fn relative_uncertainty() {
        let n = SciNum::new_with_uncertainty(20, 2, 0);
        assert_eq!(n.relative_uncertainty(), SciNum::new(1, -1));

        let n2 = SciNum::new_with_uncertainty(500, 5, 0);
        assert_eq!(n2.relative_uncertainty(), SciNum::new(1, -2));

        let n3 = SciNum::new_with_uncertainty(1000, 15, 0);
        assert_eq!(n3.relative_uncertainty(), SciNum::new(15, -3));
    }

    #[test]
    fn sigfigs() {
        let n = SciNum::from_scientific_parts(123, 0, 45, 0, 0);
        assert_eq!(n.sigfigs(), 5);

        let n2 = SciNum::from_scientific_parts(123, 1, 45, 0, 0);
        assert_eq!(n2.sigfigs(), 6);

        let n3 = SciNum::from(dec!(0.00123));
        assert_eq!(n3.sigfigs(), 3);

        let n4 = SciNum::new(1234, 0);
        assert_eq!(n4.sigfigs(), 4);
    }

    #[test]
    fn sigfigs_trailing_zeros() {
        let n = SciNum::from_scientific_parts(123, 0, 4500, 0, 0);
        assert_eq!(n.sigfigs(), 7);

        let n2 = SciNum::from(dec!(0.001230));
        assert_eq!(n2.sigfigs(), 4);

        let n3 = SciNum::new(1230, 0);
        assert_eq!(n3.sigfigs(), 4);
    }

    #[test]
    fn precision() {
        assert_eq!(sci!(0.02).precision(), -2);
        assert_eq!(SciNum::from(dec!(0.020)).precision(), -3);
        assert_eq!(SciNum::from(Decimal::TWO).precision(), 0);
        assert_eq!(SciNum::new(2, 3).precision(), 3);
        assert_eq!(SciNum::from_str("2e3").unwrap().precision(), 3);
    }

    #[test]
    fn is_exact() {
        let n1 = sci!(45.1);
        let n2 = SciNum::new_with_uncertainty(500, 5, 0);
        assert!(n1.is_exact());
        assert!(!n2.is_exact());
    }

    #[test]
    fn eq() {
        // Basic case
        assert_eq!(SciNum::new(3, 0), SciNum::new(3, 0));
        // Not equal, basic case
        assert_ne!(SciNum::new(3, 0), SciNum::new(4, 0));
        // Both zero
        assert_eq!(SciNum::new(0, 0), SciNum::new(0, 0));
        // Both zero, one is negative zero
        assert_eq!(SciNum::new(0, 0), SciNum::new(-0, 0));
        // Opposite sign but same significand
        assert_ne!(SciNum::new(3, 0), SciNum::new(-3, 0));
        // Same value but different precision
        assert_eq!(SciNum::new(200, 3), SciNum::new(2, 5));
        // Same value but different precision, small numbers
        assert_eq!(SciNum::new(200, 3), SciNum::new(2, 5));
    }

    #[test]
    fn truncate_sf() {
        // Positive
        let n = sci!(25.6949);
        assert_eq!(n.truncate_sf(2), sci!(25));
        assert_eq!(n.truncate_sf(3), sci!(25.6));
        // Negative
        let n = sci!(-3.794718);
        assert_eq!(n.truncate_sf(4), sci!(-3.794));
        assert_eq!(n.truncate_sf(3), sci!(-3.79));
        // Integer
        let n = sci!(4327890);
        assert_eq!(n.truncate_sf(4), sci!(4.327e6));
        assert_eq!(n.truncate_sf(5), sci!(4.3278e6));
        // Smaller than 1
        let n = sci!(0.4327890);
        assert_eq!(n.truncate_sf(4), sci!(4.327e-1));
        assert_eq!(n.truncate_sf(5), sci!(4.3278e-1));
    }

    #[test]
    fn addition_fn_exact() {
        let n1 = SciNum::new(40, 0);
        let n2 = sci!(5.1);
        let result = n1.add_with_correlation(n2, 0);
        assert_eq!(result, sci!(45.1));
    }

    #[test]
    fn addition_fn() {
        let n1 = SciNum::new_with_uncertainty(20, 2, 0);
        let n2 = SciNum::new_with_uncertainty(30, 5, 0);
        let result = n1.add_with_correlation(n2, 0);
        assert_eq!(result.number(), sci!(50));
        assert_eq!(
            Decimal::try_from(result.uncertainty()).unwrap().round_dp(5),
            dec!(5.3851648071345).round_dp(5)
        );
    }

    #[test]
    fn addition_op() {
        let n1 = SciNum::new_with_uncertainty(20, 2, 0);
        let n2 = SciNum::new_with_uncertainty(30, 5, 0);
        let result = n1 + n2;
        assert_eq!(result.number(), sci!(50));
        assert_eq!(
            Decimal::try_from(result.uncertainty()).unwrap().round_dp(5),
            dec!(5.3851648071345).round_dp(5)
        );
    }

    #[test]
    fn addition_with_int() {
        let n1 = SciNum::new(20, 0);
        let n2 = 30;
        let result: SciNum = n1 + n2;
        assert_eq!(result.number(), sci!(50));
    }

    #[test]
    fn subtraction() {
        let n1 = SciNum::new_with_uncertainty(20, 2, 0);
        let n2 = SciNum::new_with_uncertainty(30, 5, 0);
        let result = n1 - n2;
        assert_eq!(result, sci!(-10));
        assert_eq!(
            Decimal::try_from(result.uncertainty()).unwrap().round_dp(5),
            dec!(5.3851648071345).round_dp(5)
        );
    }

    #[test]
    fn subtraction_with_int() {
        let n1 = SciNum::new(20, 0);
        let n2 = 30;
        let result: SciNum = n1 - n2;
        assert_eq!(result, sci!(-10));
    }

    #[test]
    fn multiplication() {
        let n1 = SciNum::new_with_uncertainty(20, 2, 0);
        let n2 = SciNum::new_with_uncertainty(30, 5, 0);
        let result = n1 * n2;
        assert_eq!(result, sci!(600));
        assert_eq!(
            Decimal::try_from(result.uncertainty()).unwrap().round_dp(5),
            dec!(116.619037896906).round_dp(5)
        );
        let ft = SciNum::from(dec!(0.3048));
        let square_ft = ft * ft;
        assert_eq!(square_ft, sci!(0.09290304));
    }

    #[test]
    fn multiplication_with_int() {
        let n1 = SciNum::new(20, 0);
        let n2 = 30;
        let result: SciNum = n1 * n2;
        assert_eq!(result, sci!(600));
    }

    #[test]
    fn division() {
        let n1 = SciNum::new_with_uncertainty(20, 2, 0);
        let n2 = SciNum::new_with_uncertainty(30, 5, 0);
        let result = n1 / n2;
        assert_eq!(
            Decimal::try_from(result.uncertainty()).unwrap().round_dp(10),
            dec!(0.6666666667).round_dp(10)
        );
        assert_eq!(
            Decimal::try_from(result.uncertainty()).unwrap().round_dp(5),
            dec!(0.129576708774340).round_dp(5)
        );
    }

    #[test]
    fn division_with_int() {
        let n1 = SciNum::new(60, 0);
        let n2 = 30;
        let result: SciNum = n1 / n2;
        assert_eq!(result, SciNum::from(Decimal::TWO));
    }

    #[test]
    fn division_reversed() {
        let n1 = SciNum::new_with_uncertainty(20, 2, 0);
        let n2 = SciNum::new_with_uncertainty(30, 5, 0);
        let result = n2 / n1;
        assert_eq!(result, sci!(1.5));
        assert_eq!(
            Decimal::try_from(result.uncertainty()).unwrap().round_dp(5),
            dec!(0.2915475947422).round_dp(5)
        );
    }

    #[test]
    fn exponentiation() {
        let n1 = SciNum::new_with_uncertainty(20, 2, 0);

        let result = n1.powd(Decimal::TWO);
        assert_eq!(result.number(), sci!(400));
        assert_eq!(result.uncertainty(), sci!(80));

        let result = n1.powi(2);
        assert_eq!(result.number(), sci!(400));
        assert_eq!(result.uncertainty(), sci!(80));
    }

    #[test]
    fn natural_log() {
        let n1 = SciNum::new_with_uncertainty(20, 2, 0);
        let n2 = SciNum::new_with_uncertainty(30, 5, 0);
        let ratio = n1 / n2;
        let result = ratio.ln();
        assert_eq!(
            Decimal::try_from(result.uncertainty()).unwrap().round_dp(5),
            dec!(0.194365063161).round_dp(5)
        );
    }

    #[test]
    fn log_base10() {
        let n1 = SciNum::new_with_uncertainty(20, 2, 0);
        let n2 = SciNum::new_with_uncertainty(30, 5, 0);
        let ratio = n1 / n2;
        let result = ratio.log10();
        assert_eq!(
            Decimal::try_from(result.uncertainty()).unwrap().round_dp(5),
            dec!(0.08441167440582).round_dp(5)
        );
    }

    #[test]
    fn exponential() {
        let n1 = SciNum::new_with_uncertainty(20, 2, 0);
        let n2 = SciNum::new_with_uncertainty(30, 5, 0);
        let ratio = n1 / n2;
        let result = ratio.exp();
        assert_eq!(
            Decimal::try_from(result.uncertainty()).unwrap().round_dp(5),
            dec!(0.25238096660761).round_dp(5)
        );
    }

    //#[test]
    //fn debug() {
    //    let n = SciNum::new_with_uncertainty(20, 2, 0);
    //    assert_eq!(format!("{n:?}"), "SciNum { number: 20, uncertainty: 2 }");
    //}

    #[test]
    fn display() {
        // Small integers display normally
        assert_eq!(SciNum::new(20, 0).to_string(), "20");
        // Up to 5 places displays normally
        assert_eq!(SciNum::new(99999, 0).to_string(), "99999");
        assert_eq!(SciNum::from(dec!(0.00001)).to_string(), "0.00001");
        // Above 6 places uses scientific notation
        assert_eq!(SciNum::new(1295891, 0).to_string(), "1.295891e6");
        assert_eq!(SciNum::from(dec!(0.000000432)).to_string(), "4.32e-7");
        // Explicit zeros should be treated as significant
        assert_eq!(SciNum::new(1295800, 0).to_string(), "1.295800e6");
        // Here they shouldn't be
        assert_eq!(sci!(1.2958e6).to_string(), "1.2958e6");
        // Check uncertainty formatting
        assert_eq!(SciNum::new_with_uncertainty(20, 2, 0).to_string(), "20(2)");
        // TODO: More uncertainty display tests
    }

    #[test]
    fn from_str() {
        // Integer
        assert_eq!(SciNum::from_str("42").unwrap(), SciNum::from(dec!(42)));
        // Negative float
        assert_eq!(
            SciNum::from_str("-3.14").unwrap(),
            SciNum::from(dec!(-3.14))
        );
        // Scientific notation
        assert_eq!(
            SciNum::from_str("1.5e8").unwrap(),
            SciNum::from(dec!(1.5e8))
        );
        // TODO large exponent fails with overflow error
        //assert_eq!(SciNum::from_str("1.5e10").unwrap(),
        // SciNum::from(dec!(1.5e10))); Scientific notation with negative
        // exponent
        assert_eq!(
            SciNum::from_str("2e-5").unwrap(),
            SciNum::from(dec!(2e-5))
        );
        // Negative number with positive exponent
        assert_eq!(
            SciNum::from_str("-6.022e6").unwrap(),
            SciNum::from(dec!(-6.022e6))
        );
        // TODO large exponent fails with overflow error
        assert_eq!(
            SciNum::from_str("-6.022e23").unwrap(),
            SciNum::from(dec!(-6.022e23)));
        // Capital E for exponent
        assert_eq!(
            SciNum::from_str("1.5E8").unwrap(),
            SciNum::from(dec!(1.5E8))
        );
        // Make sure incorrectly formatted string fails
        assert!(SciNum::from_str("not a number").is_err());
    }

    #[test]
    fn sci_macro() {
        // Integer
        assert_eq!(sci!(42), SciNum::new(42, 0));
        // Negative float
        assert_eq!(sci!(-3.14), SciNum::from_scientific_parts(-3, 0, 14, 0, 0));
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
