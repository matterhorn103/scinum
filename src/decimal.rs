use std::{
    cmp::Ordering,
    fmt::{self, Debug},
    num::{FpCategory, ParseFloatError},
    ops::{Add, Div, Mul, Neg, Rem, Sub},
    str::FromStr,
};

use bigdecimal::{BigDecimal, num_bigint::BigInt};
use num_traits::{Float, FromPrimitive, Inv, Num, One, Pow, Zero};
use regex::Regex;
use rust_decimal::{Decimal, MathematicalOps};

use crate::{
    RoundingMode, SciFloat, SciNum, error::SciNumError, rounding::cmp_tie,
    uncertainties::uncertainty_fn_generator,
};

/// A decimal floating point number with an associated uncertainty.
///
/// Represents a number of the form (_m_ ± _u_) × 10<sup><i>n</i></sup>.
///
/// The design is intended to allow excellent compatibility with other numeric
/// types and provide the precision of 64-bit formats while also propagating
/// uncertainties across arithmetic operations.
/// `SciDecimal` uses a 64-bit significand in binary integer format (providing
/// 16 decimal digits of precision) and a 16-bit signed exponent with the same
/// range as `i16` (but represented as a 16-bit biased unsigned integer).
/// As such, all values covered by the IEEE 754-2008 `binary64` (i.e. `f64`)
/// and `decimal64` formats are representable.
///
/// Rounding, formatting, and parsing methods are provided with a view to
/// enabling typical scientific calculations.
#[derive(Copy, Clone, Debug, serde_with::DeserializeFromStr, serde_with::SerializeDisplay)]
pub struct SciDecimal {
    uncertainty: u32,
    uncertainty_scale: i8, // This allows the uncertainty to have a different precision
    nan: bool,
    inf: bool,
    negative: bool,
    exponent: i16,
    significand: u64,
}

/// Constants that don't belong to specific trait implementations.
impl SciDecimal {
    /// The maximum supported (unsigned) significand.
    ///
    /// `SciDecimal` supports up to 16 decimal digits, matching the precision of the
    /// IEEE 754 `decimal64` interchange format.
    ///
    /// This is slightly larger than the range of `f64` significands.
    pub const MAX_SIGNIFICAND: u64 = 10_u64.pow(16) - 1;

    /// The lowest supported signed significand.
    pub const MIN_SIGNIFICAND_SIGNED: i64 = -(Self::MAX_SIGNIFICAND as i64);

    /// The highest supported signed significand.
    pub const MAX_SIGNIFICAND_SIGNED: i64 = Self::MAX_SIGNIFICAND as i64;

    /// The lowest supported number.
    pub const MIN: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        nan: false,
        inf: false,
        negative: true,
        exponent: 0,
        significand: u64::MAX,
    };

    /// The smallest supported positive number.
    pub const MIN_POSITIVE: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        nan: false,
        inf: false,
        negative: false,
        exponent: i16::MIN,
        significand: 1,
    };

    /// The highest supported number.
    pub const MAX: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        nan: false,
        inf: false,
        negative: false,
        exponent: i16::MAX,
        significand: u64::MAX,
    };

    /// The `SciDecimal` representation of `NaN`, "not a number".
    pub const NAN: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        nan: true,
        inf: false,
        negative: false,
        exponent: 0,
        significand: 0,
    };

    /// The `SciDecimal` representation of positive infinity.
    pub const INFINITY: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        nan: false,
        inf: true,
        negative: false,
        exponent: 0,
        significand: 0,
    };

    /// The `SciDecimal` representation of negative infinity.
    pub const NEG_INFINITY: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        nan: false,
        inf: true,
        negative: true,
        exponent: 0,
        significand: 0,
    };

    /// The `SciDecimal` representation of negative zero.
    pub const NEG_ZERO: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        nan: false,
        inf: false,
        negative: true,
        exponent: 0,
        significand: 0,
    };
}

impl Zero for SciDecimal {
    #[inline]
    fn zero() -> Self {
        Self::ZERO
    }

    /// Returns true if the `SciDecimal` is equal to zero, regardless of any
    /// uncertainty.
    #[inline]
    fn is_zero(&self) -> bool {
        if self.nan | self.inf {
            false
        } else {
            self.significand == 0
        }
    }
}

impl One for SciDecimal {
    #[inline]
    fn one() -> Self {
        Self::ONE
    }
}

// Instantiation
impl SciDecimal {
    /// Creates an exact `SciDecimal` from parts corresponding to _m_ ×
    /// 10<sup><i>n</i></sup>.
    ///
    /// # Panics
    ///
    /// This function panics if the number has more than 16 significant figures
    /// (i.e. is larger than `MAX_SIGNIFICAND` = 2<sup>16</sup>)
    ///
    /// # Example
    ///
    /// ```
    /// # use scinum::SciDecimal;
    /// #
    /// let n = SciDecimal::new(251, -3);
    /// assert_eq!(n.to_string(), "0.251");
    /// ```
    pub const fn new(number: i64, exponent: i16) -> Self {
        if number < Self::MIN_SIGNIFICAND_SIGNED || number > Self::MAX_SIGNIFICAND_SIGNED {
            panic!("`number` has too many significant figures for a significand!")
        }
        Self {
            uncertainty: 0,
            uncertainty_scale: 0,
            nan: false,
            inf: false,
            negative: number.is_negative(),
            exponent,
            significand: number.unsigned_abs(),
        }
    }

    /// Creates a `SciDecimal` from parts corresponding to (_m_ ± _u_) ×
    /// 10<sup><i>n</i></sup>.
    ///
    /// This means the number of decimal places in the number and uncertainty
    /// will be the same in the created `SciDecimal`.
    ///
    /// # Panics
    ///
    /// This function panics if the number has more than 16 significant figures
    /// (i.e. is larger than `MAX_SIGNIFICAND` = 2<sup>16</sup>)
    ///
    /// # Example
    ///
    /// ```
    /// # use scinum::SciDecimal;
    /// #
    /// let n = SciDecimal::new_with_uncertainty(251, 3, -3);
    /// assert_eq!(n.to_string(), "0.251(3)");
    /// ```
    pub const fn new_with_uncertainty(number: i64, uncertainty: u32, exponent: i16) -> Self {
        if number < Self::MIN_SIGNIFICAND_SIGNED || number > Self::MAX_SIGNIFICAND_SIGNED {
            panic!("`number` has too many significant figures for a significand!")
        }
        Self {
            uncertainty,
            uncertainty_scale: 0,
            nan: false,
            inf: false,
            negative: number.is_negative(),
            exponent,
            significand: number.unsigned_abs(),
        }
    }

    /// Creates a `SciDecimal` from separate parts of a representation of the number in
    /// scientific notation.
    ///
    /// The arguments should correspond to `(ii, z, fff, uu, nn)` when the number is
    /// notated as `ii.{zeros}fff(uu) × 10^nn`, where `z` is the number of leading
    /// zeros in the fractional part.
    ///
    /// Trailing zeros in `fraction` are treated as significant, but leading zeros
    /// are not. If `fraction` is simply `0`, it is then also treated as
    /// insignificant. Passing `0` for both `zeros` and `fraction` therefore
    /// creates a `SciDecimal` with a significand equal to `integer`.
    ///
    /// To create a number with only significant zeros in the fractional part (such
    /// as `2.0`), pass `0` for `fraction` and specify the appropriate number of
    /// zeros as `zeros`.
    ///
    /// # Panics
    ///
    /// This function panics if the overall significand has more than 16 significant
    /// figures.
    ///
    /// # Example
    ///
    /// ```
    /// # use scinum::SciDecimal;
    /// #
    /// let n = SciDecimal::from_scientific_parts(2, 0, 51, 0, 0);
    /// assert_eq!(n.to_string(), "2.51");
    /// let n = SciDecimal::from_scientific_parts(2, 1, 51, 0, 0);
    /// assert_eq!(n.to_string(), "2.051");
    /// let n = SciDecimal::from_scientific_parts(2, 0, 51, 3, 0);
    /// assert_eq!(n.to_string(), "2.51(3)");
    /// let n = SciDecimal::from_scientific_parts(2, 0, 51, 3, -1);
    /// assert_eq!(n.to_string(), "0.251(3)");
    /// let n = SciDecimal::from_scientific_parts(2, 2, 0, 3, -2);
    /// assert_eq!(n.to_string(), "0.0200(3)");
    /// ```
    pub const fn from_scientific_parts(
        integer: i8,
        zeros: u8,
        fraction: u64,
        uncertainty: u32,
        exponent: i16,
    ) -> Self {
        let unsigned_integer = integer.unsigned_abs() as u64;
        let (significand, exponent) = {
            if fraction != 0 || zeros != 0 {
                let decimal_places = if fraction == 0 {
                    0
                } else {
                    fraction.ilog10() + 1
                };
                let significand =
                    (unsigned_integer * 10_u64.pow(decimal_places + zeros as u32)) + fraction;
                let exponent = exponent - (decimal_places as i16 + zeros as i16);
                (significand, exponent)
            } else {
                (unsigned_integer, exponent)
            }
        };
        if significand > Self::MAX_SIGNIFICAND {
            panic!("`significand` has too many significant figures for a significand!")
        }
        Self {
            uncertainty,
            uncertainty_scale: 0,
            nan: false,
            inf: false,
            negative: integer.is_negative(),
            exponent,
            significand,
        }
    }
}

/// Methods for obtaining parts of the contained data.
impl SciDecimal {
    /// Returns the integer part, number of fractional leading zeros,
    /// fractional part, uncertainty, and exponent of the number when represented
    /// with normalized notation i.e. with 10 > _m_ >= 1.
    ///
    /// Corresponds to `(ii, z, fff, uu, nn)` when the number is notated as
    /// `ii.{zeros}fff(uu) × 10^nn`, where `z` is the number of leading zeros
    /// in the fractional part.
    ///
    /// # Special values
    ///
    /// Unlike `significand()`, `sign()`, and `exponent()`, this method does not
    /// just return the stored values in all cases.
    ///
    /// - ±0 → `Some((0, 0, 0, 0, 0))`
    ///
    /// - ±∞ and `NaN` → `None`
    pub fn scientific_parts(&self) -> Option<(i8, u8, u64, u32, i16)> {
        if self.is_zero() {
            return Some((0, 0, 0, 0, 0));
        };
        if !self.is_finite() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        let figs = self.sf() as u32;
        let int_unsigned = self.significand / 10_u64.pow(figs - 1); // First digit
        let int = if self.negative {
            -(int_unsigned as i8)
        } else {
            int_unsigned as i8
        };
        let frac = self.significand % 10_u64.pow(figs - 1);
        // Work out how many zeros have been dropped, if any
        let figs_in_frac = frac.checked_ilog10().map_or(0, |x| x + 1);
        let zeros = (figs - 1 - figs_in_frac) as u8; // 1 is for integer digit
        let uncert = self.uncertainty;
        let exp = self.exponent + (figs as i16 - 1);
        // For example:
        // 1.23e2 = 123 is stored as (123, 0)       =>  2 =  0 + (3 - 1)
        // 4.5e6 = 4_500_000 is stored as (45, 5)   =>  6 =  5 + (2 - 1)
        // 4.5e-3 = 0.0045 is stored as (45, -4)    => -3 = -4 + (2 - 1)
        // 4.51e-3 = 0.00451 is stored as (451, -5) => -3 = -5 + (3 - 1)
        // 4.50e-3 = 0.00450 is stored as (450, -5) => -3 = -5 + (3 - 1)
        Some((int, zeros, frac, uncert, exp))
    }

    /// Returns the signed significand _m_ of the number when represented with
    /// _m_ as an integer.
    ///
    /// Corresponds to `(-1)^s × mmmmm` in the actual in-memory representation
    /// of the number as `(-1)^s × mmmmm × 10^nn`
    ///
    /// Note that the current stored value of the significand is returned even
    /// when the number is not normal (and the value of the significand therefore
    /// moot).
    #[inline]
    pub fn significand_signed(&self) -> i64 {
        if self.negative {
            -(self.significand as i64)
        } else {
            self.significand as i64
        }
    }

    /// Returns the unsigned significand _m_ of the number when represented with
    /// _m_ as an integer.
    ///
    /// Corresponds to `mmmmm` in the actual in-memory representation of the
    /// number as `(-1)^s × mmmmm × 10^nn`
    ///
    /// Note that the current stored value of the significand is returned even
    /// when the number is not normal (and the value of the significand therefore
    /// moot).
    #[inline]
    pub fn significand(&self) -> u64 {
        self.significand
    }

    /// Returns the sign bit; `true` means the `SciDecimal` is negative.
    ///
    /// Corresponds to `s` in the actual in-memory representation of the number
    /// as `(-1)^s × mmmmm × 10^nn`
    ///
    /// Note that the current stored value of the sign bit is returned even when
    /// the number is not normal (and the value of the sign therefore moot).
    #[inline]
    pub fn sign(&self) -> bool {
        self.negative
    }

    /// Returns the exponent _n_ of the number when represented with _m_ as an
    /// integer.
    ///
    /// Corresponds to `nn` in the actual in-memory representation of the number
    /// as `(-1)^s × mmmmm × 10^nn`
    ///
    /// Note that the current stored value of the exponent is returned even when
    /// the number is not normal (and the value of the exponent therefore moot).
    #[inline]
    pub fn exponent(&self) -> i16 {
        self.exponent
    }
}

/// Methods for precision, figures, and rounding that aren't part of the `SciNum`
/// implementation.
impl SciDecimal {
    /// Increases the precision of the number by adding `sf` additional
    /// significant zeros to the significand.
    ///
    /// This is equivalent to decreasing the exponent by `sf`.
    ///
    /// The uncertainty of the `SciDecimal` is left unchanged.
    ///
    /// # Panics
    ///
    /// This function panics if the increase would result in `significand`,
    /// `exponent`, or `uncertainty_scale` exceeding their maximum values.
    /// The arithmetic operations used are the strict ones i.e. they panic on
    /// overflow, regardless of whether overflow checks are enabled.
    pub fn increase_precision(mut self, sf: u8) -> Self {
        if !self.is_finite() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        for _ in 0..sf {
            self.significand = self.significand.strict_mul(10);
            // Check that it's not larger than allowed (16 sf)
            if self.significand > Self::MAX_SIGNIFICAND {
                panic!("Maximum precision (16 sf) exceeded!")
            }
            // Exponent is now too large
            self.exponent = self.exponent.strict_sub(1);
            // Uncertainty is now too small
            if !self.is_exact() {
                self.uncertainty_scale = self.uncertainty_scale.strict_add(1);
            };
        }
        self
    }

    /// Increases the precision of the number by adding `sf` additional
    /// significant zeros to the significand, without panicking.
    ///
    /// This is equivalent to decreasing the exponent by `sf`.
    ///
    /// The uncertainty of the `SciDecimal` is left unchanged.
    pub fn increase_precision_checked(mut self, sf: u8) -> Option<Self> {
        if !self.is_normal() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        for _ in 0..sf {
            self.significand = self.significand.checked_mul(10)?;
            // Also check that it's not larger than allowed (16 sf)
            if self.significand > Self::MAX_SIGNIFICAND {
                return None;
            }
            // Exponent is now too large
            self.exponent = self.exponent.checked_sub(1)?;
            // Uncertainty is now too small
            if !self.is_exact() {
                match self.uncertainty_scale.checked_add(1) {
                    Some(s) => self.uncertainty_scale = s,
                    None => {
                        // In the rare case that we can't increase the uncertainty
                        // scale we increase the uncertainty significand instead
                        self.uncertainty = self.uncertainty.checked_mul(10)?;
                    }
                }
            };
        }
        Some(self)
    }

    /// Increases the precision of the number by adding `sf` additional
    /// significant zeros to the significand, permitting values for the
    /// significand greater than `SciDecimal::MAX_SIGNIFICAND` and up to `u64::MAX`.
    ///
    /// This is equivalent to decreasing the exponent by `sf`.
    ///
    /// The uncertainty of the `SciDecimal` is left unchanged.
    ///
    /// # Panics
    ///
    /// This function panics if the increase would result in `significand`
    /// exceeding `u64::MAX` or in `exponent` or `uncertainty_scale` exceeding
    /// their maximum values.
    /// The arithmetic operations used are the strict ones i.e. they panic on
    /// overflow, regardless of whether overflow checks are enabled.
    #[allow(unused)]
    fn increase_precision_unbounded(mut self, sf: u8) -> Self {
        if !self.is_normal() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        for _ in 0..sf {
            self.significand = self.significand.strict_mul(10);
            // But explicitly *don't* check if it exceeds 16 sf!
            // Exponent is now too large
            self.exponent = self.exponent.strict_sub(1);
            // Uncertainty is now too small
            if !self.is_exact() {
                self.uncertainty_scale = self.uncertainty_scale.strict_add(1);
            };
        }
        self
    }

    /// Increases the precision of the number by adding `sf` additional
    /// significant zeros to the significand, without panicking, permitting
    /// values for the significand greater than `SciDecimal::MAX_SIGNIFICAND`
    /// and up to `u64::MAX`.
    ///
    /// This is equivalent to decreasing the exponent by `sf`.
    ///
    /// The uncertainty of the `SciDecimal` is left unchanged.
    fn increase_precision_unbounded_checked(mut self, sf: u8) -> Option<Self> {
        if !self.is_normal() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        for _ in 0..sf {
            self.significand = self.significand.checked_mul(10)?;
            // But explicitly *don't* check if it exceeds 16 sf!
            // Exponent is now too large
            self.exponent = self.exponent.checked_sub(1)?;
            // Uncertainty is now too small
            if !self.is_exact() {
                match self.uncertainty_scale.checked_add(1) {
                    Some(s) => self.uncertainty_scale = s,
                    None => {
                        // In the rare case that we can't increase the uncertainty
                        // scale we increase the uncertainty significand instead
                        self.uncertainty = self.uncertainty.checked_mul(10)?;
                    }
                }
            };
        }
        Some(self)
    }
}

/// Additional constants.
impl SciDecimal {
    // TODO Add more of the constants that f64 has https://doc.rust-lang.org/std/f64/consts/index.html

    pub const TWO: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        nan: false,
        inf: false,
        negative: false,
        exponent: 0,
        significand: 2,
    };

    pub const NEG_ONE: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        nan: false,
        inf: false,
        negative: true,
        exponent: 0,
        significand: 1,
    };

    /// The mathematical constant *π*.
    pub const PI: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        nan: false,
        inf: false,
        negative: true,
        exponent: -15,
        significand: 3_141_592_653_589_793,
    };

    /// The mathematical constant *π*, with 19 sf for internal use.
    const PI_PRECISE: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        nan: false,
        inf: false,
        negative: true,
        exponent: -18,
        significand: 3_141_592_653_589_793_238,
    };

    /// The mathematical constant *e*.
    pub const E: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        nan: false,
        inf: false,
        negative: true,
        exponent: -15,
        significand: 2_718_281_828_459_045,
    };

    /// The mathematical constant *e*, with 19 sf for internal use.
    const E_PRECISE: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        nan: false,
        inf: false,
        negative: true,
        exponent: -15,
        significand: 2_718_281_828_459_045_235,
    };

    /// The natural logarithm of 2, ln(2) = logₑ(2).
    pub const LN_2: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        nan: false,
        inf: false,
        negative: true,
        exponent: -16,
        significand: 693_147_180_559_945_3,
    };

    /// The natural logarithm of 2, ln(2) = logₑ(2), with 19 sf for internal use.
    const LN_2_PRECISE: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        nan: false,
        inf: false,
        negative: true,
        exponent: -19,
        significand: 693_147_180_559_945_309_4,
    };

    /// The natural logarithm of 10, ln(10) = logₑ(10).
    pub const LN_10: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        nan: false,
        inf: false,
        negative: true,
        exponent: -15,
        significand: 2_302_585_092_994_046,
    };

    /// The natural logarithm of 10, ln(10) = logₑ(10), with 19 sf for internal use.
    const LN_10_PRECISE: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        nan: false,
        inf: false,
        negative: true,
        exponent: -18,
        significand: 2_302_585_092_994_045_684,
    };

    /// The base-2 logarithm of *e*, log₂(*e*).
    pub const LOG2_E: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        nan: false,
        inf: false,
        negative: true,
        exponent: -15,
        significand: 1_442_695_040_888_963,
    };

    /// The base-2 logarithm of *e*, log₂(*e*), with 19 sf for internal use.
    const LOG2_E_PRECISE: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        nan: false,
        inf: false,
        negative: true,
        exponent: -18,
        significand: 1_442_695_040_888_963_407,
    };

    /// The base-2 logarithm of 10, log₂(10).
    pub const LOG2_10: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        nan: false,
        inf: false,
        negative: true,
        exponent: -15,
        significand: 3_321_928_094_887_362,
    };

    /// The base-2 logarithm of 10, log₂(10), with 19 sf for internal use.
    const LOG2_10_PRECISE: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        nan: false,
        inf: false,
        negative: true,
        exponent: -18,
        significand: 3_321_928_094_887_362_348,
    };

    /// The base-10 logarithm of 2, log₁₀(2).
    pub const LOG10_2: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        nan: false,
        inf: false,
        negative: true,
        exponent: -16,
        significand: 301_029_995_663_981_2,
    };

    /// The base-10 logarithm of 2, log₁₀(2), with 19 sf for internal use.
    const LOG10_2_PRECISE: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        nan: false,
        inf: false,
        negative: true,
        exponent: -19,
        significand: 301_029_995_663_981_195_2,
    };

    /// The base-10 logarithm of *e*, log₁₀(*e*).
    pub const LOG10_E: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        nan: false,
        inf: false,
        negative: true,
        exponent: -16,
        significand: 434_294_481_903_251_8,
    };

    /// The base-10 logarithm of *e*, log₁₀(*e*), with 19 sf for internal use.
    const LOG10_E_PRECISE: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        nan: false,
        inf: false,
        negative: true,
        exponent: -19,
        significand: 434_294_481_903_251_827_7,
    };
}

impl SciNum for SciDecimal {
    type Number = SciDecimal;

    const ZERO: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        nan: false,
        inf: false,
        negative: false,
        exponent: 0,
        significand: 0,
    };

    const ONE: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        nan: false,
        inf: false,
        negative: false,
        exponent: 0,
        significand: 1,
    };

    #[inline]
    fn number(&self) -> Self {
        Self {
            uncertainty: 0,
            uncertainty_scale: 0,
            ..*self
        }
    }

    #[inline]
    fn uncertainty(&self) -> Self {
        if self.nan {
            Self::NAN
        } else if self.inf {
            Self::INFINITY
        } else {
            Self {
                uncertainty: 0,
                uncertainty_scale: 0,
                nan: false,
                inf: false,
                negative: false,
                exponent: self.exponent + self.uncertainty_scale as i16,
                significand: self.uncertainty.into(),
            }
        }
    }

    /// Returns the relative uncertainty as an exact `SciDecimal`.
    ///
    /// The relative uncertainty is always positive.
    #[inline]
    fn relative_uncertainty(&self) -> Self {
        self.uncertainty() / self.number().abs()
    }

    /// Creates a new `SciDecimal` with the same number but the provided
    /// uncertainty.
    ///
    /// If the uncertainty has a significand greater than `u32::MAX` (i.e. more
    /// than ~9 significant figures), it is first truncated to 9 s.f.
    ///
    /// # Example
    ///
    /// ```
    /// # use scinum::SciDecimal;
    /// #
    /// let n = SciDecimal::new(251, -3).with_uncertainty(SciDecimal::new(3, -3));
    /// assert_eq!(n.to_string(), "0.251(3)");
    /// assert_eq!(n, SciDecimal::new_with_uncertainty(251, 3, -3));
    #[inline]
    fn with_uncertainty(mut self, uncertainty: Self) -> Self {
        if !uncertainty.is_finite() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        let narrowed_uncertainty = if uncertainty.significand > u32::MAX.into() {
            uncertainty.trunc_sf(9);
            uncertainty
        } else {
            uncertainty
        };
        self.uncertainty_scale = (narrowed_uncertainty.exponent - self.exponent)
            .try_into()
            .expect(
                "Difference in precision of number and uncertainty should never be this large!",
            );
        self.uncertainty = narrowed_uncertainty
            .significand
            .try_into()
            .expect("Already made sure that this is not greater than `u32::MAX`");
        self
    }

    /// Returns true if the `SciDecimal` has an uncertainty of zero.
    #[inline]
    fn is_exact(&self) -> bool {
        if !self.is_finite() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        self.uncertainty == 0
    }

    /// Returns the scale of the least significant place.
    ///
    /// For example:
    /// - 0.02 returns -2
    /// - 0.020 returns -3
    /// - 2 returns 0
    /// - 200 returns 2 or 1 or 0, depending on the precision of the number
    #[inline]
    fn precision(&self) -> i16 {
        if !self.is_normal() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        self.exponent
    }

    /// Returns the scale of the most significant place.
    ///
    /// This is equivalent to the exponent _n_ of the number when represented with
    /// normalized notation i.e. with 10 > _m_ >= 1.
    ///
    /// For example:
    /// - 0.02 returns -2
    /// - 0.025 returns -2
    /// - 0.020 returns -2
    /// - 2 returns 0
    /// - 321 returns 2
    #[inline]
    fn precision_most_significant_fig(&self) -> i16 {
        if !self.is_normal() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        self.exponent + (i16::from(self.sf()) - 1)
    }

    /// Returns the scale of the least significant place of the uncertainty.
    fn precision_uncertainty(&self) -> Option<i16> {
        if !self.is_normal() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        if self.is_exact() {
            None
        } else {
            Some(self.exponent + self.uncertainty_scale as i16)
        }
    }

    #[inline]
    fn dp(&self) -> u16 {
        if !self.is_normal() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        if self.precision() >= 0 {
            0
        } else {
            self.precision().unsigned_abs()
        }
    }

    #[inline]
    fn sf(&self) -> u8 {
        if !self.is_normal() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        if let Some(log) = self.significand.checked_ilog10() {
            log as u8 + 1
        } else {
            0
        }
    }

    #[inline]
    fn sf_uncertainty(&self) -> u8 {
        if !self.is_normal() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        if let Some(log) = self.uncertainty.checked_ilog10() {
            log as u8 + 1
        } else {
            0
        }
    }

    fn round_precision(self, prec: i16, mode: RoundingMode) -> Self {
        if !self.is_normal() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        if self.exponent == prec {
            return self;
        }
        let mut new = self;
        let current_prec = new.exponent;
        if prec < current_prec {
            // Simply add zeros to fulfil request
            new = new.increase_precision(
                (current_prec - prec)
                    .try_into()
                    .expect("Requested change in precision must be less than 256 places/figures!"),
            );
        } else {
            // Decrease precision while following the specified rounding mode
            let shifted = prec - current_prec; // Places to remove, will be a positive number
            let divisor = 10_u64.pow(shifted as u32);
            let mut new_sig = self.significand / divisor;
            let removed_digits = self.significand % divisor;
            if removed_digits == 0 {
                // No rounding to be done, we only removed significant zeros
            } else {
                match mode {
                    RoundingMode::HalfUp => match cmp_tie(removed_digits) {
                        Ordering::Less => {}
                        Ordering::Equal => new_sig += 1,
                        Ordering::Greater => new_sig += 1,
                    },
                    RoundingMode::HalfDown => match cmp_tie(removed_digits) {
                        Ordering::Less => {}
                        Ordering::Equal => {}
                        Ordering::Greater => new_sig += 1,
                    },
                    RoundingMode::HalfEven => match cmp_tie(removed_digits) {
                        Ordering::Less => {}
                        Ordering::Equal => {
                            if !new_sig.is_multiple_of(2) {
                                new_sig += 1;
                            }
                        }
                        Ordering::Greater => new_sig += 1,
                    },
                    RoundingMode::Up => new_sig += 1,
                    RoundingMode::Down => {}
                    RoundingMode::Ceiling => {
                        if !new.negative {
                            new_sig += 1
                        }
                    }
                    RoundingMode::Floor => {
                        if new.negative {
                            new_sig += 1
                        }
                    }
                }
            }
            new.significand = new_sig;
            new.exponent = prec;
            if !new.is_exact() {
                new.uncertainty_scale -= shifted as i8;
            }
        }
        new
    }

    #[inline]
    fn round_dp(self, dp: u16, mode: RoundingMode) -> Self {
        let desired_prec = (dp as i16).neg();
        self.round_precision(desired_prec, mode)
    }

    #[inline]
    fn round_sf(self, sf: u8, mode: RoundingMode) -> Self {
        let current_sf = self.sf(); // e.g. 3
        let prec_change = (current_sf as i16) - (sf as i16); // 3 - 1 = 2
        let desired_prec = self.precision() + prec_change;
        self.round_precision(desired_prec, mode)
    }

    #[inline]
    fn round_match_uncertainty(self, mode: RoundingMode) -> Self {
        if self.is_exact() {
            self
        } else {
            self.round_precision(self.precision_uncertainty().unwrap(), mode)
        }
    }

    #[inline]
    fn round_match_uncertainty_sf(self, sf: u8, mode: RoundingMode) -> Self {
        if self.is_exact() {
            self
        } else {
            self.with_uncertainty(self.uncertainty().round_sf(sf, mode))
                .round_match_uncertainty(mode)
        }
    }

    #[inline]
    fn round_uncertainty_precision(self, prec: i16, mode: RoundingMode) -> Self {
        if self.is_exact() {
            self
        } else {
            self.with_uncertainty(self.uncertainty().round_precision(prec, mode))
        }
    }

    #[inline]
    fn round_uncertainty_dp(self, dp: u16, mode: RoundingMode) -> Self {
        if self.is_exact() {
            self
        } else {
            self.with_uncertainty(self.uncertainty().round_dp(dp, mode))
        }
    }

    #[inline]
    fn round_uncertainty_sf(self, sf: u8, mode: RoundingMode) -> Self {
        if self.is_exact() {
            self
        } else {
            self.with_uncertainty(self.uncertainty().round_sf(sf, mode))
        }
    }

    #[inline]
    fn round_uncertainty_match_number(self, mode: RoundingMode) -> Self {
        if self.is_exact() {
            self
        } else {
            self.with_uncertainty(self.uncertainty().round_precision(self.precision(), mode))
        }
    }

    fn trunc_sf(mut self, sf: u8) -> Self {
        if !self.is_normal() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        if self.sf() < sf {
            panic!()
        };
        while self.sf() > sf {
            self.significand /= 10;
            // Exponent is now too small
            self.exponent += 1;
            // Uncertainty is now too large
            if !self.is_exact() {
                self.uncertainty_scale -= 1;
            };
        }
        self
    }
}

impl Num for SciDecimal {
    type FromStrRadixErr = SciNumError;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, SciNumError> {
        // For now, just make use of the BigDecimal implementation
        let dec = BigDecimal::from_str_radix(str, radix)
            .or(Err(SciNumError::Parse(format!("Couldn't parse {}", str))))?;
        Ok(dec.cast())
    }
}

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
        self.nan
    }

    #[inline]
    fn is_infinite(self) -> bool {
        // The NaN flag overrides the infinity flag i.e. if a `SciDecimal` has
        // both `true` then it is considered a NaN and therefore *not infinite*
        self.inf & !self.nan
    }

    #[inline]
    fn is_finite(self) -> bool {
        !(self.inf | self.nan)
    }

    #[inline]
    fn is_normal(self) -> bool {
        !(self.inf | self.nan | (self.significand == 0))
    }

    #[inline]
    fn classify(self) -> FpCategory {
        if self.nan {
            FpCategory::Nan
        } else if self.inf {
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
        if self.nan {
            Self::NAN
        } else {
            Self {
                negative: false,
                ..self
            }
        }
    }

    fn signum(self) -> Self {
        if self.nan {
            Self::NAN
        } else if self.negative {
            Self::ONE.neg()
        } else {
            Self::ONE
        }
    }

    /// Returns true if the sign bit is positive.
    /// Zero is also considered positive.
    #[inline]
    //#[must_use]
    fn is_sign_positive(self) -> bool {
        if self.is_nan() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        !self.negative
    }

    /// Returns true if the sign bit is negative.
    /// Zero is considered positive.
    #[inline]
    //#[must_use]
    fn is_sign_negative(self) -> bool {
        if self.is_nan() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        self.negative
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
        let number = Decimal::cast_from(self.number()).ln();
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
        let number = Decimal::cast_from(self.number()).log10();
        if self.is_exact() {
            number.cast()
        } else {
            let uncertainty = (Decimal::cast_from(self.uncertainty())
                / (Decimal::TEN.ln() * Decimal::cast_from(self.number())))
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

impl PartialEq for SciDecimal {
    fn eq(&self, other: &Self) -> bool {
        // NaN is never equal to anything, even itself
        if self.nan | other.nan {
            false
        // +0 == +0, but also +0 == -0
        } else if self.is_zero() && other.is_zero() {
            true
        // Can't be equal if sign is different, so short circuit if so
        } else if self.negative != other.negative {
            false
        // ∞ == ∞, -∞ == -∞, +∞ != -∞ but we already checked the signs are the same
        } else if self.inf & other.inf {
            true
        } else if self.exponent == other.exponent {
            self.significand == other.significand
        // Might be the same value but to different precision
        } else if self.significand.is_multiple_of(other.significand) {
            let factor = self.significand / other.significand;
            if factor.is_multiple_of(10) {
                let order_diff = factor.ilog10();
                self.exponent + order_diff as i16 == other.exponent
            } else {
                false
            }
        } else if other.significand.is_multiple_of(self.significand) {
            let factor = other.significand / self.significand;
            if factor.is_multiple_of(10) {
                let order_diff = factor.ilog10();
                other.exponent + order_diff as i16 == self.exponent
            } else {
                false
            }
        } else {
            false
        }
    }
}

impl PartialOrd for SciDecimal {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;

        // NaN can't be compared
        if self.nan | other.nan {
            return None;
        }
        // Zeros are equal regardless of sign
        if self.is_zero() && other.is_zero() {
            return Some(Ordering::Equal);
        }
        // Different signs are easily ordered
        if self.negative != other.negative {
            return Some(if self.negative {
                Ordering::Less
            } else {
                Ordering::Greater
            });
        }
        // Infinities
        match (self.inf, other.inf) {
            (true, true) => {
                // Must be same sign because we already compared signs
                return Some(Ordering::Equal);
            }
            (true, false) => {
                return Some(if self.negative {
                    Ordering::Less
                } else {
                    Ordering::Greater
                });
            }
            (false, true) => {
                return Some(if other.negative {
                    Ordering::Greater
                } else {
                    Ordering::Less
                });
            }
            (false, false) => {}
        }
        // Same sign, both finite, neither is zero
        let lhs_magnitude = self.precision_most_significant_fig();
        let rhs_magnitude = other.precision_most_significant_fig();
        let ordering = if lhs_magnitude != rhs_magnitude {
            lhs_magnitude.cmp(&rhs_magnitude)
        } else if self.exponent == other.exponent {
            // Easy to compare them if the exponents are the same
            self.significand.cmp(&other.significand)
        } else {
            // Otherwise have to set to same precision
            let mut lhs_sig = self.significand as u128;
            let mut rhs_sig = other.significand as u128;
            let exp_diff = self.exponent - other.exponent;
            if exp_diff.is_positive() {
                // e.g. self = 3e6, other = 3072e3
                // Have to make self into 3000e3 to be able to compare
                lhs_sig *= 10_u128.pow(exp_diff.unsigned_abs().into());
                lhs_sig.cmp(&rhs_sig)
            } else {
                // e.g. self = 3072e3, other = 3e6
                // Have to make other into 3000e3 to be able to compare
                rhs_sig *= 10_u128.pow(exp_diff.unsigned_abs().into());
                lhs_sig.cmp(&rhs_sig)
            }
        };
        // If both are negative then the order is actually the reverse
        Some(if self.negative {
            ordering.reverse()
        } else {
            ordering
        })
    }
}

/// Arithmetic operations that return exact results with potentially excess precision,
/// useful for intermediate results to avoid rounding errors, but not to be
/// returned to the end user.
impl SciDecimal {
    /// Calculates `self + rhs` without uncertainty, permitting values for the
    /// significand greater than `SciDecimal::MAX_SIGNIFICAND` and up to `u64::MAX`.
    fn unbounded_add(self, rhs: Self) -> Self {
        // TODO If significand would be too large for u64, just round it and
        // increase the exponent instead of panicking

        // Handle NaN
        if self.nan | rhs.nan {
            return Self::NAN;
        }
        // Handle infinities
        match (self.inf, rhs.inf) {
            (true, true) => {
                if self.negative == rhs.negative {
                    // ∞ + ∞ = ∞, -∞ + -∞ = -∞
                    return self;
                } else {
                    // ∞ - ∞ = NaN
                    return Self::NAN;
                }
            }
            (true, false) => {
                return self;
            }
            (false, true) => {
                return rhs;
            }
            (false, false) => {}
        }

        match self.exponent.cmp(&rhs.exponent) {
            // In the simplest case, the exponents are the same
            Ordering::Equal => {
                let number = self.significand_signed() + rhs.significand_signed();
                Self::new(number, self.exponent)
            }
            // Otherwise have to try and set the exponent to the same for both terms
            // Use whichever exponent is smallest
            Ordering::Less => {
                let exp_diff = rhs.exponent - self.exponent;
                let scaled = rhs.increase_precision(exp_diff.try_into().unwrap());
                let number = self.significand_signed() + scaled.significand_signed();
                Self::new(number, self.exponent)
            }
            Ordering::Greater => {
                let exp_diff = self.exponent - rhs.exponent;
                let scaled = self.increase_precision(exp_diff.try_into().unwrap());
                let number = scaled.significand_signed() + rhs.significand_signed();
                Self::new(number, scaled.exponent)
            }
        }
    }

    /// Calculates `self * rhs` without uncertainty, permitting values for the
    /// significand greater than `SciDecimal::MAX_SIGNIFICAND` and up to `u64::MAX`.
    fn unbounded_mul(self, rhs: Self) -> Self {
        // Handle NaN
        if self.nan | rhs.nan {
            return Self::NAN;
        }
        let negative = self.negative ^ rhs.negative;
        // Handle infinities
        match (self.inf, rhs.inf) {
            (true, true) => {
                if negative {
                    return Self::NEG_INFINITY;
                } else {
                    return Self::INFINITY;
                }
            }
            (true, false) => {
                if rhs.is_zero() {
                    return Self::NAN;
                } else if negative {
                    return Self::NEG_INFINITY;
                } else {
                    return Self::INFINITY;
                }
            }
            (false, true) => {
                if self.is_zero() {
                    return Self::NAN;
                } else if negative {
                    return Self::NEG_INFINITY;
                } else {
                    return Self::INFINITY;
                }
            }
            (false, false) => {}
        }
        let (significand, exponent) = match self.significand.checked_mul(rhs.significand) {
            Some(s) => (s, self.exponent + rhs.exponent),
            None => {
                // Even though we have ~3 spare decimal digits of precision in
                // `u64` above our max allowed significand, multiplication can
                // still result in overflow as
                // `(10_u64.pow(16) - 1) * (10_u64.pow(16) - 1) == 99999999999999980000000000000001`
                // which is too large for `u64`:
                // `99999999999999980000000000000001 / (u64::MAX as u128) == 5421010862427`
                // but `u128` has plenty of space:
                // `u128::MAX / 99999999999999980000000000000001 == 3402823`
                // So, convert to `u128`, do mul (which won't ever overflow), then
                // truncate to get back to a significand representable as a `u64`
                // We want to truncate rather than round since we'll have three
                // excess digits left anyway, meaning we'll have to round later
                // before returning to the user - if we round now as well then
                // we'll get cumulative rounding errors!
                let mut too_wide = (self.significand as u128) * (rhs.significand as u128);
                let mut e = self.exponent + rhs.exponent;
                let s: u64 = loop {
                    match u64::try_from(too_wide) {
                        Err(_) => {
                            // Still too wide so divide by 10 to truncate
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
            }
        };
        Self {
            uncertainty: 0,
            uncertainty_scale: 0,
            nan: false,
            inf: false,
            negative,
            exponent,
            significand,
        }
    }

    /// Calculates `self / rhs` without uncertainty, permitting values for the
    /// significand greater than `SciDecimal::MAX_SIGNIFICAND` and up to `u64::MAX`.
    fn unbounded_div(self, rhs: Self) -> Self {
        // Handle NaN
        if self.nan | rhs.nan {
            return Self::NAN;
        }
        let negative = self.negative ^ rhs.negative;
        // Handle infinities
        match (self.inf, rhs.inf) {
            (true, true) => {
                // ∞/∞ is undefined
                return Self::NAN;
            }
            (true, false) => {
                // ∞/n = ∞ for all n, including 0
                if negative {
                    return Self::NEG_INFINITY;
                } else {
                    return Self::INFINITY;
                }
            }
            (false, true) => {
                // n/∞ = 0 for all n, including 0
                if negative {
                    return Self::NEG_ZERO;
                } else {
                    return Self::ZERO;
                }
            }
            (false, false) => {}
        }
        // Handle zeros
        if rhs.is_zero() {
            if self.is_zero() {
                // 0/0 is undefined
                return Self::NAN;
            } else if negative {
                return Self::NEG_INFINITY;
            } else {
                return Self::INFINITY;
            }
        }
        if self.is_zero() {
            // Already checked for rhs being zero
            if negative {
                return Self::NEG_ZERO;
            } else {
                return Self::ZERO;
            }
        }
        // Increase precision of the numerator until the denominator goes into
        // it an exact number of times, or until the maximum precision - of
        // `u64` - is reached
        let mut lhs = self;
        //let mut iterations: u8 = 0;
        // Loop because we only want to increase the precision as much as we
        // absolutely have to
        while !lhs.significand.is_multiple_of(rhs.significand) {
            // iterations += 1;
            // if iterations > 100 {
            //     panic!("{}", iterations)
            // }
            // Crucially, we allow the precision to increase beyond 16 sf up to
            // the maximum of `u64`
            // Only allowing 16 sf means that the max significand is
            // 0b0000000000100011100001101111001001101111110000001111111111111111
            // which fits into 54 bits
            // This gives us ~10 bits of spare precision to use (~3 sig figs)
            match lhs.increase_precision_unbounded_checked(1) {
                Some(new) => {
                    lhs = new;
                }
                None => {
                    // Max precision was already reached last iteration
                    break;
                }
            }
        }
        let significand = lhs.significand / rhs.significand;
        let exponent = lhs.exponent - rhs.exponent;
        Self {
            uncertainty: 0,
            uncertainty_scale: 0,
            nan: false,
            inf: false,
            negative,
            exponent,
            significand,
        }
    }

    /// Calculates `self.powi(rhs)` without uncertainty, permitting values for the
    /// significand greater than `SciDecimal::MAX_SIGNIFICAND` and up to `u64::MAX`.
    fn unbounded_powi(self, n: i32) -> Self {
        if !self.is_normal() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        if n.is_negative() {
            self.powi(n.abs()).inv()
        } else {
            let number = self.significand_signed().pow(n.try_into().unwrap());
            let exponent = self.exponent * i16::try_from(n).unwrap();
            Self::new(number, exponent)
        }
    }

    /// Calculates `self.pow(rhs)` without uncertainty, permitting values for the
    /// significand greater than `SciDecimal::MAX_SIGNIFICAND` and up to `u64::MAX`.
    fn unbounded_powf(self, rhs: Self) -> Self {
        if !(self.is_normal() && rhs.is_normal()) {
            todo!("Special values are not yet handled correctly by this method!")
        }
        todo!()
    }
}

impl Add for SciDecimal {
    type Output = Self;

    /// Performs the `+` operation.
    ///
    /// # Special values
    ///
    /// - ±0: no special behaviour
    ///
    /// - ±∞: if one number is an infinity, that infinity is returned; otherwise:
    ///   - ∞ + ∞ → ∞
    ///   - -∞ + -∞ → -∞
    ///   - ∞ - ∞ → `NaN`
    ///   - -∞ + ∞ → `NaN`
    ///
    /// - `NaN`: if either number is `NaN`, returns `NaN`
    fn add(self, rhs: Self) -> Self {
        let exact = self.unbounded_add(rhs);
        let result = if self.is_exact() && rhs.is_exact() {
            exact
        } else {
            let uncertainty =
                ((self.uncertainty().pow(2.into())) + rhs.uncertainty().pow(2.into())).sqrt();
            exact.with_uncertainty(uncertainty)
        };
        if result.significand > Self::MAX_SIGNIFICAND {
            result.round_sf(16, RoundingMode::HalfUp)
        } else {
            result
        }
    }
}

impl Add for &SciDecimal {
    type Output = SciDecimal;

    fn add(self, rhs: Self) -> SciDecimal {
        *self + *rhs
    }
}

impl Sub for SciDecimal {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let rhs = -rhs;
        self + rhs
    }
}

impl Sub for &SciDecimal {
    type Output = SciDecimal;

    fn sub(self, rhs: Self) -> SciDecimal {
        *self - *rhs
    }
}

impl Mul for SciDecimal {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let exact = self.unbounded_mul(rhs);
        let result = if self.is_exact() && rhs.is_exact() {
            exact
        } else {
            let uncertainty =
                (self.relative_uncertainty().powi(2) + rhs.relative_uncertainty().powi(2)).sqrt()
                    * exact.abs();
            exact.with_uncertainty(uncertainty)
        };
        if result.significand > Self::MAX_SIGNIFICAND {
            result.round_sf(16, RoundingMode::HalfUp)
        } else {
            result
        }
    }
}

impl Mul for &SciDecimal {
    type Output = SciDecimal;

    fn mul(self, rhs: Self) -> SciDecimal {
        *self * *rhs
    }
}

impl Div for SciDecimal {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        let exact = self.unbounded_div(rhs);
        let result = if self.is_exact() && rhs.is_exact() {
            exact
        } else {
            let uncertainty =
                (self.relative_uncertainty().powi(2) + rhs.relative_uncertainty().powi(2)).sqrt()
                    * exact.abs();
            exact.with_uncertainty(uncertainty)
        };
        if result.significand > Self::MAX_SIGNIFICAND {
            result.round_sf(16, RoundingMode::HalfUp)
        } else {
            result
        }
    }
}

impl Div for &SciDecimal {
    type Output = SciDecimal;

    fn div(self, rhs: Self) -> SciDecimal {
        *self / *rhs
    }
}

impl Rem for SciDecimal {
    type Output = Self;

    /// Performs the `%` operation.
    ///
    /// WARNING: Uncertainty propagation is not yet implemented for this method,
    /// and the returned result will be exact.
    fn rem(self, rhs: Self) -> Self {
        // Handle NaN
        if self.nan | rhs.nan {
            return Self::NAN;
        }
        // Handle infinities
        if self.inf {
            // Can't find remainder of infinity
            return Self::NAN;
        } else if rhs.inf {
            return self;
        }
        // Handle zeros
        if rhs.is_zero() {
            // n % 0 is undefined just like n / 0
            return Self::NAN;
        }
        // TODO implement natively, not via Decimal
        dbg!(&self);
        dbg!(self.to_string());
        let number = Decimal::cast_from(self.number()) % Decimal::cast_from(rhs.number());
        // Don't calculate uncertainty as the remainder function is discontinuous,
        // making it tricky
        number.cast()
    }
}

impl Rem for &SciDecimal {
    type Output = SciDecimal;

    /// Performs the `%` operation.
    ///
    /// WARNING: Uncertainty propagation is not yet implemented for this method,
    /// and the returned result will be exact.
    fn rem(self, rhs: Self) -> SciDecimal {
        *self % *rhs
    }
}

impl Pow<Self> for SciDecimal {
    type Output = Self;

    /// Raise the `SciDecimal` to a `SciDecimal` power.
    fn pow(self, rhs: Self) -> Self {
        let exact = if rhs.is_exact()
            && rhs.exponent.is_zero()
            && (rhs.exponent <= i8::MAX.into() && rhs.exponent >= i8::MIN.into())
        {
            let n = rhs.significand_signed();
            self.unbounded_powi(
                n.try_into()
                    .expect("n has already been checked and should fit into even an i8"),
            )
        } else {
            self.unbounded_powf(rhs)
        };
        if result.significand > Self::MAX_SIGNIFICAND {
            result.round_sf(16, RoundingMode::HalfUp)
        } else {
            result
        }
    }
}

impl Pow<Self> for &SciDecimal {
    type Output = SciDecimal;

    fn pow(self, rhs: Self) -> SciDecimal {
        (*self).pow(*rhs)
    }
}

impl Neg for SciDecimal {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            negative: !self.negative,
            ..self
        }
    }
}

impl Neg for &SciDecimal {
    type Output = SciDecimal;

    #[inline]
    fn neg(self) -> SciDecimal {
        SciDecimal {
            negative: !self.negative,
            ..*self
        }
    }
}

impl Inv for SciDecimal {
    type Output = Self;

    #[inline]
    fn inv(self) -> Self {
        Self::ONE / self
    }
}

impl Inv for &SciDecimal {
    type Output = SciDecimal;

    #[inline]
    fn inv(self) -> SciDecimal {
        SciDecimal::ONE / *self
    }
}

/// Methods to get correlated uncertainties.
impl SciDecimal {
    /// Function that calculates a result and its uncertainty for any non-linear
    /// differentiable function f(a, b).
    ///
    /// `f` is a function that gives the *exact* result of f(a, b), where a is `self`.
    ///
    /// `partderiv_a` and `partderiv_b` are the partial derivatives of f w.r.t. a and b.
    ///
    /// `ρ_ab` is the correlation between a and b, which should be 0 (uncorrelated),
    /// 1 (completely correlated) or a value between the two.
    fn calculate_with_uncertainty<F, A, B>(
        self,
        b: Self,
        f: F,
        partderiv_a: A,
        partderiv_b: B,
        ρ_ab: Self,
    ) -> Self
    where
        F: Fn(Self, Self) -> Self,
        A: Fn(Self, Self) -> Self,
        B: Fn(Self, Self) -> Self,
    {
        let exact = f(self, b);
        let uncertainty_fn = uncertainty_fn_generator(partderiv_a, partderiv_b);
        let uncertainty = uncertainty_fn(self, b, self.uncertainty(), b.uncertainty(), ρ_ab);
        exact.with_uncertainty(uncertainty)
    }

    /// Calculates the sum of two values with correlated uncertainties.
    ///
    /// `correlation` must be 0 (uncorrelated), 1 (completely correlated) or a
    /// value between the two.
    pub fn correlated_add(self, rhs: Self, correlation: Self) -> Self {
        if correlation < Self::ZERO || correlation > Self::ONE {
            panic!("Correlation must be between 0 and 1!")
        }
        self.calculate_with_uncertainty(
            rhs,
            Self::add,
            |a, b| Self::ONE,
            |a, b| Self::ONE,
            correlation,
        )
    }
}

/// String-related methods.
impl SciDecimal {
    pub fn to_plain_string(&self) -> String {
        // Handle NaN
        if self.nan {
            return String::from("NaN");
        }
        // Get sign character
        let sign = if self.negative {
            String::from("-")
        } else {
            String::new()
        };
        // Handle infinities
        if self.inf {
            return format!("{sign}inf");
        }
        // Handle zeros
        if self.is_zero() {
            // TODO Have this display the uncertainty properly once they can be
            // displayed with +/-
            return format!("{}0", sign);
        }
        let significand = self.significand;
        let uncertainty = if self.is_exact() {
            String::new()
        } else {
            format!("({})", self.uncertainty)
        };
        // Integers
        if self.precision() > 0 {
            // Need to add appropriate zeros as padding
            // e.g. 25(2)e4 needs to become 250000(20000)
            let zeros = "0".repeat(self.precision() as usize);
            let uncertainty = if !uncertainty.is_empty() {
                uncertainty.replace(")", &zeros) + ")"
            } else {
                uncertainty
            };
            format!("{sign}{significand}{zeros}{uncertainty}")
        // Numbers with both integral and fractional parts
        } else if self.precision_most_significant_fig() >= 0 {
            // 3.1 has precision = -1, sigfigs = 2
            // 42.764 has precision = -3, sigfigs = 5
            // 3.02 has precision = -2, sigfigs = 3
            let int_figs = self.sf() as u16 - self.precision().unsigned_abs();
            let mut int = significand.to_string();
            let frac = int.split_off(int_figs.into());
            format!("{sign}{int}.{frac}{uncertainty}")
        // Numbers with only a fractional part
        } else {
            // 0.005 needs to have two zeros, precision = -3, sigfigs = 1
            let zeros = // (-3).abs() - 1 = 2
                "0".repeat((self.precision().unsigned_abs() - self.sf() as u16).into());
            format!("{sign}0.{zeros}{significand}{uncertainty}")
        }
    }

    pub fn to_scientific_string(&self) -> String {
        // Handle NaN
        if self.nan {
            return String::from("NaN");
        }
        // Get sign character
        let sign = if self.negative {
            String::from("-")
        } else {
            String::new()
        };
        // Handle infinities
        if self.inf {
            return format!("{}inf", sign);
        }
        let uncertainty = if self.is_exact() {
            String::new()
        } else {
            format!("({})", self.uncertainty)
        };
        let (int, zeros, frac, _, exp) = self.scientific_parts().unwrap();
        let zeros = "0".repeat(zeros.into());
        // Fractional part might not have any places at all (e.g. 2e6)
        if frac == 0 {
            format!("{int}{uncertainty}e{exp}")
        } else {
            format!("{int}.{zeros}{frac}{uncertainty}e{exp}")
        }
    }
}

impl fmt::Display for SciDecimal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Numbers with up to five places either side of the decimal point should
        // be displayed using normal notation:
        // - 0.0325 = 3.25e-2 = (325, -4) => "0.0325"
        // - 85.130 = 8.5130e1 = (85130, -3) => "85.130"
        // If this is exceeded, display in scientific notation:
        // - 0.000325 = 3.25e-4 = (325, -6) => "3.25e-4"
        // - 8174036 = 8.174036e6 = (8174036, 0) => "8.174036e6"
        // Scientific notation should also be used if there are insignificant zeros
        // before the decimal point, so that the precision is indicated:
        // - 81700 with 3 sf = 8.17e4 = (817, 2) => "8.17e4"
        // - 81700 with 5 sf = 8.1700e4 = (81700, 0) => "81700"
        if self.precision() <= 0
            && self.precision() >= -5
            && self.precision_most_significant_fig() <= 4
        {
            write!(f, "{}", self.to_plain_string())
        // Otherwise, use scientific notation
        } else {
            write!(f, "{}", self.to_scientific_string())
        }
    }
}

impl FromStr for SciDecimal {
    type Err = SciNumError;

    /// Parses a string and attempts to create a corresponding `SciDecimal`.
    ///
    /// A correctly formed string will *always* return a `SciDecimal`:
    ///
    /// - Excess precision is rounded to 16 significant figures (according to
    ///   [`RoundingMode::HalfEven`]).
    ///
    /// - If the absolute magnitude of the number is too *large* to be represented,
    ///   an infinity with the appropriate sign is returned.
    ///
    /// - If the absolute magnitude of the number is too *small* to be represented,
    ///   a zero with the appropriate sign is returned.
    ///
    /// In this way the behaviour is like that of [`SciCast::cast`], and indeed
    /// this method is used to effect the casts from several types.
    ///
    /// Fails if the string cannot be parsed at all.
    ///
    /// # Special values
    ///
    /// `inf` and `+inf` are parsed to positive infinity, `-inf` and `−inf` (with
    /// a hyphen-minus or a proper minus sign) are parsed to negative infinity.
    ///
    /// Likewise, `0` and `+0` are parsed to positive zero, `-0` and `−0` to negative.
    ///
    /// All variations of `NaN` are parsed case-insensitively to `NaN`, with any
    /// sign ignored.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // TODO Support special values
        // TODO Allow underscores to group digits
        let re =
            Regex::new(r"^(-)?(\d+)?(?:[.,](\d+))?(?:\((\d+)\))?(?:[eE]([+-]?\d+))?$").unwrap();
        let caps = re.captures(s).ok_or(SciNumError::Parse(s.into()))?;
        // Example given with "6.971e-7"
        let negative = caps.get(1).is_some(); // false
        let int = caps.get(2).map_or("", |m| m.as_str()); // "6"
        let frac = caps.get(3).map_or("", |m| m.as_str()); // "971"
        let frac_places = frac.len(); // 3
        let significand_string = int.to_owned() + frac;
        let mut significand_slice = &*significand_string;
        // Remove any leading zeros
        significand_slice = significand_slice.trim_start_matches('0');
        // But if the significand is just zero, it will have been removed
        if significand_slice.is_empty() {
            significand_slice = "0";
        };
        // If the precision in the string is too high, get it down to 19 sf
        // We'll round to 16 sf at the end
        let truncated_places = if significand_slice.len() > 19 {
            let excess_places = significand_slice.len() - 19;
            // OK to re-slice like this because our regex will only ever return
            // a string of 1-byte chars
            significand_slice = &significand_slice[..19];
            excess_places
        } else {
            0
        };
        let significand =
            u64::from_str(significand_slice).map_err(|_e| SciNumError::Parse(s.into()))?; // "6971"
        let uncertainty = caps
            .get(4)
            .map_or(Ok(0), |m| u32::from_str(m.as_str()))
            .map_err(|_e| SciNumError::Parse(s.into()))?; // 0
        let exponent = caps
            .get(5)
            .map_or(Ok(0), |m| i16::from_str(m.as_str()))
            .map_err(|_e| SciNumError::Parse(s.into()))?
            - frac_places as i16
            + truncated_places as i16; // -7
        // "6.971e-7" should be represented as (6971, -10)
        let num = Self {
            uncertainty,
            uncertainty_scale: 0,
            nan: false,
            inf: false,
            negative,
            exponent,
            significand,
        };
        if num.sf() > 16 {
            Ok(num.round_sf(16, RoundingMode::HalfUp))
        } else {
            Ok(num)
        }
    }
}

#[macro_export]
macro_rules! sci {
    ($s:expr) => {
        <SciDecimal as std::str::FromStr>::from_str(stringify!($s)).unwrap()
    };
}

macro_rules! impl_from_int {
    ($T:ty) => {
        impl From<$T> for SciDecimal {
            fn from(t: $T) -> Self {
                Self::new(t.into(), 0)
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
    use rust_decimal_macros::dec;

    #[test]
    fn new_from_int() {
        // Using new
        let n = SciDecimal::new(30, 0);
        assert_eq!(n.number(), SciDecimal::new(30, 0));
        assert_eq!(n.uncertainty(), SciDecimal::new(0, 0));
        // Using from
        let n = SciDecimal::from(42);
        assert_eq!(n.number(), SciDecimal::new(42, 0));
        assert_eq!(n.uncertainty(), SciDecimal::new(0, 0));
    }

    #[test]
    fn new_from_int_with_uncertainty() {
        let n = SciDecimal::new_with_uncertainty(20, 2, 0);
        assert_eq!(n.number(), SciDecimal::from(20));
        assert_eq!(n.uncertainty(), SciDecimal::new(2, 0));
    }

    #[test]
    fn from_scientific_parts() {
        let n1 = SciDecimal::from_scientific_parts(67, 0, 2, 0, 0); // 67.2
        assert_eq!(n1.to_string(), "67.2");
        assert_eq!(n1, SciDecimal::new(672, -1));

        let n2 = SciDecimal::from_scientific_parts(67, 1, 0, 0, 0); // 67.0
        assert_eq!(n2.to_string(), "67.0");
        assert_eq!(n2, SciDecimal::new(670, -1));

        let n3 = SciDecimal::from_scientific_parts(2, 0, 36, 0, 5);
        assert_eq!(n3.to_string(), "2.36e5");
        assert_eq!(n3, sci!(2.36e5));

        let n4 = SciDecimal::from_scientific_parts(23, 0, 61, 0, -7);
        assert_eq!(n4.to_string(), "2.361e-6");
        assert_eq!(n4, sci!(2.361e-6));
    }

    #[test]
    fn new_large() {
        let _n = SciDecimal::new(236, 40);
    }

    #[test]
    fn new_small() {
        let _n = SciDecimal::new(49, -76);
    }

    #[test]
    fn new_largest_exponent() {
        let _n = SciDecimal::new(1, i16::MAX);
    }

    #[test]
    fn new_smallest_exponent() {
        let _n = SciDecimal::new(1, i16::MIN);
    }

    #[test]
    fn new_largest_significand() {
        let _n = SciDecimal::new(SciDecimal::MAX_SIGNIFICAND_SIGNED, 0);
    }

    #[test]
    fn new_largest_negative_significand() {
        let _n = SciDecimal::new(SciDecimal::MIN_SIGNIFICAND_SIGNED, 0);
    }

    #[test]
    #[should_panic]
    fn new_invalid_significand() {
        let _n = SciDecimal::new(SciDecimal::MAX_SIGNIFICAND_SIGNED + 1, 0);
    }

    #[test]
    fn nan() {
        // Important to check this not just with the `NAN` const but also to
        // confirm that the different flag bits override each other in the
        // expected way.
        // Any `SciDecimal` with `self.nan == true` should be considered a NaN,
        // even if `self.inf` is `true`, so there are 2^127 different NaNs.
        // It is important that none of them are ever treated as a normal number,
        // or as an infinity, or as negative, etc.
        for nan in [
            SciDecimal::NAN,
            SciDecimal::nan(),
            SciDecimal {
                uncertainty: 3,
                uncertainty_scale: 0,
                nan: true,
                inf: true,
                negative: false,
                exponent: 1,
                significand: 0,
            },
            SciDecimal {
                uncertainty: 3,
                uncertainty_scale: -1,
                nan: true,
                inf: true,
                negative: true,
                exponent: -4,
                significand: 25,
            },
            SciDecimal {
                uncertainty: 373,
                uncertainty_scale: 2,
                nan: true,
                inf: true,
                negative: false,
                exponent: 38,
                significand: 9234872,
            },
        ] {
            assert!(nan.is_nan());
            assert_ne!(nan, SciDecimal::NAN); // Characteristic of NaN
            assert!(nan.is_nan());
            assert!(!nan.is_infinite());
            assert!(!nan.is_finite()); // NaN is neither finite nor infinite
            assert!(!nan.is_normal());
            assert!(!nan.is_zero());
            assert!(nan.number().is_nan());
            assert!(nan.uncertainty().is_nan());
        }
    }

    #[test]
    fn infinities() {
        // Similarly, any `SciDecimal` that has `self.inf == true` is an infinity
        // (*unless it also has `self.nan == true`*, see above), and thus there
        // are also 2^126 different infinities…
        for (inf, ninf) in [
            (SciDecimal::INFINITY, SciDecimal::NEG_INFINITY),
            (SciDecimal::infinity(), SciDecimal::neg_infinity()),
        ] {
            assert!(!inf.is_nan());
            assert!(!ninf.is_nan());
            assert!(inf.is_infinite());
            assert!(ninf.is_infinite());
            assert!(!inf.is_finite());
            assert!(!ninf.is_finite());
            assert!(!inf.is_normal());
            assert!(!ninf.is_normal());
            assert!(!inf.is_zero());
            assert!(!ninf.is_zero());
            assert_ne!(inf, ninf);
        }
    }

    #[test]
    fn uncertainty() {
        let n = SciDecimal::new_with_uncertainty(30, 5, 0);
        assert_eq!(n.uncertainty(), SciDecimal::new(5, 0));
    }

    #[test]
    fn relative_uncertainty() {
        let n = SciDecimal::new_with_uncertainty(20, 2, 0);
        assert_eq!(n.relative_uncertainty(), SciDecimal::new(1, -1));

        let n2 = SciDecimal::new_with_uncertainty(500, 5, 0);
        assert_eq!(n2.relative_uncertainty(), SciDecimal::new(1, -2));

        let n3 = SciDecimal::new_with_uncertainty(1000, 15, 0);
        assert_eq!(n3.relative_uncertainty(), SciDecimal::new(15, -3));
    }

    #[test]
    fn sf() {
        let n = SciDecimal::from_scientific_parts(123, 0, 45, 0, 0);
        assert_eq!(n.sf(), 5);

        let n2 = SciDecimal::from_scientific_parts(123, 1, 45, 0, 0);
        assert_eq!(n2.sf(), 6);

        let n3 = sci!(0.00123);
        assert_eq!(n3.sf(), 3);

        let n4 = SciDecimal::new(1234, 0);
        assert_eq!(n4.sf(), 4);
    }

    #[test]
    fn sf_trailing_zeros() {
        let n = SciDecimal::from_scientific_parts(123, 0, 4500, 0, 0);
        assert_eq!(n.sf(), 7);

        let n2 = sci!(0.001230);
        assert_eq!(n2.sf(), 4);

        let n3 = SciDecimal::new(1230, 0);
        assert_eq!(n3.sf(), 4);
    }

    #[test]
    fn dp() {
        assert_eq!(sci!(0.2).dp(), 1);
        assert_eq!(sci!(0.02).dp(), 2);
        assert_eq!(sci!(0.020).dp(), 3);
        assert_eq!(sci!(0.021).dp(), 3);
        assert_eq!(sci!(2).dp(), 0);
        assert_eq!(sci!(20).dp(), 0);
        assert_eq!(sci!(2.0e5).dp(), 0);
        assert_eq!(sci!(2.0).dp(), 1);
        assert_eq!(sci!(2.0e-3).dp(), 4);
    }

    #[test]
    fn precision() {
        assert_eq!(sci!(0.02).precision(), -2);
        assert_eq!(sci!(0.020).precision(), -3);
        assert_eq!(sci!(2).precision(), 0);
        assert_eq!(SciDecimal::new(2, 3).precision(), 3);
        assert_eq!(SciDecimal::from_str("2e3").unwrap().precision(), 3);
    }

    #[test]
    fn precision_most_significant_fig() {
        assert_eq!(sci!(0.02).precision_most_significant_fig(), -2);
        assert_eq!(sci!(0.025).precision_most_significant_fig(), -2);
        assert_eq!(sci!(0.020).precision_most_significant_fig(), -2);
        assert_eq!(sci!(2).precision_most_significant_fig(), 0);
        assert_eq!(sci!(321).precision_most_significant_fig(), 2);
    }

    #[test]
    fn is_exact() {
        let n1 = sci!(45.1);
        let n2 = SciDecimal::new_with_uncertainty(500, 5, 0);
        assert!(n1.is_exact());
        assert!(!n2.is_exact());
    }

    #[test]
    fn eq() {
        // Basic case
        assert_eq!(SciDecimal::new(3, 0), SciDecimal::new(3, 0));
        // Not equal, basic case
        assert_ne!(SciDecimal::new(3, 0), SciDecimal::new(4, 0));
        // Both zero
        assert_eq!(SciDecimal::new(0, 0), SciDecimal::new(0, 0));
        // Both zero, one is negative zero
        assert_eq!(SciDecimal::new(0, 0), SciDecimal::new(-0, 0));
        // Opposite sign but same significand
        assert_ne!(SciDecimal::new(3, 0), SciDecimal::new(-3, 0));
        // Same value but different precision
        assert_eq!(SciDecimal::new(200, 3), SciDecimal::new(2, 5));
        // Same value but different precision, small numbers
        assert_eq!(SciDecimal::new(200, 3), SciDecimal::new(2, 5));
    }

    #[test]
    fn truncate() {
        // Positive
        let n = sci!(25.6949);
        assert_eq!(n.trunc_sf(2), sci!(25));
        assert_eq!(n.trunc_sf(3), sci!(25.6));
        // Negative
        let n = sci!(-3.794718);
        assert_eq!(n.trunc_sf(4), sci!(-3.794));
        assert_eq!(n.trunc_sf(3), sci!(-3.79));
        // Integer
        let n = sci!(4327890);
        assert_eq!(n.trunc_sf(4), sci!(4.327e6));
        assert_eq!(n.trunc_sf(5), sci!(4.3278e6));
        // Smaller than 1
        let n = sci!(0.4327890);
        assert_eq!(n.trunc_sf(4), sci!(4.327e-1));
        assert_eq!(n.trunc_sf(5), sci!(4.3278e-1));
    }

    #[test]
    fn increase_precision() {
        // Currently fails due to Display failing
        //let n = sci!(25.69);
        //assert_eq!(n.to_string(), "25.69");
        //assert_eq!(n.add_sf(2).to_string(), "25.6900");
        let n2 = sci!(2.69e7);
        assert_eq!(n2.to_string(), "2.69e7");
        assert_eq!(n2.increase_precision(2).to_string(), "2.6900e7");
    }

    #[test]
    #[rustfmt::skip]
    fn round_prec() {
        // Next digits < half
        let n = sci!(1.23);
        assert_eq!(n.round_precision(-1, RoundingMode::Up),         sci!(1.3));
        assert_eq!(n.round_precision(-1, RoundingMode::Down),       sci!(1.2));
        assert_eq!(n.round_precision(-1, RoundingMode::Ceiling),    sci!(1.3));
        assert_eq!(n.round_precision(-1, RoundingMode::Floor),      sci!(1.2));
        assert_eq!(n.round_precision(-1, RoundingMode::HalfUp),     sci!(1.2));
        assert_eq!(n.round_precision(-1, RoundingMode::HalfDown),   sci!(1.2));
        assert_eq!(n.round_precision(-1, RoundingMode::HalfEven),   sci!(1.2));
        // As above but negative
        let n = sci!(-1.23);
        assert_eq!(n.round_precision(-1, RoundingMode::Up),         sci!(-1.3));
        assert_eq!(n.round_precision(-1, RoundingMode::Down),       sci!(-1.2));
        assert_eq!(n.round_precision(-1, RoundingMode::Ceiling),    sci!(-1.2));
        assert_eq!(n.round_precision(-1, RoundingMode::Floor),      sci!(-1.3));
        assert_eq!(n.round_precision(-1, RoundingMode::HalfUp),     sci!(-1.2));
        assert_eq!(n.round_precision(-1, RoundingMode::HalfDown),   sci!(-1.2));
        assert_eq!(n.round_precision(-1, RoundingMode::HalfEven),   sci!(-1.2));
        // Different precision, next digits still < half
        let n = sci!(1.623);
        assert_eq!(n.round_precision(-2, RoundingMode::Up),         sci!(1.63));
        assert_eq!(n.round_precision(-2, RoundingMode::Down),       sci!(1.62));
        assert_eq!(n.round_precision(-2, RoundingMode::Ceiling),    sci!(1.63));
        assert_eq!(n.round_precision(-2, RoundingMode::Floor),      sci!(1.62));
        assert_eq!(n.round_precision(-2, RoundingMode::HalfUp),     sci!(1.62));
        assert_eq!(n.round_precision(-2, RoundingMode::HalfDown),   sci!(1.62));
        assert_eq!(n.round_precision(-2, RoundingMode::HalfEven),   sci!(1.62));
        // Integers, next digits still < half
        let n = sci!(1230);
        assert_eq!(n.round_precision(2, RoundingMode::Up),          sci!(1300));
        assert_eq!(n.round_precision(2, RoundingMode::Down),        sci!(1200));
        assert_eq!(n.round_precision(2, RoundingMode::Ceiling),     sci!(1300));
        assert_eq!(n.round_precision(2, RoundingMode::Floor),       sci!(1200));
        assert_eq!(n.round_precision(2, RoundingMode::HalfUp),      sci!(1200));
        assert_eq!(n.round_precision(2, RoundingMode::HalfDown),    sci!(1200));
        assert_eq!(n.round_precision(2, RoundingMode::HalfEven),    sci!(1200));
        // Next digits > half
        let n = sci!(1.27);
        assert_eq!(n.round_precision(-1, RoundingMode::Up),         sci!(1.3));
        assert_eq!(n.round_precision(-1, RoundingMode::Down),       sci!(1.2));
        assert_eq!(n.round_precision(-1, RoundingMode::Ceiling),    sci!(1.3));
        assert_eq!(n.round_precision(-1, RoundingMode::Floor),      sci!(1.2));
        assert_eq!(n.round_precision(-1, RoundingMode::HalfUp),     sci!(1.3));
        assert_eq!(n.round_precision(-1, RoundingMode::HalfDown),   sci!(1.3));
        assert_eq!(n.round_precision(-1, RoundingMode::HalfEven),   sci!(1.3));
        // Next digits = half
        let n = sci!(1.25);
        assert_eq!(n.round_precision(-1, RoundingMode::Up),         sci!(1.3));
        assert_eq!(n.round_precision(-1, RoundingMode::Down),       sci!(1.2));
        assert_eq!(n.round_precision(-1, RoundingMode::Ceiling),    sci!(1.3));
        assert_eq!(n.round_precision(-1, RoundingMode::Floor),      sci!(1.2));
        assert_eq!(n.round_precision(-1, RoundingMode::HalfUp),     sci!(1.3));
        assert_eq!(n.round_precision(-1, RoundingMode::HalfDown),   sci!(1.2));
        assert_eq!(n.round_precision(-1, RoundingMode::HalfEven),   sci!(1.2));
        // Next digits = half, rounding to even goes up
        let n = sci!(1.35);
        assert_eq!(n.round_precision(-1, RoundingMode::Up),         sci!(1.4));
        assert_eq!(n.round_precision(-1, RoundingMode::Down),       sci!(1.3));
        assert_eq!(n.round_precision(-1, RoundingMode::Ceiling),    sci!(1.4));
        assert_eq!(n.round_precision(-1, RoundingMode::Floor),      sci!(1.3));
        assert_eq!(n.round_precision(-1, RoundingMode::HalfUp),     sci!(1.4));
        assert_eq!(n.round_precision(-1, RoundingMode::HalfDown),   sci!(1.3));
        assert_eq!(n.round_precision(-1, RoundingMode::HalfEven),   sci!(1.4));
        // Next digits = half, negative
        let n = sci!(-1.35);
        assert_eq!(n.round_precision(-1, RoundingMode::Up),         sci!(-1.4));
        assert_eq!(n.round_precision(-1, RoundingMode::Down),       sci!(-1.3));
        assert_eq!(n.round_precision(-1, RoundingMode::Ceiling),    sci!(-1.3));
        assert_eq!(n.round_precision(-1, RoundingMode::Floor),      sci!(-1.4));
        assert_eq!(n.round_precision(-1, RoundingMode::HalfUp),     sci!(-1.4));
        assert_eq!(n.round_precision(-1, RoundingMode::HalfDown),   sci!(-1.3));
        assert_eq!(n.round_precision(-1, RoundingMode::HalfEven),   sci!(-1.4));
        // Next digits start with 5 but are > half
        let n = sci!(1.252);
        assert_eq!(n.round_precision(-1, RoundingMode::Up),         sci!(1.3));
        assert_eq!(n.round_precision(-1, RoundingMode::Down),       sci!(1.2));
        assert_eq!(n.round_precision(-1, RoundingMode::Ceiling),    sci!(1.3));
        assert_eq!(n.round_precision(-1, RoundingMode::Floor),      sci!(1.2));
        assert_eq!(n.round_precision(-1, RoundingMode::HalfUp),     sci!(1.3));
        assert_eq!(n.round_precision(-1, RoundingMode::HalfDown),   sci!(1.3));
        assert_eq!(n.round_precision(-1, RoundingMode::HalfEven),   sci!(1.3));
    }

    #[test]
    fn round_dp() {
        assert_eq!(sci!(1.4324).round_dp(0, RoundingMode::HalfUp), sci!(1));
        assert_eq!(sci!(1.4324).round_dp(1, RoundingMode::HalfUp), sci!(1.4));
        assert_eq!(sci!(1.4324).round_dp(2, RoundingMode::HalfUp), sci!(1.43));
        assert_eq!(sci!(1.4324).round_dp(3, RoundingMode::HalfUp), sci!(1.432));
        assert_eq!(sci!(1.4324).round_dp(4, RoundingMode::HalfUp), sci!(1.4324));
        assert_eq!(
            sci!(1.4324).round_dp(5, RoundingMode::HalfUp),
            sci!(1.43240)
        );
        assert_eq!(sci!(0.0024).round_dp(3, RoundingMode::HalfUp), sci!(0.002));
        assert_eq!(sci!(0.0024).round_dp(2, RoundingMode::HalfUp), sci!(0));
        assert_eq!(sci!(1.435).round_dp(2, RoundingMode::HalfUp), sci!(1.44));
        assert_eq!(sci!(-1.435).round_dp(2, RoundingMode::HalfUp), sci!(-1.44));
        assert_eq!(sci!(14).round_dp(0, RoundingMode::HalfUp), sci!(14));
        assert_eq!(sci!(3.5e4).round_dp(0, RoundingMode::HalfUp), sci!(3.5e4));
        assert_eq!(sci!(9.96).round_dp(1, RoundingMode::HalfUp), sci!(10.0));
    }

    #[test]
    fn round_sf() {
        assert_eq!(sci!(1.4324).round_sf(1, RoundingMode::HalfUp), sci!(1));
        assert_eq!(sci!(1.4324).round_sf(2, RoundingMode::HalfUp), sci!(1.4));
        assert_eq!(sci!(1.4924).round_sf(2, RoundingMode::HalfUp), sci!(1.5));
        assert_eq!(sci!(1.4324).round_sf(3, RoundingMode::HalfUp), sci!(1.43));
        assert_eq!(sci!(1.5324).round_sf(1, RoundingMode::HalfUp), sci!(2));
        assert_eq!(sci!(1.5324).round_sf(2, RoundingMode::HalfUp), sci!(1.5));
        assert_eq!(
            sci!(0.001234).round_sf(2, RoundingMode::HalfUp),
            sci!(0.0012)
        );
        assert_eq!(
            sci!(0.001264).round_sf(2, RoundingMode::HalfUp),
            sci!(0.0013)
        );
        assert_eq!(sci!(1.45).round_sf(2, RoundingMode::HalfUp), sci!(1.5));
        assert_eq!(sci!(1.45).round_sf(2, RoundingMode::HalfDown), sci!(1.4));
        assert_eq!(sci!(1.45).round_sf(2, RoundingMode::HalfEven), sci!(1.4));
        assert_eq!(sci!(1.55).round_sf(2, RoundingMode::HalfEven), sci!(1.6));
        assert_eq!(sci!(1.4354).round_sf(3, RoundingMode::HalfDown), sci!(1.44));
        assert_eq!(sci!(0.0024).round_sf(1, RoundingMode::HalfUp), sci!(0.002));
        assert_eq!(sci!(0.0024).round_sf(2, RoundingMode::HalfUp), sci!(0.0024));
        assert_eq!(sci!(12345).round_sf(3, RoundingMode::HalfUp), sci!(12300));
        assert_eq!(sci!(12645).round_sf(3, RoundingMode::HalfUp), sci!(12600));
        assert_eq!(sci!(99999).round_sf(2, RoundingMode::HalfUp), sci!(100000));
        assert_eq!(sci!(1.4).round_sf(2, RoundingMode::HalfUp), sci!(1.4));
        assert_eq!(sci!(1.4).round_sf(4, RoundingMode::HalfUp), sci!(1.400));
        assert_eq!(sci!(9.96).round_sf(1, RoundingMode::HalfUp), sci!(10));
        assert_eq!(sci!(9.96).round_sf(2, RoundingMode::HalfUp), sci!(10));
    }

    #[test]
    fn round_match_uncertainty() {
        // Exact values remain unchanged
        assert_eq!(
            sci!(1.4324).round_match_uncertainty(RoundingMode::HalfUp),
            sci!(1.4324)
        );
        // If precisions already match, no rounding occurs
        assert_eq!(
            SciDecimal::from_str("1.4324(6)")
                .unwrap()
                .round_match_uncertainty(RoundingMode::HalfUp),
            SciDecimal::from_str("1.4324(6)").unwrap()
        );
        // Uncertainty less precise than number
        assert_eq!(
            sci!(1.3424)
                .with_uncertainty(sci!(0.03))
                .round_match_uncertainty(RoundingMode::HalfUp),
            SciDecimal::from_str("1.34(3)").unwrap()
        );
        // Uncertainty more precise than number
        assert_eq!(
            sci!(1.3424)
                .with_uncertainty(sci!(0.0338274))
                .round_match_uncertainty(RoundingMode::HalfUp),
            SciDecimal::from_str("1.3424000(338274)").unwrap()
        );
    }

    #[test]
    fn round_match_uncertainty_sf() {
        assert_eq!(
            SciDecimal::from_str("1.4324(6)")
                .unwrap()
                .round_match_uncertainty_sf(1, RoundingMode::HalfUp),
            SciDecimal::from_str("1.4324(6)").unwrap()
        );
        assert_eq!(
            SciDecimal::from_str("1.4324(16)")
                .unwrap()
                .round_match_uncertainty_sf(1, RoundingMode::HalfUp),
            SciDecimal::from_str("1.432(2)").unwrap()
        );
        assert_eq!(
            SciDecimal::from_str("1.4324(386)")
                .unwrap()
                .round_match_uncertainty_sf(2, RoundingMode::HalfUp),
            SciDecimal::from_str("1.432(39)").unwrap()
        );
        assert_eq!(
            SciDecimal::from_str("1.4324(16)e6")
                .unwrap()
                .round_match_uncertainty_sf(1, RoundingMode::HalfUp),
            SciDecimal::from_str("1.432(2)e6").unwrap()
        );
        // Rounding mode applies to both
        assert_eq!(
            SciDecimal::from_str("1.4025(35)")
                .unwrap()
                .round_match_uncertainty_sf(1, RoundingMode::HalfDown),
            SciDecimal::from_str("1.402(3)").unwrap()
        );
    }

    #[test]
    fn round_uncertainty_precision() {
        assert_eq!(
            sci!(1.4324)
                .with_uncertainty(sci!(0.0016))
                .round_uncertainty_precision(-3, RoundingMode::HalfUp),
            sci!(1.4324).with_uncertainty(sci!(0.002))
        );
        assert_eq!(
            sci!(1.4324)
                .with_uncertainty(sci!(0.0386))
                .round_uncertainty_precision(-3, RoundingMode::HalfUp),
            sci!(1.4324).with_uncertainty(sci!(0.039))
        );
        assert_eq!(
            sci!(1.4324)
                .with_uncertainty(sci!(0.016))
                .round_uncertainty_precision(-1, RoundingMode::HalfUp),
            sci!(1.4324).with_uncertainty(sci!(0.0))
        );
    }

    #[test]
    fn round_uncertainty_dp() {
        assert_eq!(
            sci!(1.4324)
                .with_uncertainty(sci!(0.0386))
                .round_uncertainty_dp(3, RoundingMode::HalfUp),
            sci!(1.4324).with_uncertainty(sci!(0.039))
        );
    }

    #[test]
    fn round_uncertainty_sf() {
        assert_eq!(
            sci!(1.4324)
                .with_uncertainty(sci!(0.0016))
                .round_uncertainty_sf(1, RoundingMode::HalfUp),
            sci!(1.4324).with_uncertainty(sci!(0.002))
        );
        assert_eq!(
            sci!(1.4324)
                .with_uncertainty(sci!(0.0386))
                .round_uncertainty_sf(2, RoundingMode::HalfUp),
            sci!(1.4324).with_uncertainty(sci!(0.039))
        );
        assert_eq!(
            sci!(1.4324)
                .with_uncertainty(sci!(0.016))
                .round_uncertainty_sf(3, RoundingMode::HalfUp),
            sci!(1.4324).with_uncertainty(sci!(0.0160))
        );
    }

    #[test]
    fn round_uncertainty_match_number() {
        // Exact values remain unchanged
        assert_eq!(
            sci!(1.4324).round_uncertainty_match_number(RoundingMode::HalfUp),
            sci!(1.4324)
        );
        // If precisions already match, no rounding occurs
        assert_eq!(
            SciDecimal::from_str("1.4324(6)")
                .unwrap()
                .round_uncertainty_match_number(RoundingMode::HalfUp),
            SciDecimal::from_str("1.4324(6)").unwrap()
        );
        // Uncertainty less precise than number
        assert_eq!(
            sci!(1.3424)
                .with_uncertainty(sci!(0.03))
                .round_uncertainty_match_number(RoundingMode::HalfUp),
            SciDecimal::from_str("1.3424(300)").unwrap()
        );
        // Uncertainty more precise than number
        assert_eq!(
            sci!(1.3424)
                .with_uncertainty(sci!(0.0338724))
                .round_uncertainty_match_number(RoundingMode::HalfUp),
            SciDecimal::from_str("1.3424(339)").unwrap()
        );
    }

    #[test]
    fn add_exact() {
        let n1 = SciDecimal::new(40, 0);
        let n2 = sci!(5.1);
        let result = n1 + n2;
        assert_eq!(result, sci!(45.1));
    }

    #[test]
    fn add_with_uncertainty() {
        let n1 = SciDecimal::new_with_uncertainty(20, 2, 0);
        let n2 = SciDecimal::new_with_uncertainty(30, 5, 0);
        let result = n1 + n2;
        assert_eq!(result.number(), sci!(50));
        //assert_eq!(
        //    Decimal::try_from(result.uncertainty()).unwrap().round_dp(5),
        //    dec!(5.3851648071345).round_dp(5)
        //);
    }

    #[test]
    #[rustfmt::skip]
    fn add_special() {
        let p = sci!(2.5e5);
        let n = sci!(-2.5e5);
        let nan = SciDecimal::NAN;
        let inf = SciDecimal::INFINITY;
        let ninf = SciDecimal::NEG_INFINITY;
        let zero = SciDecimal::ZERO;
        let nzero = SciDecimal::NEG_ZERO;
        // Check positive zero is always created when summing to zero
        assert_eq!( (p      + n     ),  zero);
        assert_eq!( (n      + p     ),  zero);
        // NaN
        assert!(    (nan    + nan   )   .is_nan());
        assert!(    (nan    + p     )   .is_nan());
        assert!(    (nan    + n     )   .is_nan());
        assert!(    (nan    + inf   )   .is_nan());
        assert!(    (nan    + ninf  )   .is_nan());
        assert!(    (nan    + zero  )   .is_nan());
        assert!(    (nan    + nzero )   .is_nan());
        assert!(    (p      + nan   )   .is_nan());
        assert!(    (n      + nan   )   .is_nan());
        assert!(    (inf    + nan   )   .is_nan());
        assert!(    (ninf   + nan   )   .is_nan());
        assert!(    (zero   + nan   )   .is_nan());
        assert!(    (nzero  + nan   )   .is_nan());
        // Infinities
        assert_eq!( (inf    + inf   ),  inf);
        assert_eq!( (ninf   + ninf  ),  ninf);
        assert!(    (inf    + ninf  )   .is_nan());
        assert!(    (ninf   + inf   )   .is_nan());
        assert_eq!( (inf    + p     ),  inf);
        assert_eq!( (inf    + n     ),  inf);
        assert_eq!( (inf    + zero  ),  inf);
        assert_eq!( (inf    + nzero ),  inf);
        assert_eq!( (p      + inf   ),  inf);
        assert_eq!( (n      + inf   ),  inf);
        assert_eq!( (zero   + inf   ),  inf);
        assert_eq!( (nzero  + inf   ),  inf);
        assert_eq!( (ninf   + p     ),  ninf);
        assert_eq!( (ninf   + n     ),  ninf);
        assert_eq!( (ninf   + zero  ),  ninf);
        assert_eq!( (ninf   + nzero ),  ninf);
        assert_eq!( (p      + ninf  ),  ninf);
        assert_eq!( (n      + ninf  ),  ninf);
        assert_eq!( (zero   + ninf  ),  ninf);
        assert_eq!( (nzero  + ninf  ),  ninf);
        // Zeros
        assert_eq!( (zero   + zero  ),  zero);
        assert_eq!( (nzero  + nzero ),  nzero);
        assert_eq!( (zero   + nzero ),  zero);
        assert_eq!( (nzero  + zero  ),  zero);
        assert_eq!( (zero   + p     ),  p);
        assert_eq!( (zero   + n     ),  n);
        assert_eq!( (p      + zero  ),  p);
        assert_eq!( (n      + zero  ),  n);
        assert_eq!( (nzero  + p     ),  p);
        assert_eq!( (nzero  + n     ),  n);
        assert_eq!( (p      + nzero ),  p);
        assert_eq!( (n      + nzero ),  n);
    }

    #[test]
    fn sub_exact() {
        let n1 = SciDecimal::new(20, 0);
        let n2 = SciDecimal::new(30, 0);
        assert_eq!(n1 - n2, sci!(-10));
    }

    #[test]
    fn sub_with_uncertainty() {
        let n1 = SciDecimal::new_with_uncertainty(20, 2, 0);
        let n2 = SciDecimal::new_with_uncertainty(30, 5, 0);
        let result = n1 - n2;
        assert_eq!(result, sci!(-10));
        assert_eq!(
            Decimal::cast_from(result.uncertainty()).round_dp(5),
            dec!(5.3851648071345).round_dp(5)
        );
    }

    #[test]
    #[rustfmt::skip]
    fn sub_special() {
        let p = sci!(2.5e5);
        let n = sci!(-2.5e5);
        let nan = SciDecimal::NAN;
        let inf = SciDecimal::INFINITY;
        let ninf = SciDecimal::NEG_INFINITY;
        let zero = SciDecimal::ZERO;
        let nzero = SciDecimal::NEG_ZERO;
        // Check positive zero is always created when summing to zero
        assert_eq!( (p      - p     ),  zero);
        assert_eq!( (n      - n     ),  zero);
        // NaN
        assert!(    (nan    - nan   )   .is_nan());
        assert!(    (nan    - p     )   .is_nan());
        assert!(    (nan    - n     )   .is_nan());
        assert!(    (nan    - inf   )   .is_nan());
        assert!(    (nan    - ninf  )   .is_nan());
        assert!(    (nan    - zero  )   .is_nan());
        assert!(    (nan    - nzero )   .is_nan());
        assert!(    (p      - nan   )   .is_nan());
        assert!(    (n      - nan   )   .is_nan());
        assert!(    (inf    - nan   )   .is_nan());
        assert!(    (ninf   - nan   )   .is_nan());
        assert!(    (zero   - nan   )   .is_nan());
        assert!(    (nzero  - nan   )   .is_nan());
        // Infinities
        assert!(    (inf    - inf   )   .is_nan());
        assert!(    (ninf   - ninf  )   .is_nan());
        assert_eq!( (inf    - ninf  ),  inf);
        assert_eq!( (ninf   - inf   ),  ninf);
        assert_eq!( (inf    - p     ),  inf);
        assert_eq!( (inf    - n     ),  inf);
        assert_eq!( (inf    - zero  ),  inf);
        assert_eq!( (inf    - nzero ),  inf);
        assert_eq!( (p      - inf   ),  ninf);
        assert_eq!( (n      - inf   ),  ninf);
        assert_eq!( (zero   - inf   ),  ninf);
        assert_eq!( (nzero  - inf   ),  ninf);
        assert_eq!( (ninf   - p     ),  ninf);
        assert_eq!( (ninf   - n     ),  ninf);
        assert_eq!( (ninf   - zero  ),  ninf);
        assert_eq!( (ninf   - nzero ),  ninf);
        assert_eq!( (p      - ninf  ),  inf);
        assert_eq!( (n      - ninf  ),  inf);
        assert_eq!( (zero   - ninf  ),  inf);
        assert_eq!( (nzero  - ninf  ),  inf);
        // Zeros
        assert_eq!( (zero   - zero  ),  zero);
        assert_eq!( (nzero  - nzero ),  zero);
        assert_eq!( (zero   - nzero ),  zero);
        assert_eq!( (nzero  - zero  ),  nzero);
        assert_eq!( (zero   - p     ),  n);
        assert_eq!( (zero   - n     ),  p);
        assert_eq!( (p      - zero  ),  p);
        assert_eq!( (n      - zero  ),  n);
        assert_eq!( (nzero  - p     ),  n);
        assert_eq!( (nzero  - n     ),  p);
        assert_eq!( (p      - nzero ),  p);
        assert_eq!( (n      - nzero ),  n);
    }

    #[test]
    fn mul_exact() {
        let n1 = SciDecimal::new(20, 0);
        let n2 = SciDecimal::new(30, 0);
        assert_eq!(n1 * n2, sci!(600));
        // Large values to confirm rounding behaviour
        // Max significand is 9_999_999_999_999_999
        // First, where the true significand doesn't even fit into a u64
        let n1 = SciDecimal::new(99_999_999_999_999, 0);
        let n2 = SciDecimal::new(89_999_999_999_999, 0);
        // Result = 8999999999999810000000000001 ≈ 8_999_999_999_999_810 * 10^12
        assert_eq!(n1 * n2, SciDecimal::new(8_999_999_999_999_810, 12));
        // Then check where the true significand just exceeds 16 sf but not u64::MAX
        let n1 = SciDecimal::new(999_999_999, 0);
        let n2 = SciDecimal::new(899_999_999, 0);
        // Result = 899_999_998_100_000_001 ≈ 8_999_999_981_000_000 * 10^2
        assert_eq!(n1 * n2, SciDecimal::new(8_999_999_981_000_000, 2));
    }

    #[test]
    fn mul_with_uncertainty() {
        let n1 = SciDecimal::new_with_uncertainty(20, 2, 0);
        let n2 = SciDecimal::new_with_uncertainty(30, 5, 0);
        let result = n1 * n2;
        assert_eq!(result.number(), sci!(600));
        assert_eq!(
            Decimal::cast_from(result.uncertainty()).round_dp(5),
            dec!(116.619037896906).round_dp(5)
        );
        let ft = sci!(0.3048);
        let square_ft = ft * ft;
        assert_eq!(square_ft, sci!(0.09290304));
    }

    #[test]
    #[rustfmt::skip]
    fn mul_special() {
        let p = sci!(2.5e5);
        let n = sci!(-2.5e5);
        let nan = SciDecimal::NAN;
        let inf = SciDecimal::INFINITY;
        let ninf = SciDecimal::NEG_INFINITY;
        let zero = SciDecimal::ZERO;
        let nzero = SciDecimal::NEG_ZERO;
        // NaN
        assert!(    (nan    * nan   )   .is_nan());
        assert!(    (nan    * p     )   .is_nan());
        assert!(    (nan    * n     )   .is_nan());
        assert!(    (nan    * inf   )   .is_nan());
        assert!(    (nan    * ninf  )   .is_nan());
        assert!(    (nan    * zero  )   .is_nan());
        assert!(    (nan    * nzero )   .is_nan());
        assert!(    (p      * nan   )   .is_nan());
        assert!(    (n      * nan   )   .is_nan());
        assert!(    (inf    * nan   )   .is_nan());
        assert!(    (ninf   * nan   )   .is_nan());
        assert!(    (zero   * nan   )   .is_nan());
        assert!(    (nzero  * nan   )   .is_nan());
        // Infinities
        assert_eq!( (inf    * inf   ),  inf);
        assert_eq!( (ninf   * ninf  ),  inf);
        assert_eq!( (inf    * ninf  ),  ninf);
        assert_eq!( (ninf   * inf   ),  ninf);
        assert_eq!( (inf    * p     ),  inf);
        assert_eq!( (inf    * n     ),  ninf);
        assert!(    (inf    * zero  )   .is_nan());
        assert!(    (inf    * nzero )   .is_nan());
        assert_eq!( (p      * inf   ),  inf);
        assert_eq!( (n      * inf   ),  ninf);
        assert!(    (zero   * inf   )   .is_nan());
        assert!(    (nzero  * inf   )   .is_nan());
        assert_eq!( (ninf   * p     ),  ninf);
        assert_eq!( (ninf   * n     ),  inf);
        assert!(    (ninf   * zero  )   .is_nan());
        assert!(    (ninf   * nzero )   .is_nan());
        assert_eq!( (p      * ninf  ),  ninf);
        assert_eq!( (n      * ninf  ),  inf);
        assert!(    (zero   * ninf  )   .is_nan());
        assert!(    (nzero  * ninf  )   .is_nan());
        // Zeros
        assert_eq!( (zero   * zero  ),  zero);
        assert_eq!( (nzero  * nzero ),  zero);
        assert_eq!( (zero   * nzero ),  nzero);
        assert_eq!( (nzero  * zero  ),  nzero);
        assert_eq!( (zero   * p     ),  zero);
        assert_eq!( (zero   * n     ),  nzero);
        assert_eq!( (p      * zero  ),  zero);
        assert_eq!( (n      * zero  ),  nzero);
        assert_eq!( (nzero  * p     ),  nzero);
        assert_eq!( (nzero  * n     ),  zero);
        assert_eq!( (p      * nzero ),  nzero);
        assert_eq!( (n      * nzero ),  zero);
    }

    #[test]
    fn div_exact() {
        // Non-recurring result with same exponent
        assert_eq!(
            SciDecimal::new(60, 0) / SciDecimal::new(30, 0),
            SciDecimal::new(2, 0),
        );
        // Non-recurring result with different exponent
        assert_eq!(
            SciDecimal::new(30, 0) / SciDecimal::new(60, 0),
            SciDecimal::new(5, -1),
        );
        // Identical recurring results with different pairs of starting numbers
        assert_eq!(
            SciDecimal::new(30, 0) / SciDecimal::new(60, 0),
            SciDecimal::new(3, 6) / SciDecimal::new(6, 6),
        );
        // Recurring results to confirm rounding behaviour
        assert_eq!(
            (SciDecimal::new(1, 0) / SciDecimal::new(3, 0)),
            SciDecimal::new(3333333333333333, -16),
        );
        assert_eq!(
            (SciDecimal::new(2, 0) / SciDecimal::new(3, 0)),
            SciDecimal::new(6666666666666667, -16),
        );
        assert_eq!(
            (SciDecimal::new(1, 0) / SciDecimal::new(9, 0)),
            SciDecimal::new(1111111111111111, -16),
        );
    }

    #[test]
    fn div_with_uncertainty() {
        let n1 = SciDecimal::new_with_uncertainty(20, 2, 0);
        let n2 = SciDecimal::new_with_uncertainty(30, 5, 0);
        let result = n1 / n2;
        assert_eq!(
            Decimal::cast_from(result.uncertainty()).round_dp(10),
            dec!(0.6666666667).round_dp(10)
        );
        assert_eq!(
            Decimal::cast_from(result.uncertainty()).round_dp(5),
            dec!(0.129576708774340).round_dp(5)
        );
    }

    #[test]
    fn div_with_uncertainty_reversed() {
        let n1 = SciDecimal::new_with_uncertainty(20, 2, 0);
        let n2 = SciDecimal::new_with_uncertainty(30, 5, 0);
        let result = n2 / n1;
        assert_eq!(result, sci!(1.5));
        assert_eq!(
            Decimal::cast_from(result.uncertainty()).round_dp(5),
            dec!(0.2915475947422).round_dp(5)
        );
    }

    #[test]
    #[rustfmt::skip]
    fn div_special() {
        let p = sci!(2.5e5);
        let n = sci!(-2.5e5);
        let nan = SciDecimal::NAN;
        let inf = SciDecimal::INFINITY;
        let ninf = SciDecimal::NEG_INFINITY;
        let zero = SciDecimal::ZERO;
        let nzero = SciDecimal::NEG_ZERO;
        // NaN
        assert!(    (nan    / nan   )   .is_nan());
        assert!(    (nan    / p     )   .is_nan());
        assert!(    (nan    / n     )   .is_nan());
        assert!(    (nan    / inf   )   .is_nan());
        assert!(    (nan    / ninf  )   .is_nan());
        assert!(    (nan    / zero  )   .is_nan());
        assert!(    (nan    / nzero )   .is_nan());
        assert!(    (p      / nan   )   .is_nan());
        assert!(    (n      / nan   )   .is_nan());
        assert!(    (inf    / nan   )   .is_nan());
        assert!(    (ninf   / nan   )   .is_nan());
        assert!(    (zero   / nan   )   .is_nan());
        assert!(    (nzero  / nan   )   .is_nan());
        // Infinities
        assert!(    (inf    / inf  )    .is_nan());
        assert!(    (ninf   / ninf   )  .is_nan());
        assert!(    (inf    / ninf  )   .is_nan());
        assert!(    (ninf   / inf   )   .is_nan());
        assert_eq!( (inf    / p     ),  inf);
        assert_eq!( (inf    / n     ),  ninf);
        assert_eq!( (inf    / zero  ),  inf);
        assert_eq!( (inf    / nzero ),  ninf);
        assert_eq!( (p      / inf   ),  zero);
        assert_eq!( (n      / inf   ),  nzero);
        assert_eq!( (zero   / inf   ),  zero);
        assert_eq!( (nzero  / inf   ),  nzero);
        assert_eq!( (ninf   / p     ),  ninf);
        assert_eq!( (ninf   / n     ),  inf);
        assert_eq!( (ninf   / zero  ),  ninf);
        assert_eq!( (ninf   / nzero ),  inf);
        assert_eq!( (p      / ninf  ),  nzero);
        assert_eq!( (n      / ninf  ),  zero);
        assert_eq!( (zero   / ninf  ),  nzero);
        assert_eq!( (nzero  / ninf  ),  zero);
        // Zeros
        assert!(    (zero   / zero )    .is_nan());
        assert!(    (nzero  / nzero)    .is_nan());
        assert!(    (zero   / nzero)    .is_nan());
        assert!(    (nzero  / zero )    .is_nan());
        assert_eq!( (zero   / p     ),  zero);
        assert_eq!( (zero   / n     ),  nzero);
        assert_eq!( (p      / zero  ),  zero);
        assert_eq!( (n      / zero  ),  nzero);
        assert_eq!( (nzero  / p     ),  nzero);
        assert_eq!( (nzero  / n     ),  zero);
        assert_eq!( (p      / nzero ),  nzero);
        assert_eq!( (n      / nzero ),  zero);
    }

    #[test]
    #[rustfmt::skip]
    fn rem_special() {
        let p = sci!(2.5e5);
        let n = sci!(-2.5e5);
        let nan = SciDecimal::NAN;
        let inf = SciDecimal::INFINITY;
        let ninf = SciDecimal::NEG_INFINITY;
        let zero = SciDecimal::ZERO;
        let nzero = SciDecimal::NEG_ZERO;
        // Check zero has the sign of the dividend
        assert_eq!( (p      % p     ),  zero);
        assert_eq!( (n      % n     ),  nzero);
        assert_eq!( (p      % n     ),  zero);
        assert_eq!( (n      % p     ),  nzero);
        // NaN
        assert!(    (nan    % nan   )   .is_nan());
        assert!(    (nan    % p     )   .is_nan());
        assert!(    (nan    % n     )   .is_nan());
        assert!(    (nan    % inf   )   .is_nan());
        assert!(    (nan    % ninf  )   .is_nan());
        assert!(    (nan    % zero  )   .is_nan());
        assert!(    (nan    % nzero )   .is_nan());
        assert!(    (p      % nan   )   .is_nan());
        assert!(    (n      % nan   )   .is_nan());
        assert!(    (inf    % nan   )   .is_nan());
        assert!(    (ninf   % nan   )   .is_nan());
        assert!(    (zero   % nan   )   .is_nan());
        assert!(    (nzero  % nan   )   .is_nan());
        // Infinities
        assert!(    (inf    % inf   )   .is_nan());
        assert!(    (ninf   % ninf  )   .is_nan());
        assert!(    (inf    % ninf  )   .is_nan());
        assert!(    (ninf   % inf   )   .is_nan());
        assert!(    (inf    % p     )   .is_nan());
        assert!(    (inf    % n     )   .is_nan());
        assert!(    (inf    % zero  )   .is_nan());
        assert!(    (inf    % nzero )   .is_nan());
        assert_eq!( (p      % inf   ),  p);
        assert_eq!( (n      % inf   ),  n);
        assert_eq!( (zero   % inf   ),  zero);
        assert_eq!( (nzero  % inf   ),  nzero);
        assert!(    (ninf   % p     )   .is_nan());
        assert!(    (ninf   % n     )   .is_nan());
        assert!(    (ninf   % zero  )   .is_nan());
        assert!(    (ninf   % nzero )   .is_nan());
        assert_eq!( (p      % ninf   ), p);
        assert_eq!( (n      % ninf   ), n);
        assert_eq!( (zero   % ninf   ), zero);
        assert_eq!( (nzero  % ninf   ), nzero);
        // Zeros
        assert!(    (zero   % zero )   .is_nan());
        assert!(    (nzero  % nzero)   .is_nan());
        assert!(    (zero   % nzero)   .is_nan());
        assert!(    (nzero  % zero )   .is_nan());
        assert_eq!( (zero   % p     ),  zero);
        assert_eq!( (zero   % n     ),  zero);
        assert!(    (p      % zero  )   .is_nan());
        assert!(    (n      % zero  )   .is_nan());
        assert_eq!( (nzero  % p     ),  nzero);
        assert_eq!( (nzero  % n     ),  nzero);
        assert!(    (p      % nzero  )   .is_nan());
        assert!(    (n      % nzero  )   .is_nan());
    }

    #[test]
    fn powi_exact() {
        let n = SciDecimal::new(4, 0);
        assert_eq!(n.powi(2), sci!(16));
        assert_eq!(n.powi(3), sci!(64));
        assert_eq!(n.powi(-1), sci!(0.25));
        assert_eq!(n.powi(-2), sci!(0.0625));
    }

    #[test]
    fn powi_with_uncertainty() {
        let n = SciDecimal::new_with_uncertainty(20, 2, 0);
        let result = n.powi(2);
        assert_eq!(result.number(), sci!(400));
        // Currently fails, calculates an uncertainty of 8000
        assert_eq!(result.uncertainty(), sci!(80));
    }

    #[test]
    fn inv() {
        assert_eq!(SciDecimal::new(4, 0).inv(), SciDecimal::new(25, -2));
        assert_eq!(SciDecimal::new(5, -1).inv(), SciDecimal::new(2, 0));
    }

    #[test]
    fn neg() {
        let n_pos = SciDecimal::new(4, 0);
        let n_neg = n_pos.neg();
        assert_eq!(n_neg, SciDecimal::new(-4, 0));
        assert!(n_neg.negative);
        assert_eq!(n_neg.significand, 4);
        let n_roundtrip = n_neg.neg();
        assert!(!n_roundtrip.negative);
        assert_eq!(n_roundtrip, n_pos);
    }

    #[test]
    fn natural_log() {
        let n1 = SciDecimal::new_with_uncertainty(20, 2, 0);
        let n2 = SciDecimal::new_with_uncertainty(30, 5, 0);
        let ratio = n1 / n2;
        let result = ratio.ln();
        assert_eq!(
            Decimal::cast_from(result.uncertainty()).round_dp(5),
            dec!(0.194365063161).round_dp(5)
        );
    }

    #[test]
    fn log_base10() {
        let n1 = SciDecimal::new_with_uncertainty(20, 2, 0);
        let n2 = SciDecimal::new_with_uncertainty(30, 5, 0);
        let ratio = n1 / n2;
        let result = ratio.log10();
        assert_eq!(
            Decimal::cast_from(result.uncertainty()).round_dp(5),
            dec!(0.08441167440582).round_dp(5)
        );
    }

    #[test]
    fn exponential() {
        let n1 = SciDecimal::new_with_uncertainty(20, 2, 0);
        let n2 = SciDecimal::new_with_uncertainty(30, 5, 0);
        let ratio = n1 / n2;
        let result = ratio.exp();
        assert_eq!(
            Decimal::cast_from(result.uncertainty()).round_dp(5),
            dec!(0.25238096660761).round_dp(5)
        );
    }

    //#[test]
    //fn debug() {
    //    let n = SciDecimal::new_with_uncertainty(20, 2, 0);
    //    assert_eq!(format!("{n:?}"), "SciDecimal { number: 20, uncertainty: 2 }");
    //}

    #[test]
    fn to_plain_string() {
        assert_eq!(
            SciDecimal::from_str("25e4").unwrap().to_plain_string(),
            "250000"
        );
        assert_eq!(
            SciDecimal::from_str("25(2)e4").unwrap().to_plain_string(),
            "250000(20000)"
        );
    }

    #[test]
    fn display() {
        // NaN and infinity should match the native `f64`
        assert_eq!(SciDecimal::NAN.to_string(), f64::NAN.to_string()); // "NaN"
        assert_eq!(SciDecimal::INFINITY.to_string(), f64::INFINITY.to_string()); // "inf"
        assert_eq!(
            SciDecimal::NEG_INFINITY.to_string(),
            f64::NEG_INFINITY.to_string()
        ); // "-inf"
        // As should +/- zero, as long as there's no uncertainty
        assert_eq!(SciDecimal::ZERO.to_string(), (0.0).to_string()); // "0"
        assert_eq!(SciDecimal::NEG_ZERO.to_string(), (-0.0).to_string()); // "-0"
        // Numbers with up to five places either side of the decimal point should
        // be displayed using normal notation
        // Integers should display without any decimal point at all
        assert_eq!(SciDecimal::new(20, 0).to_string(), "20");
        assert_eq!(SciDecimal::new(-20, 0).to_string(), "-20");
        assert_eq!(SciDecimal::new(99999, 0).to_string(), "99999");
        assert_eq!(SciDecimal::new(10000, 0).to_string(), "10000");
        assert_eq!(SciDecimal::new(1000, 0).to_string(), "1000");
        assert_eq!(SciDecimal::new(100, 0).to_string(), "100");
        assert_eq!(SciDecimal::new(10, 0).to_string(), "10");
        assert_eq!(SciDecimal::new(1, 0).to_string(), "1");
        assert_eq!(SciDecimal::new(1, -1).to_string(), "0.1");
        assert_eq!(SciDecimal::new(1, -2).to_string(), "0.01");
        assert_eq!(SciDecimal::new(1, -3).to_string(), "0.001");
        assert_eq!(SciDecimal::new(1, -4).to_string(), "0.0001");
        assert_eq!(SciDecimal::new(1, -5).to_string(), "0.00001");
        assert_eq!(sci!(0.00001).to_string(), "0.00001");
        assert_eq!(SciDecimal::new(325, -4).to_string(), "0.0325");
        assert_eq!(SciDecimal::new(-325, -4).to_string(), "-0.0325");
        assert_eq!(SciDecimal::new(85130, -3).to_string(), "85.130");
        assert_eq!(sci!(25691.29854).to_string(), "25691.29854");
        // If the maximum number of places (5) is exceeded, use scientific notation
        assert_eq!(SciDecimal::new(1295891, 0).to_string(), "1.295891e6");
        assert_eq!(SciDecimal::new(325, -6).to_string(), "3.25e-4"); // Not 0.000325
        assert_eq!(SciDecimal::new(-325, -6).to_string(), "-3.25e-4");
        assert_eq!(SciDecimal::new(8174036, 0).to_string(), "8.174036e6");
        assert_eq!(sci!(0.000000432).to_string(), "4.32e-7");
        // Importantly, explicit zeros should be treated as significant
        assert_eq!(SciDecimal::new(1295800, 0).to_string(), "1.295800e6");
        // Scientific notation should also be used if there are insignificant zeros
        // before the decimal point, even when the maximum number of places (5)
        // is not exceeded, so that the precision is indicated
        // 81700 with 3 sf = 8.17e4 = (817, 2) => "8.17e4"
        assert_eq!(SciDecimal::new(817, 2).to_string(), "8.17e4");
        // 81700 with 5 sf = 8.1700e4 = (81700, 0) => "81700"
        assert_eq!(SciDecimal::new(81700, 0).to_string(), "81700");

        // Check uncertainty formatting
        assert_eq!(
            SciDecimal::new_with_uncertainty(20, 2, 0).to_string(),
            "20(2)"
        );
        // TODO: More uncertainty display tests
    }

    #[test]
    fn from_str() {
        // Integer
        assert_eq!(SciDecimal::from_str("42").unwrap(), SciDecimal::new(42, 0));
        // Zero
        assert_eq!(SciDecimal::from_str("0").unwrap(), SciDecimal::ZERO);
        // Decimal
        assert_eq!(
            SciDecimal::from_str("0.0859").unwrap(),
            SciDecimal::new(859, -4)
        );
        // Decimal without integral part before decimal point
        assert_eq!(
            SciDecimal::from_str(".0859").unwrap(),
            SciDecimal::new(859, -4)
        );
        // Negative decimal
        assert_eq!(
            SciDecimal::from_str("-3.14").unwrap(),
            SciDecimal::new(-314, -2)
        );
        // Small number but not scientific notation
        assert_eq!(
            SciDecimal::from_str("0.0000000000000000000000000022250738585072").unwrap(),
            SciDecimal::new(22250738585072, -40)
        );
        // Scientific notation
        assert_eq!(
            SciDecimal::from_str("1.5e8").unwrap(),
            SciDecimal::new(15, 7)
        );
        // Scientific notation with negative exponent
        assert_eq!(
            SciDecimal::from_str("2e-5").unwrap(),
            SciDecimal::new(2, -5)
        );
        // Negative number with positive exponent
        assert_eq!(
            SciDecimal::from_str("-6.022e6").unwrap(),
            SciDecimal::new(-6022, 3)
        );
        // Large exponents
        assert_eq!(
            SciDecimal::from_str("1.5e18").unwrap(),
            SciDecimal::new(15, 17)
        );
        assert_eq!(
            SciDecimal::from_str("-6.022e23").unwrap(),
            SciDecimal::new(-6022, 20)
        );
        // Capital E for exponent
        assert_eq!(
            SciDecimal::from_str("1.5E8").unwrap(),
            SciDecimal::new(15, 7)
        );
        // 16 significant figures must always be fine
        assert_eq!(
            SciDecimal::from_str("0.5293040185492948").unwrap(),
            SciDecimal::new(5293040185492948, -16)
        );
        // Excess precision should be silently truncated to 16 sf
        // TODO: maybe in future should be rounded rather than truncated?
        assert_eq!(
            SciDecimal::from_str("0.529304018549294841").unwrap(),
            SciDecimal::new(5293040185492948, -16)
        );
        // Make sure incorrectly formatted strings fail
        assert!(SciDecimal::from_str("not a number").is_err());
        assert!(SciDecimal::from_str("x.482").is_err());
        assert!(SciDecimal::from_str("52.x").is_err());
        assert!(SciDecimal::from_str("-2.42F-4").is_err());
    }

    #[test]
    fn sci_macro() {
        // Integer
        assert_eq!(sci!(42), SciDecimal::new(42, 0));
        // Negative float
        assert_eq!(
            sci!(-3.14),
            SciDecimal::from_scientific_parts(-3, 0, 14, 0, 0)
        );
        // Scientific notation
        assert_eq!(sci!(1.5e8), SciDecimal::new(15, 7));
        // Scientific notation with large exponent
        assert_eq!(sci!(1.5e10), SciDecimal::new(15, 9));
        // Scientific notation with negative exponent
        assert_eq!(sci!(2e-5), SciDecimal::new(2, -5));
        // Negative number with positive exponent
        assert_eq!(sci!(-6.022e6), SciDecimal::new(-6022, 3));
        // Negative number with large exponent
        assert_eq!(sci!(-6.022e23), SciDecimal::new(-6022, 20));
        // Capital E for exponent
        assert_eq!(sci!(1.5E8), SciDecimal::new(15, 7));
    }
}
