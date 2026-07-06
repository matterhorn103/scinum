use std::{cmp::Ordering, ops::Neg};

use num_traits::{Float, One, Zero};

use crate::{RoundingMode, SciNum, rounding::cmp_tie};

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
    pub(crate) uncertainty: u32,
    pub(crate) uncertainty_scale: i8, // This allows the uncertainty to have a different precision
    pub(crate) uncertainty_nan: bool,
    pub(crate) uncertainty_inf: bool,
    pub(crate) nan: bool,
    pub(crate) inf: bool,
    pub(crate) negative: bool,
    pub(crate) exponent: i16,
    pub(crate) significand: u64,
}

/// Identity-related constants.
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
        uncertainty_nan: false,
        uncertainty_inf: false,
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
        uncertainty_nan: false,
        uncertainty_inf: false,
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
        uncertainty_nan: false,
        uncertainty_inf: false,
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
        uncertainty_nan: false,
        uncertainty_inf: false,
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
        uncertainty_nan: false,
        uncertainty_inf: false,
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
        uncertainty_nan: false,
        uncertainty_inf: false,
        nan: false,
        inf: true,
        negative: true,
        exponent: 0,
        significand: 0,
    };

    /// The `SciDecimal` representation of (positive) zero.
    pub const ZERO: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        uncertainty_nan: false,
        uncertainty_inf: false,
        nan: false,
        inf: false,
        negative: false,
        exponent: 0,
        significand: 0,
    };

    /// The `SciDecimal` representation of negative zero.
    pub const NEG_ZERO: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        uncertainty_nan: false,
        uncertainty_inf: false,
        nan: false,
        inf: false,
        negative: true,
        exponent: 0,
        significand: 0,
    };
}

/// Associated constructor functions.
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
            uncertainty_nan: false,
            uncertainty_inf: false,
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
            uncertainty_nan: false,
            uncertainty_inf: false,
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
            uncertainty_nan: false,
            uncertainty_inf: false,
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
    pub(crate) fn increase_precision_unbounded(mut self, sf: u8) -> Self {
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
    pub(crate) fn increase_precision_unbounded_checked(mut self, sf: u8) -> Option<Self> {
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

impl SciNum for SciDecimal {
    type Number = SciDecimal;

    const ZERO: SciDecimal = SciDecimal::ZERO;

    const ONE: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        uncertainty_nan: false,
        uncertainty_inf: false,
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
            uncertainty_nan: false,
            uncertainty_inf: false,
            ..*self
        }
    }

    /// Returns the absolute uncertainty as an exact `SciDecimal`.
    ///
    /// The uncertainty is always positive.
    ///
    /// # Special values
    ///
    /// - ±0 → the actual uncertainty (0 ± 3 is perfectly valid, for example)
    ///
    /// - ±∞ → ∞
    ///
    /// - `NaN` → `NaN`
    #[inline]
    fn uncertainty(&self) -> Self {
        if self.nan {
            Self::NAN
        } else if self.inf {
            // TODO Should the uncertainty of ∞ be ∞ or NaN?
            Self::INFINITY
        } else if self.uncertainty_nan {
            Self::NAN
        } else if self.uncertainty_inf {
            Self::INFINITY
        } else {
            Self {
                uncertainty: 0,
                uncertainty_scale: 0,
                uncertainty_nan: false,
                uncertainty_inf: false,
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
    ///
    /// # Special values
    ///
    /// - ±0 → ∞
    ///
    /// - ±∞ → `NaN`
    ///
    /// - `NaN` → `NaN`
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
        if uncertainty.is_nan() {
            self.uncertainty_nan = true;
        } else if uncertainty.is_infinite() {
            self.uncertainty_inf = true;
        } else {
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
        }
        self
    }

    /// Returns `true` if the `SciDecimal` has an uncertainty of zero.
    ///
    /// # Special values
    ///
    /// - ±0 → `true` or `false` according to the actual uncertainty
    ///
    /// - ±∞ → `false`
    ///
    /// - `NaN` → `false`
    #[inline]
    fn is_exact(&self) -> bool {
        // We could just do self.uncertainty().is_zero() but faster if we avoid
        // creating a new SciDecimal
        if self.nan | self.inf {
            false
        } else {
            self.uncertainty == 0
        }
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
        if !self.is_finite() {
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

/// Additional constants.
impl SciDecimal {
    // TODO Add more of the constants that f64 has https://doc.rust-lang.org/std/f64/consts/index.html

    pub const TWO: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        uncertainty_nan: false,
        uncertainty_inf: false,
        nan: false,
        inf: false,
        negative: false,
        exponent: 0,
        significand: 2,
    };

    pub const NEG_ONE: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        uncertainty_nan: false,
        uncertainty_inf: false,
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
        uncertainty_nan: false,
        uncertainty_inf: false,
        nan: false,
        inf: false,
        negative: true,
        exponent: -15,
        significand: 3_141_592_653_589_793,
    };

    /// The mathematical constant *π*, with 19 sf for internal use.
    pub(crate) const PI_PRECISE: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        uncertainty_nan: false,
        uncertainty_inf: false,
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
        uncertainty_nan: false,
        uncertainty_inf: false,
        nan: false,
        inf: false,
        negative: true,
        exponent: -15,
        significand: 2_718_281_828_459_045,
    };

    /// The mathematical constant *e*, with 19 sf for internal use.
    pub(crate) const E_PRECISE: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        uncertainty_nan: false,
        uncertainty_inf: false,
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
        uncertainty_nan: false,
        uncertainty_inf: false,
        nan: false,
        inf: false,
        negative: true,
        exponent: -16,
        significand: 693_147_180_559_945_3,
    };

    /// The natural logarithm of 2, ln(2) = logₑ(2), with 19 sf for internal use.
    pub(crate) const LN_2_PRECISE: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        uncertainty_nan: false,
        uncertainty_inf: false,
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
        uncertainty_nan: false,
        uncertainty_inf: false,
        nan: false,
        inf: false,
        negative: true,
        exponent: -15,
        significand: 2_302_585_092_994_046,
    };

    /// The natural logarithm of 10, ln(10) = logₑ(10), with 19 sf for internal use.
    pub(crate) const LN_10_PRECISE: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        uncertainty_nan: false,
        uncertainty_inf: false,
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
        uncertainty_nan: false,
        uncertainty_inf: false,
        nan: false,
        inf: false,
        negative: true,
        exponent: -15,
        significand: 1_442_695_040_888_963,
    };

    /// The base-2 logarithm of *e*, log₂(*e*), with 19 sf for internal use.
    pub(crate) const LOG2_E_PRECISE: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        uncertainty_nan: false,
        uncertainty_inf: false,
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
        uncertainty_nan: false,
        uncertainty_inf: false,
        nan: false,
        inf: false,
        negative: true,
        exponent: -15,
        significand: 3_321_928_094_887_362,
    };

    /// The base-2 logarithm of 10, log₂(10), with 19 sf for internal use.
    pub(crate) const LOG2_10_PRECISE: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        uncertainty_nan: false,
        uncertainty_inf: false,
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
        uncertainty_nan: false,
        uncertainty_inf: false,
        nan: false,
        inf: false,
        negative: true,
        exponent: -16,
        significand: 301_029_995_663_981_2,
    };

    /// The base-10 logarithm of 2, log₁₀(2), with 19 sf for internal use.
    pub(crate) const LOG10_2_PRECISE: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        uncertainty_nan: false,
        uncertainty_inf: false,
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
        uncertainty_nan: false,
        uncertainty_inf: false,
        nan: false,
        inf: false,
        negative: true,
        exponent: -16,
        significand: 434_294_481_903_251_8,
    };

    /// The base-10 logarithm of *e*, log₁₀(*e*), with 19 sf for internal use.
    pub(crate) const LOG10_E_PRECISE: SciDecimal = SciDecimal {
        uncertainty: 0,
        uncertainty_scale: 0,
        uncertainty_nan: false,
        uncertainty_inf: false,
        nan: false,
        inf: false,
        negative: true,
        exponent: -19,
        significand: 434_294_481_903_251_827_7,
    };
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use crate::sci;

    use super::*;

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
                uncertainty_nan: false,
                uncertainty_inf: false,
                nan: true,
                inf: true,
                negative: false,
                exponent: 1,
                significand: 0,
            },
            SciDecimal {
                uncertainty: 3,
                uncertainty_scale: -1,
                uncertainty_nan: false,
                uncertainty_inf: false,
                nan: true,
                inf: true,
                negative: true,
                exponent: -4,
                significand: 25,
            },
            SciDecimal {
                uncertainty: 373,
                uncertainty_scale: 2,
                uncertainty_nan: false,
                uncertainty_inf: false,
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
}
