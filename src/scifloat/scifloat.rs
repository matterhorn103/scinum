use num_traits::{Float, One, Zero};

use crate::{RoundingMode, SciNum};

/// A binary floating point number with an associated uncertainty.
///
/// Wraps the native `f64` type.
#[derive(Debug, Clone, Copy, serde_with::DeserializeFromStr, serde_with::SerializeDisplay)]
pub struct SciFloat {
    pub(crate) number: f64,
    pub(crate) uncertainty: f64,
}

/// Associated constructor functions.
impl SciFloat {
    pub const fn new(number: f64) -> Self {
        Self {
            number,
            uncertainty: 0.0,
        }
    }

    pub const fn new_with_uncertainty(number: f64, uncertainty: f64) -> Self {
        Self {
            number,
            uncertainty,
        }
    }
}

/// Identity-related constants.
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

    /// The `SciFloat` representation of (positive) zero.
    pub const ZERO: Self = SciFloat {
        number: 0.0,
        uncertainty: 0.0,
    };

    /// The `SciFloat` representation of negative zero.
    pub const NEG_ZERO: SciFloat = SciFloat {
        number: -0.0,
        uncertainty: 0.0,
    };

    /// The `SciFloat` representation of one.
    pub const ONE: Self = SciFloat {
        number: 1.0,
        uncertainty: 0.0,
    };

    /// The `SciFloat` representation of minus one.
    pub const NEG_ONE: Self = SciFloat {
        number: 1.0,
        uncertainty: 0.0,
    };
}

#[allow(unused_variables)] // TODO Remove once all methods are implemented
impl SciNum for SciFloat {
    type Number = f64;

    const ZERO: SciFloat = SciFloat::ZERO;

    const ONE: SciFloat = SciFloat::ONE;

    /// Returns the number as an `f64`.
    #[inline]
    fn number(&self) -> f64 {
        self.number
    }

    /// Returns the absolute uncertainty as an `f64`.
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
    ///
    /// Note that the uncertainty itself may well be ∞ or `NaN` as a result of
    /// arithmetic operations.
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

    fn precision(&self) -> i16 {
        todo!()
    }

    fn precision_most_significant_fig(&self) -> i16 {
        todo!()
    }

    fn precision_uncertainty(&self) -> Option<i16> {
        todo!()
    }

    fn dp(&self) -> u16 {
        todo!()
    }

    fn sf(&self) -> u8 {
        todo!()
    }

    fn sf_uncertainty(&self) -> u8 {
        todo!()
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

    fn trunc_sf(mut self, sf: u8) -> Self {
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
}
