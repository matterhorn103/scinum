//! Casting methods and trait implementations for [`SciDecimal`].

use std::cmp::Ordering;

use bigdecimal::BigDecimal;
use num_traits::{Float, FromPrimitive, NumCast, ToPrimitive, Zero};
use rust_decimal::Decimal;

use crate::{
    RoundingMode, SciDecimal, SciFloat, SciNum,
    scicast::{CheckedSciCast, SciCast},
};

// No-op casting from integers

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

// Casting to SciDecimal from other types with SciCast

impl SciCast<SciDecimal> for f64 {
    fn cast(self) -> SciDecimal {
        self.to_string()
            .parse()
            .expect("All possible f64 values are representable as a SciDecimal")
    }
}

impl SciCast<SciDecimal> for SciFloat {
    fn cast(self) -> SciDecimal {
        self.number()
            .cast()
            .with_uncertainty(self.uncertainty().cast())
    }
}

impl SciCast<SciDecimal> for BigDecimal {
    // BigDecimal has:
    // - up to 2^63 digits of precision (more than SciDecimal)
    // - i64 exponent and corresponding range (bigger than SciDecimal)
    // - no -0, no inf, no NaN

    fn cast(self) -> SciDecimal {
        self.to_scientific_notation()
            .parse()
            .expect("We can assume BigDecimal formats numbers correctly")
    }
}

impl SciCast<SciDecimal> for Decimal {
    // Decimal has:
    // - up to 28 digits of precision (more than SciDecimal)
    // - an exponent range of 28 (much smaller than SciDecimal)
    // - no -0, no inf, no NaN

    fn cast(self) -> SciDecimal {
        let mut exponent = -(self.scale() as i16);
        let signed_significand = self.mantissa();
        let negative = signed_significand.is_negative();
        let significand = signed_significand.unsigned_abs();
        let narrowed_significand: u64 = match significand.try_into() {
            Ok(narrowed) => narrowed,
            Err(_) => {
                let excess_places = 64 - significand.leading_zeros();
                exponent += excess_places as i16;
                (significand >> excess_places) as u64
            }
        };
        let unrounded = SciDecimal {
            uncertainty: 0,
            uncertainty_scale: 0,
            uncertainty_nan: false,
            uncertainty_inf: false,
            nan: false,
            inf: false,
            negative,
            exponent,
            significand: narrowed_significand,
        };
        if unrounded.sf() > 16 {
            unrounded.round_sf(16, RoundingMode::HalfEven)
        } else {
            unrounded
        }
    }
}

// Casting to other types from SciDecimal with SciCast

impl SciCast<f64> for SciDecimal {
    fn cast(self) -> f64 {
        if self.nan {
            f64::NAN
        } else if self.inf {
            if self.negative {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            }
        } else if self > f64::MAX.cast() {
            f64::INFINITY
        } else if self < f64::MIN.cast() {
            f64::NEG_INFINITY
        } else if self.abs() < f64::MIN_POSITIVE.cast() {
            if self.negative { -0.0 } else { 0.0 }
        } else {
            // Otherwise, must be able to fit, if we just drop excess precision
            // Don't waste time adding trailing zeros if we don't have to
            let narrowed = if self.sf() > 15 {
                self.round_sf(15, RoundingMode::HalfEven)
            } else {
                self
            };
            narrowed
                .to_string()
                .parse()
                .expect("All other possible values should fit into an f64")
        }
    }
}

impl CheckedSciCast<BigDecimal> for SciDecimal {
    // Can't implement SciCast at all because the max/min values of BigDecimal
    // that +inf/-inf would have to cast to are ridiculous
    //
    // BigDecimal has:
    // - up to 2^63 digits of precision (more than SciDecimal)
    // - i64 exponent and corresponding range (bigger than SciDecimal)
    // - no -0, no inf, no NaN

    fn checked_cast(self) -> Option<BigDecimal> {
        if self.nan {
            None
        } else if self.inf {
            None
        } else {
            Some(
                self.to_string()
                    .parse()
                    .expect("All other values should be representable by BigDecimal"),
            )
        }
    }
}

impl SciCast<Decimal> for SciDecimal {
    // Decimal has:
    // - up to 28 digits of precision (more than SciDecimal)
    // - an exponent range of 28 (much smaller than SciDecimal)
    // - no -0, no inf, no NaN

    fn cast(self) -> Decimal {
        if self.nan {
            Decimal::ZERO
        } else if self > Decimal::MAX.cast() {
            // Includes inf
            // Saturate to highest possible value
            Decimal::MAX
        } else if self < Decimal::MIN.cast() {
            // Includes -inf
            // Saturate to lowest possible value
            Decimal::MIN
        } else {
            Decimal::from_scientific_lossy(&self.number().to_scientific_string())
                .expect("All other values should be representable by rust_decimal::Decimal")
        }
    }
}

impl CheckedSciCast<Decimal> for SciDecimal {
    fn checked_cast(self) -> Option<Decimal> {
        if self.nan {
            None
        } else if self > Decimal::MAX.cast() {
            // Includes inf
            None
        } else if self < Decimal::MIN.cast() {
            // Includes -inf
            None
        } else {
            Some(
                Decimal::from_scientific_lossy(&self.number().to_scientific_string())
                    .expect("All other values should be representable by rust_decimal::Decimal"),
            )
        }
    }
}

// num_traits casting traits

impl FromPrimitive for SciDecimal {
    #[inline]
    fn from_i64(n: i64) -> Option<Self> {
        if n > Self::MAX_SIGNIFICAND_SIGNED {
            None
        } else {
            Some(Self::new(n, 0))
        }
    }

    #[inline]
    fn from_u64(n: u64) -> Option<Self> {
        if n > Self::MAX_SIGNIFICAND {
            None
        } else {
            Some(Self::new(n as i64, 0))
        }
    }

    fn from_f64(n: f64) -> Option<Self> {
        Some(n.cast())
    }
}

impl ToPrimitive for SciDecimal {
    fn to_i64(&self) -> Option<i64> {
        if self.is_infinite() {
            return None;
        }
        match self.precision().cmp(&0) {
            Ordering::Less => {
                // Significand is guaranteed to not be larger than 10^16 - 1 and
                // therefore so is the resulting number
                Some(
                    self.round_precision(0, RoundingMode::HalfUp)
                        .significand_signed(),
                )
            }
            Ordering::Equal => Some(self.significand_signed()),
            Ordering::Greater => self
                .significand_signed()
                .checked_mul(10_i64.pow(self.exponent() as u32)),
        }
    }

    #[inline]
    fn to_u64(&self) -> Option<u64> {
        if self.is_sign_negative() {
            if self.is_zero() {
                return Some(0);
            } else {
                return None;
            }
        }
        self.to_i64().map(|n| n as u64)
    }
}

impl NumCast for SciDecimal {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        if let Some(f) = n.to_f64() {
            Self::from_f64(f)
        } else if let Some(i) = n.to_i64() {
            Self::from_i64(i)
        } else {
            Self::from_u64(n.to_u64()?)
        }
    }
}
/*
#[cfg(test)]
mod tests {
    use rust_decimal_macros::dec;

    use crate::sci;
    use crate::scicast::{CheckedSciCastFrom, SciCastFrom};

    use super::*;

    #[test]
    fn cast_f64() {
        assert_eq!(f64::cast_from(sci!(2.5e5)), 2.5e5_f64);
        assert_eq!(f64::cast_from(sci!(-2.5e5)), -2.5e5_f64);
        assert_eq!(f64::cast_from(SciDecimal::ZERO), 0_f64);
        assert_eq!(f64::cast_from(SciDecimal::NEG_ZERO), -0_f64);
        assert_eq!(f64::cast_from(SciDecimal::INFINITY), f64::INFINITY);
        assert_eq!(f64::cast_from(SciDecimal::NEG_INFINITY), f64::NEG_INFINITY);
        assert!(f64::cast_from(SciDecimal::NAN).is_nan());
    }

    #[test]
    fn cast_from_f64() {
        assert_eq!(SciDecimal::cast_from(2.5e5_f64), sci!(2.5e5));
        assert_eq!(
            SciDecimal::cast_from(f64::MIN_POSITIVE),
            sci!(2.225073858507201e-308)
        )
    }

    #[test]
    fn cast_decimal() {
        let n1 = SciDecimal::new_with_uncertainty(20, 2, 0);
        assert_eq!(Decimal::cast_from(n1), dec!(20));
        let n2 = sci!(2.5e5);
        assert_eq!(Decimal::cast_from(n2), dec!(2.5e5));
    }

    #[test]
    fn checked_cast_decimal_fails() {
        let n = SciDecimal::new_with_uncertainty(20, 2, 40);
        assert!(Decimal::checked_cast_from(n).is_none());
    }

    #[test]
    fn cast_from_decimal() {
        let n = sci!(20);
        assert_eq!(n.number(), SciDecimal::new(20, 0));
        assert_eq!(n.number(), sci!(20));
        assert_eq!(n.uncertainty(), SciDecimal::new(0, 0));
        assert_eq!(n.uncertainty(), SciDecimal::ZERO);
    }
}
*/
