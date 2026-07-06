use std::cmp::Ordering;

use bigdecimal::BigDecimal;
use num_traits::{Float, FromPrimitive, NumCast, ToPrimitive, Zero};
use rust_decimal::Decimal;

use crate::{RoundingMode, SciDecimal, SciFloat, SciNum};

/// A trait for the implementation of infallible but potentially lossy conversions.
///
/// These casts follow the semantics of [numeric casting with the `as` keyword](https://doc.rust-lang.org/reference/expressions/operator-expr.html#numeric-cast):
///
/// - Casting never fails or panics.
///
/// - The output value is the closest possible to the input value representable
///   by the target type.
///
/// - If both types support uncertainties, any uncertainty is preserved; otherwise
///   any uncertainty is dropped and the output is always exact.
///
/// - Loss of precision is acceptable, with rounding as follows:
///
///   - Casting from a fractional type (including floating and fixed point types)
///   to an integer type rounds towards zero i.e. [`scinum::RoundingMode::Down`].
///
///   - Casting to a fractional type, whether from another fractional type or from
///   an integer type, rounds to nearest with ties to even i.e. [`scinum::RoundingMode::HalfEven`].
///
/// - Saturation is acceptable, as follows:
///
///   - A value too small (i.e. too close to zero) to be represented by the
///     target type returns `0` with the appropriate sign.
///
///   - A value too large to be represented by the target type returns the saturated
///     maximum value of the type, or infinity if the type has an infinity.
///
///   - A value too low (i.e. too negative) to be represented by the target type
///     returns the saturated minimum value of the type, or negative infinity if
///     the type has it.
///
/// - Special float values are preserved if the target type also supports them;
///   if not, alternatives are used as follows:
///
///   - `-0` becomes `0` if zero is unsigned in the target type
///
///   - `inf` becomes the saturated maximum value of the target type
///
///   - `-inf` becomes the saturated minimum value of the target type, which will
///     be `0` for types with no negative numbers
///
///   - `NaN` becomes `0`
///
/// WARNING: If the target type cannot represent non-finite numbers (`inf` and/or
/// `NaN`) then `cast()` converts them to finite numbers, with the potentially
/// misleading implication that an arithmetic result was finite. This could cause
/// very real problems in some situations, so care should be exercised. If such
/// conversions would be problematic, use [`CheckedSciCast::checked_cast`], which
/// returns `None` if a non-finite value would become finite.
///
/// As such, `CheckedSciCast` should also be implemented for types that cannot
/// represent `inf` and/or `NaN`.
///
/// # Comparison to similar traits
///
/// Given the rules above, `SciCast` serves a different role to `From`, `TryFrom`,
/// and `ToPrimitive`/`NumCast`, which each make a different set of promises:
///
/// - `from()` and `into()` in `std` are also infallible must be lossless.
///
/// - `try_from()` and `try_into()` in `std` are fallible, but must also be
///   lossless i.e. a lossy conversion must fail.
///
/// - The methods of `ToPrimitive` and `FromPrimitive` and thus also `NumCast::from()`
///   *may* be lossy, in the sense that loss of precision is acceptable, but they
///   are not infallible conversions.
///
/// In summary:
///
/// | Conversion                    | Return type   | Infallible?   | Rounding?     | Saturating?   | Non-finite -> finite? |
/// | ----------------------------- | ------------- | ------------- | ------------- | ------------- | --------------------- |
/// | Numeric casting with `as`     | `T`           | Yes           | Yes           | Yes           | Yes                   |
/// | `std::convert::From`          | `T`           | Yes           | No            | No            | No                    |
/// | `std::convert::TryFrom`       | `Result<T>`   | No            | No            | No            | No                    |
/// | `num::{To, From}Primitive`    | `Option<T>`   | No            | Yes           | Yes           | ?                     |
/// | `SciCast`                     | `T`           | Yes           | Yes           | Yes           | Sometimes†            |
/// | `CheckedSciCast`              | `Option<T>`   | No            | Yes           | Sometimes‡    | No                    |
///
/// † Only if the target type cannot represent the non-finite value (i.e. it
/// does not support infinities or `NaN` or both)
///
/// ‡ Only to `inf` or `0`
pub trait SciCast<T> {
    /// Converts `self` to a `T` infallibly, but potentially lossily, dropping
    /// any uncertainty, rounding and saturating as necessary.
    fn cast(self) -> T;
}

/// A trait for the implementation of infallible but potentially lossy conversions
/// in a fashion similar to [`SciCast`] but without saturating behaviour and no
/// silent conversion of infinity or `NaN` to finite values.
///
/// The conversion by [`SciCast::cast()`] of `inf` and `NaN` to finite numbers,
/// and the potentially misleading implication that an arithmetic result was
/// finite, could cause very real problems in some situations. This trait helps
/// with this by providing a `checked_cast()` method that returns `None` if a
/// non-finite value would become finite.
///
/// These casts are made according to the following rules:
///
/// - Casting never panics.
///
/// - Casting only fails if saturation would occur or a non-finite value would
///   become finite, in which case `None` is returned.
///
/// - The output value is the closest possible to the input value representable
///   by the target type.
///
/// - If both types support uncertainties, any uncertainty is preserved; otherwise
///   any uncertainty is dropped and the output is always exact.
///
/// - Loss of precision *is* acceptable, with rounding as follows:
///
///   - Casting from a fractional type (including floating and fixed point types)
///   to an integer type rounds towards zero i.e. [`scinum::RoundingMode::Down`].
///
///   - Casting to a fractional type, whether from another fractional type or from
///   an integer type, rounds to nearest with ties to even i.e. [`scinum::RoundingMode::HalfEven`].
///
/// - Saturation is *not* acceptable:
///
///   - A value too small (i.e. too close to zero) to be represented by the
///     target type returns `None`.
///
///   - A value too large to be represented by the target type returns `None`
///
///   - A value too low (i.e. too negative) to be represented by the target type
///     returns `None`.
///
/// - Special float values are preserved if the target type also supports them;
///   otherwise:
///
///   - `-0` becomes `0` if zero is unsigned in the target type
///
///   - `inf`, `-inf` and `NaN` return `None`
///
/// # Comparison to similar traits
///
/// Given the rules above, `SciCast` serves a different role to `From`, `TryFrom`,
/// and `ToPrimitive`/`NumCast`, which each make a different set of promises:
///
/// - `from()` and `into()` in `std` are also infallible must be lossless.
///
/// - `try_from()` and `try_into()` in `std` are fallible, but must also be
///   lossless i.e. a lossy conversion must fail.
///
/// - The methods of `ToPrimitive` and `FromPrimitive` and thus also `NumCast::from()`
///   *may* be lossy, in the sense that loss of precision is acceptable, but they
///   are not infallible conversions.
///
/// In summary:
///
/// | Conversion                    | Return type   | Infallible?   | Rounding?     | Saturating?   | Non-finite -> finite? |
/// | ----------------------------- | ------------- | ------------- | ------------- | ------------- | --------------------- |
/// | Numeric casting with `as`     | `T`           | Yes           | Yes           | Yes           | Yes                   |
/// | `std::convert::From`          | `T`           | Yes           | No            | No            | No                    |
/// | `std::convert::TryFrom`       | `Result<T>`   | No            | No            | No            | No                    |
/// | `num::{To, From}Primitive`    | `Option<T>`   | No            | Yes           | Yes           | ?                     |
/// | `SciCast`                     | `T`           | Yes           | Yes           | Yes           | Sometimes†            |
/// | `CheckedSciCast`              | `Option<T>`   | No            | Yes           | Sometimes‡    | No                    |
///
/// † Only if the target type cannot represent the non-finite value (i.e. it
/// does not support infinities or `NaN` or both)
///
/// ‡ Only to `inf` or `0`
pub trait CheckedSciCast<T> {
    /// Converts `self` to a `T` similarly to [`SciCast::cast`], dropping
    /// uncertainty and rounding as necessary, but with saturating behaviour
    /// only in limited cases and no silent conversion of infinity or `NaN` to
    /// finite values.
    fn checked_cast(self) -> Option<T>;
}

/// A companion trait to [`SciCast`] to form a bidirectional pair analogous to
/// `From` and `Into`.
pub trait SciCastFrom<N>: Sized
where
    N: SciCast<Self>,
{
    fn cast_from(n: N) -> Self {
        n.cast()
    }
}

/// A companion trait to [`CheckedSciCast`] to form a bidirectional pair analogous to
/// `TryFrom` and `TryInto`.
pub trait CheckedSciCastFrom<N>: Sized
where
    N: CheckedSciCast<Self>,
{
    fn checked_cast_from(n: N) -> Option<Self> {
        n.checked_cast()
    }
}

// Blanket implementations so that something that is a target for a cast method
// gets a method automatically implemented to cast it from the source type.
// Note that while this is the same idea as with from/into, it works the opposite
// way round: the source type should implement `SciCast` and the target type gets
// `SciCastFrom` for free, whereas with from/into, it's the target type that
// should implement `From` while the source type gets `Into` for free.
impl<T, N> SciCastFrom<N> for T where N: SciCast<T> {}

impl<T, N> CheckedSciCastFrom<N> for T where N: CheckedSciCast<T> {}

// Casting to SciDecimal from other types

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

// Casting to other types from SciDecimal

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

impl FromPrimitive for SciFloat {
    fn from_i64(n: i64) -> Option<Self> {
        f64::from_i64(n).map(|f| f.into())
    }

    fn from_u64(n: u64) -> Option<Self> {
        f64::from_u64(n).map(|f| f.into())
    }

    fn from_f64(n: f64) -> Option<Self> {
        Some(Self::new(n))
    }
}

impl ToPrimitive for SciFloat {
    fn to_i64(&self) -> Option<i64> {
        self.number().to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.number().to_u64()
    }

    fn to_f64(&self) -> Option<f64> {
        Some(self.number())
    }
}

impl NumCast for SciFloat {
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

#[cfg(test)]
mod tests {
    use rust_decimal_macros::dec;

    use crate::sci;

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
