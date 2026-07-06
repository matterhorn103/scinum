//! Useful functions for the manipulation of integers, for internal use.

use num_integer::Integer;
use num_traits::{PrimInt, Unsigned};

#[allow(dead_code)] // TODO Review
pub(crate) trait UnsignedInt: PrimInt + Integer + Unsigned {
    const ONE: Self;
    const FIVE: Self;
    const TEN: Self;

    fn ilog10(self) -> u32;

    /// Multiply the integer by the given power of ten.
    fn mul_power10(self, exp: u32) -> Self {
        self * Self::TEN.pow(exp)
    }

    /// Divide the integer by the given power of ten by truncation i.e. rounding
    /// towards zero, returning whether the absolute remainder is greater than,
    /// lower than, or equal to half.
    fn div_power10(self, exp: u32) -> (Self, CmpTieResult) {
        let divisor = Self::TEN.pow(exp);
        let (quotient, remainder) = self.div_rem(&divisor);
        (quotient, cmp_tie(remainder))
    }

    /// Divide the integer by the given power of ten while rounding according to
    /// the `RoundingMode::HalfUp` strategy.
    fn div_power10_and_round_half_up(self, exp: u32) -> Self {
        let (quotient, cmp) = self.div_power10(exp);
        match cmp {
            CmpTieResult::Greater => quotient + Self::ONE,
            CmpTieResult::Equal => quotient + Self::ONE,
            CmpTieResult::Less => quotient,
            CmpTieResult::Zero => quotient,
        }
    }

    /// Divide the integer by the given power of ten while rounding according to
    /// the `RoundingMode::HalfEven` strategy.
    fn div_power10_and_round_half_even(self, exp: u32) -> Self {
        let (quotient, cmp) = self.div_power10(exp);
        match cmp {
            CmpTieResult::Greater => quotient + Self::ONE,
            CmpTieResult::Equal => {
                if quotient.is_even() {
                    quotient
                } else {
                    quotient + Self::ONE
                }
            }
            CmpTieResult::Less => quotient,
            CmpTieResult::Zero => quotient,
        }
    }
}

macro_rules! impl_unsigned_int {
    ($T:ty) => {
        impl UnsignedInt for $T {
            const ONE: $T = 1;
            const FIVE: $T = 5;
            const TEN: $T = 10;

            fn ilog10(self) -> u32 {
                self.ilog10()
            }
        }
    };
}

impl_unsigned_int!(u8);
impl_unsigned_int!(u16);
impl_unsigned_int!(u32);
impl_unsigned_int!(u64);
impl_unsigned_int!(u128);

#[allow(dead_code)] // TODO Review
pub(crate) enum CmpTieResult {
    Greater,
    Equal,
    Less,
    Zero,
}

/// Determines if a remainder from rounding is above half, below half, or half.
///
/// For example, when `1.4728` is rounded to two decimal places, the remainder
/// and `digits` would be `28`, and the result would be `Ordering::Less`.
/// For `1.4758` and `digits = 58`, the result would be `Ordering::Greater`.
/// For `1.4750` and `digits = 50`, the result would be `Ordering::Equal`.
///
/// Returns `Ordering::Less` if `digits` is equal to 0.
#[allow(dead_code)] // TODO Review
pub(crate) fn cmp_tie<T: UnsignedInt>(digits: T) -> CmpTieResult {
    if digits.is_zero() {
        return CmpTieResult::Zero;
    }
    let divisor = T::FIVE * T::TEN.pow(digits.ilog10());
    match ((digits / divisor).is_zero(), (digits % divisor).is_zero()) {
        (false, false) => CmpTieResult::Greater,
        (false, true) => CmpTieResult::Equal,
        (true, _) => CmpTieResult::Less,
    }
}
