// SPDX-FileCopyrightText: 2026 Matthew Milner <matterhorn103@proton.me>
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::cmp::Ordering;

use num_traits::Zero;

#[derive(Debug, Clone, Copy)]
pub enum RoundingMode {
    HalfUp,     // Traditional "rounding up"
    HalfDown,   // Traditional "rounding down"
    HalfEven,   // Bankers' rounding
    Up,         // Away from zero
    Down,       // Towards zero
    Ceiling,    // Towards positive infinity
    Floor,      // Towards negative infinity
}

impl From<RoundingMode> for rust_decimal::RoundingStrategy {
    fn from(mode: RoundingMode) -> Self {
        match mode {
            RoundingMode::HalfUp => Self::MidpointAwayFromZero,
            RoundingMode::HalfDown => Self::MidpointTowardZero,
            RoundingMode::HalfEven => Self::MidpointNearestEven,
            RoundingMode::Up => Self::AwayFromZero,
            RoundingMode::Down => Self::ToZero,
            RoundingMode::Ceiling => Self::ToPositiveInfinity,
            RoundingMode::Floor => Self::ToNegativeInfinity,
        }
    }
}

impl From<RoundingMode> for bigdecimal::RoundingMode {
    fn from(mode: RoundingMode) -> Self {
        match mode {
            RoundingMode::HalfUp => Self::HalfUp,
            RoundingMode::HalfDown => Self::HalfDown,
            RoundingMode::HalfEven => Self::HalfEven,
            RoundingMode::Up => Self::Up,
            RoundingMode::Down => Self::Down,
            RoundingMode::Ceiling => Self::Ceiling,
            RoundingMode::Floor => Self::Floor,
        }
    }
}

/// Determines if a remainder from rounding is above half, below half, or half.
/// 
/// For example, when `1.4728` is rounded to two decimal places, the remainder
/// and `digits` would be `28`, and the result would be `Ordering::Less`.
/// For `1.4758` and `digits = 58`, the result would be `Ordering::Greater`.
/// For `1.4750` and `digits = 50`, the result would be `Ordering::Equal`.
/// 
/// # Panics
/// 
/// This function panics if `digits` is `0`.
pub(crate) fn cmp_tie(digits: u64) -> Ordering {
    let divisor = 5_u64 * 10_u64.pow(digits.ilog10());
    match ((digits / divisor).is_zero(), (digits % divisor).is_zero()) {
        (false, false) => Ordering::Greater,
        (false, true) => Ordering::Equal,
        (true, _) => Ordering::Less,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cmp_tie() {
        assert_eq!(cmp_tie(28), Ordering::Less);
        assert_eq!(cmp_tie(58), Ordering::Greater);
        assert_eq!(cmp_tie(50), Ordering::Equal);
        assert_eq!(cmp_tie(4829), Ordering::Less);
        assert_eq!(cmp_tie(6291), Ordering::Greater);
        assert_eq!(cmp_tie(5004), Ordering::Greater);
        assert_eq!(cmp_tie(5), Ordering::Equal);
        assert_eq!(cmp_tie(5000), Ordering::Equal);
    }
}
