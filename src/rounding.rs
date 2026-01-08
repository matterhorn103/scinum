// SPDX-FileCopyrightText: 2026 Matthew Milner <matterhorn103@proton.me>
// SPDX-License-Identifier: MIT OR Apache-2.0

pub enum RoundingMode {
    ToNearestTiesAwayFromZero, // Traditional "rounding up"
    ToNearestTiesTowardZero,   // Traditional "rounding down"
    ToNearestTiesToEven,       // Bankers' rounding
    AwayFromZero,              // Up
    TowardsZero,               // Down
    TowardsPositiveInfinity,   // Ceiling
    TowardsNegativeInfinity,   // Floor
}

impl From<RoundingMode> for rust_decimal::RoundingStrategy {
    fn from(mode: RoundingMode) -> Self {
        match mode {
            RoundingMode::ToNearestTiesAwayFromZero => Self::MidpointAwayFromZero,
            RoundingMode::ToNearestTiesTowardZero => Self::MidpointTowardZero,
            RoundingMode::ToNearestTiesToEven => Self::MidpointNearestEven,
            RoundingMode::AwayFromZero => Self::AwayFromZero,
            RoundingMode::TowardsZero => Self::ToZero,
            RoundingMode::TowardsPositiveInfinity => Self::ToPositiveInfinity,
            RoundingMode::TowardsNegativeInfinity => Self::ToNegativeInfinity,
        }
    }
}

impl From<RoundingMode> for bigdecimal::RoundingMode {
    fn from(mode: RoundingMode) -> Self {
        match mode {
            RoundingMode::ToNearestTiesAwayFromZero => Self::HalfUp,
            RoundingMode::ToNearestTiesTowardZero => Self::HalfDown,
            RoundingMode::ToNearestTiesToEven => Self::HalfEven,
            RoundingMode::AwayFromZero => Self::Up,
            RoundingMode::TowardsZero => Self::Down,
            RoundingMode::TowardsPositiveInfinity => Self::Ceiling,
            RoundingMode::TowardsNegativeInfinity => Self::Floor,
        }
    }
}
