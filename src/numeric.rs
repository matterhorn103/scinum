// SPDX-FileCopyrightText: 2025 Matthew Milner <matterhorn103@proton.me>
// SPDX-License-Identifier: MIT

use num_traits::Num;

/// A trait for numeric types that have an associated uncertainty.
pub trait SciNumeric: Num {
    /// The type that is returned by accessing the number or uncertainty
    type Numeric: Num;

    /// Returns the number in exact form without its uncertainty.
    fn number(&self) -> Self::Numeric;

    /// Returns the absolute uncertainty as an exact number.
    ///
    /// The uncertainty is always positive.
    fn uncertainty(&self) -> Self::Numeric;
}
