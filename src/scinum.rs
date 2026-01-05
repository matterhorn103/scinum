// SPDX-FileCopyrightText: 2025 Matthew Milner <matterhorn103@proton.me>
// SPDX-License-Identifier: MIT OR Apache-2.0

use num_traits::{Num, Zero};

use crate::SciDecimal;

/// A trait for numeric types that have an associated uncertainty.
pub trait SciNum: Num + TryFrom<SciDecimal> {
    /// The type that is returned by accessing the number or uncertainty.
    type Number: Num + Into<Self>;

    /// Returns the number in exact form without its uncertainty.
    fn number(&self) -> Self::Number;

    /// Returns the absolute uncertainty as an exact number.
    ///
    /// The uncertainty is always positive.
    fn uncertainty(&self) -> Self::Number;

    /// Returns the relative uncertainty as an exact number.
    ///
    /// The relative uncertainty is always positive.
    fn relative_uncertainty(&self) -> Self::Number {
        self.uncertainty() / self.number()
    }

    /// Creates a new number with the same value but the provided uncertainty.
    fn with_uncertainty(self, uncertainty: Self::Number) -> Self;

    /// Returns true if the number has an uncertainty of zero.
    fn is_exact(&self) -> bool {
        self.uncertainty().is_zero()
    }
}
