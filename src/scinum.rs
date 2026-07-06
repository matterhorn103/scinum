use num_traits::{Num, Zero};

use crate::RoundingMode;

/// A trait for numeric types that have an associated uncertainty.
pub trait SciNum: Num {
    /// The type that is returned by accessing the number or uncertainty.
    ///
    /// Need not be `Self`, but must always be fully compatible with `Self`.
    type Number: Num + Into<Self>;

    /// The type's representation of the value 0, in exact form.
    const ZERO: Self;

    /// The type's representation of the value 1, in exact form.
    const ONE: Self;

    /// Returns the number in exact form without its uncertainty.
    fn number(&self) -> Self::Number;

    /// Returns the absolute uncertainty as an exact number.
    ///
    /// The uncertainty is always positive.
    fn uncertainty(&self) -> Self::Number;

    /// Returns the relative uncertainty as an exact number.
    ///
    /// The relative uncertainty is always positive.
    fn relative_uncertainty(&self) -> Self::Number;

    /// Creates a new number with the same value but the provided uncertainty.
    fn with_uncertainty(self, uncertainty: Self::Number) -> Self;

    /// Returns `true` if the number has an uncertainty of zero.
    fn is_exact(&self) -> bool {
        self.uncertainty().is_zero()
    }

    /// Returns the scale of the least significant place.
    fn precision(&self) -> i16;

    /// Returns the scale of the most significant place.
    fn precision_most_significant_fig(&self) -> i16;

    /// Returns the scale of the least significant place of the uncertainty.
    fn precision_uncertainty(&self) -> Option<i16>;

    /// Returns the number of significant decimal digits after the radix point
    /// when expressed in normal (non-scientific) notation, including leading
    /// zeros.
    fn dp(&self) -> u16;

    /// Returns the number of significant decimal digits in the significand.
    /// 0 is considered to have 0 significant figures.
    fn sf(&self) -> u8;

    /// Returns the number of significant decimal digits in the uncertainty.
    /// 0 is considered to have 0 significant figures.
    fn sf_uncertainty(&self) -> u8;

    /// Rounds to the specified precision i.e. to 10<sup>(prec)</sup>.
    ///
    /// The uncertainty is left unchanged.
    ///
    /// Note that for non-normalized types that support significant trailing zeros
    /// (e.g. `SciDecimal`, but not `SciFloat`), requesting a precision greater than
    /// the number currently has will result in an increase in the precision.
    ///
    /// # Panics
    ///
    /// This function will panic if the requested precision cannot be represented
    /// by the type.
    fn round_precision(self, prec: i16, mode: RoundingMode) -> Self;

    /// Rounds to the specified number of decimal places.
    ///
    /// The uncertainty is left unchanged.
    ///
    /// Note that for non-normalized types that support significant trailing zeros
    /// (e.g. `SciDecimal`, but not `SciFloat`), requesting a precision greater than
    /// the number currently has will result in an increase in the precision.
    ///
    /// # Panics
    ///
    /// This function will panic if the requested precision cannot be represented
    /// by the type.
    fn round_dp(self, dp: u16, mode: RoundingMode) -> Self;

    /// Rounds to the specified number of significant figures.
    ///
    /// The uncertainty is left unchanged.
    ///
    /// Note that for non-normalized types that support significant trailing zeros
    /// (e.g. `SciDecimal`, but not `SciFloat`), requesting a precision greater than
    /// the number currently has will result in an increase in the precision.
    ///
    /// # Panics
    ///
    /// This function will panic if the requested precision cannot be represented
    /// by the type.
    fn round_sf(self, sf: u8, mode: RoundingMode) -> Self;

    /// Rounds the number so that its precision matches that of the uncertainty.
    ///
    /// If the number is exact, it is not rounded at all.
    ///
    /// The uncertainty is left unchanged.
    ///
    /// Note that for non-normalized types that support significant trailing zeros
    /// (e.g. `SciDecimal`, but not `SciFloat`), requesting a precision greater than
    /// the number currently has will result in an increase in the precision.
    ///
    /// # Panics
    ///
    /// This function will panic if the requested precision cannot be represented
    /// by the type.
    fn round_match_uncertainty(self, mode: RoundingMode) -> Self;

    /// Rounds the uncertainty to the specified number of significant figures,
    /// and then rounds the number to the same precision.
    ///
    /// If the number is exact, it is not rounded at all.
    ///
    /// Note that for non-normalized types that support significant trailing zeros
    /// (e.g. `SciDecimal`, but not `SciFloat`), requesting a precision greater than
    /// the number currently has will result in an increase in the precision.
    ///
    /// # Panics
    ///
    /// This function will panic if the requested precision cannot be represented
    /// by the type.
    fn round_match_uncertainty_sf(self, sf: u8, mode: RoundingMode) -> Self;

    /// Rounds the uncertainty to the specified precision i.e. to 10<sup>(prec)</sup>.
    ///
    /// The number itself is left unchanged.
    ///
    /// Note that for non-normalized types that support significant trailing zeros
    /// (e.g. `SciDecimal`, but not `SciFloat`), requesting a precision greater than
    /// the number currently has will result in an increase in the precision.
    ///
    /// # Panics
    ///
    /// This function will panic if the requested precision cannot be represented
    /// by the type.
    fn round_uncertainty_precision(self, prec: i16, mode: RoundingMode) -> Self;

    /// Rounds the uncertainty to the specified number of decimal places.
    ///
    /// The number itself is left unchanged.
    ///
    /// Note that for non-normalized types that support significant trailing zeros
    /// (e.g. `SciDecimal`, but not `SciFloat`), requesting a precision greater than
    /// the number currently has will result in an increase in the precision.
    ///
    /// # Panics
    ///
    /// This function will panic if the requested precision cannot be represented
    /// by the type.
    fn round_uncertainty_dp(self, dp: u16, mode: RoundingMode) -> Self;

    /// Rounds the uncertainty to the specified number of significant figures.
    ///
    /// The number itself is left unchanged.
    ///
    /// Note that for non-normalized types that support significant trailing zeros
    /// (e.g. `SciDecimal`, but not `SciFloat`), requesting a precision greater than
    /// the number currently has will result in an increase in the precision.
    ///
    /// # Panics
    ///
    /// This function will panic if the requested precision cannot be represented
    /// by the type.
    fn round_uncertainty_sf(self, sf: u8, mode: RoundingMode) -> Self;

    /// Rounds the uncertainty so that its precision matches that of the number itself.
    ///
    /// The number itself is left unchanged.
    ///
    /// Note that for non-normalized types that support significant trailing zeros
    /// (e.g. `SciDecimal`, but not `SciFloat`), requesting a precision greater than
    /// the number currently has will result in an increase in the precision.
    ///
    /// # Panics
    ///
    /// This function will panic if the requested precision cannot be represented
    /// by the type.
    fn round_uncertainty_match_number(self, mode: RoundingMode) -> Self;

    /// Removes significant figures from the significand to afford the desired
    /// number.
    ///
    /// Equivalent to rounding towards zero.
    ///
    /// The uncertainty of the `SciNum` is left unchanged.
    ///
    /// # Panics
    ///
    /// This function may panic if the `SciNum` already has fewer significant figures
    /// than the requested number.
    fn trunc_sf(self, sf: u8) -> Self;
}
