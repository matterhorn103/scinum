//! Arithmetic methods and trait implementations for [`SciFloat`].
//!
//! This module does not include the implementation of `num_traits::Float`, which
//! defines most of the more complicated operations.

use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use num_traits::{Float, Inv, Pow};

use crate::{SciFloat, SciNum};

impl Add for SciFloat {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let number = self.number + rhs.number;
        let uncertainty = (self.uncertainty.powi(2) + rhs.uncertainty.powi(2)).sqrt();
        Self {
            number,
            uncertainty,
        }
    }
}

impl Sub for SciFloat {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let number = self.number - rhs.number;
        let uncertainty = (self.uncertainty.powi(2) + rhs.uncertainty.powi(2)).sqrt();
        Self {
            number,
            uncertainty,
        }
    }
}

impl Mul for SciFloat {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let number = self.number * rhs.number;
        let uncertainty =
            (self.relative_uncertainty().powi(2) + rhs.relative_uncertainty().powi(2)).sqrt()
                * number.abs();
        Self {
            number,
            uncertainty,
        }
    }
}

impl Div for SciFloat {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        let number = self.number / rhs.number;
        let uncertainty =
            (self.relative_uncertainty().powi(2) + rhs.relative_uncertainty().powi(2)).sqrt()
                * number.abs();
        Self {
            number,
            uncertainty,
        }
    }
}

impl Rem for SciFloat {
    type Output = Self;

    /// Performs the `%` operation.
    ///
    /// NOTE: Uncertainty propagation is not implemented for this method,
    /// and the returned result is exact.
    fn rem(self, rhs: Self) -> Self {
        let number = self.number % rhs.number;
        // Don't calculate uncertainty as the remainder function is discontinuous,
        // making it tricky
        number.into()
    }
}

impl Pow<Self> for SciFloat {
    type Output = Self;

    /// Raise the `SciFloat` to a `SciFloat` power.
    /// Currently missing correlated uncertainties.
    fn pow(self, rhs: Self) -> Self {
        if !(self.is_finite() && rhs.is_finite()) {
            todo!("Special values are not yet handled correctly by this method!")
        }
        let result = self.number.pow(rhs.number);
        if self.is_exact() {
            SciFloat::new(result)
        } else {
            let uncertainty = result.abs()
                * ((self.relative_uncertainty() * rhs.number).powi(2)
                    + (self.number.ln() * rhs.uncertainty).powi(2))
                .sqrt();
            SciFloat::new_with_uncertainty(result, uncertainty)
        }
    }
}

impl Pow<Self> for &SciFloat {
    type Output = SciFloat;

    fn pow(self, rhs: Self) -> SciFloat {
        (*self).pow(*rhs)
    }
}

impl Neg for SciFloat {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            number: -self.number,
            ..self
        }
    }
}

impl Neg for &SciFloat {
    type Output = SciFloat;

    #[inline]
    fn neg(self) -> SciFloat {
        SciFloat {
            number: -self.number,
            ..*self
        }
    }
}

impl Inv for SciFloat {
    type Output = Self;

    #[inline]
    fn inv(self) -> Self {
        Self::ONE / self
    }
}

impl Inv for &SciFloat {
    type Output = SciFloat;

    #[inline]
    fn inv(self) -> SciFloat {
        SciFloat::ONE / *self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_exact() {
        let n1 = SciFloat::new(40.0);
        let n2 = SciFloat::new(5.1);
        let result = n1 + n2;
        assert_eq!(result, SciFloat::new(45.1));
    }

    #[test]
    fn add_with_uncertainty() {
        let n1 = SciFloat::new_with_uncertainty(20.0, 2.0);
        let n2 = SciFloat::new_with_uncertainty(30.0, 5.0);
        let result = n1 + n2;
        assert_eq!(result.number(), 50.0);
        //assert_eq!(
        //    Decimal::try_from(result.uncertainty()).unwrap().round_dp(5),
        //    dec!(5.3851648071345).round_dp(5)
        //);
    }

    #[test]
    fn sub_exact() {
        let n1 = SciFloat::new(20.0);
        let n2 = SciFloat::new(30.0);
        assert_eq!(n1 - n2, SciFloat::new(-10.0));
    }

    #[test]
    fn sub_with_uncertainty() {
        let n1 = SciFloat::new_with_uncertainty(20.0, 2.0);
        let n2 = SciFloat::new_with_uncertainty(30.0, 5.0);
        let result = n1 - n2;
        assert_eq!(result, SciFloat::new(-10.0));
        assert_eq!(result.uncertainty(), 5.385164807134504);
    }

    #[test]
    fn mul_exact() {
        let n1 = SciFloat::new(20.0);
        let n2 = SciFloat::new(30.0);
        assert_eq!(n1 * n2, SciFloat::new(600.0));
    }

    #[test]
    fn mul_with_uncertainty() {
        let n1 = SciFloat::new_with_uncertainty(20.0, 2.0);
        let n2 = SciFloat::new_with_uncertainty(30.0, 5.0);
        let result = n1 * n2;
        assert_eq!(result.number(), 600.0);
        assert_eq!(result.uncertainty(), 116.61903789690601);
        let ft = SciFloat::from(0.3048);
        let square_ft = ft * ft;
        assert_eq!(square_ft, SciFloat::new(0.09290304));
    }

    #[test]
    fn div_exact() {
        // Non-recurring result with same exponent
        assert_eq!(
            SciFloat::new(60.0) / SciFloat::new(30.0),
            SciFloat::new(2.0),
        );
        // Non-recurring result with different exponent
        assert_eq!(
            SciFloat::new(30.0) / SciFloat::new(60.0),
            SciFloat::new(5e-1),
        );
        // Identical recurring results with different pairs of starting numbers
        assert_eq!(
            SciFloat::new(30.0) / SciFloat::new(60.0),
            SciFloat::new(3e6) / SciFloat::new(6e6),
        );
        // Recurring results
        assert_eq!(
            (SciFloat::new(1.0) / SciFloat::new(3.0)),
            SciFloat::new(3333333333333333e-16),
        );
        assert_eq!(
            (SciFloat::new(1.0) / SciFloat::new(9.0)),
            SciFloat::new(1111111111111111e-16),
        );
    }

    #[test]
    fn div_with_uncertainty() {
        let n1 = SciFloat::new_with_uncertainty(20.0, 2.0);
        let n2 = SciFloat::new_with_uncertainty(30.0, 5.0);
        let result = n1 / n2;
        dbg!(result);
        assert_eq!(result.number(), 0.6666666666666666);
        assert_eq!(result.uncertainty(), 0.129576708774340);
    }

    #[test]
    fn div_with_uncertainty_reversed() {
        let n1 = SciFloat::new_with_uncertainty(20.0, 2.0);
        let n2 = SciFloat::new_with_uncertainty(30.0, 5.0);
        let result = n2 / n1;
        assert_eq!(result, SciFloat::new(1.5));
        assert_eq!(result.uncertainty(), 0.291547594742265);
    }

    #[test]
    fn inv() {
        assert_eq!(SciFloat::new(4.0).inv(), SciFloat::new(25e-2));
        assert_eq!(SciFloat::new(5e-1).inv(), SciFloat::new(2.0));
    }

    #[test]
    fn neg() {
        let n_pos = SciFloat::new(4.0);
        let n_neg = n_pos.neg();
        assert_eq!(n_neg, SciFloat::new(-4.0));
        assert_eq!(n_neg.number(), -4.0);
        let n_roundtrip = n_neg.neg();
        assert_eq!(n_roundtrip, n_pos);
    }
}
