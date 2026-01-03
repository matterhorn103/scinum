// SPDX-FileCopyrightText: 2026 Matthew Milner <matterhorn103@proton.me>
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::ops::{Add, Div, Mul, Rem, Sub};

use bigdecimal::BigDecimal;
use num_traits::{Num, One, Zero};

use crate::{SciDecimal, SciNum};

#[derive(Clone, Debug)]
pub struct SciBigDecimal {
    number: BigDecimal,
    uncertainty: BigDecimal,
}

impl SciBigDecimal {
    pub fn new(number: BigDecimal) -> Self {
        Self { number, uncertainty: BigDecimal::zero() }
    }

    pub fn new_with_uncertainty(number: BigDecimal, uncertainty: BigDecimal) -> Self {
        Self { number, uncertainty }
    }
}

impl SciNum for SciBigDecimal {
    type Number = BigDecimal;

    fn number(&self) -> Self::Number {
        self.number.clone()
    }

    fn uncertainty(&self) -> Self::Number {
        self.uncertainty.clone()
    }

    fn with_uncertainty(self, uncertainty: Self::Number) -> Self {
        Self {
            number: self.number,
            uncertainty,
        }
    }
}

impl From<BigDecimal> for SciBigDecimal {
    fn from(n: BigDecimal) -> Self {
        Self { number: n, uncertainty: BigDecimal::zero() }
    }
}

impl From<SciDecimal> for SciBigDecimal {
    fn from(n: SciDecimal) -> Self {
        Self { number: n.number().into(), uncertainty: n.uncertainty().into() }
    }
}

impl Num for SciBigDecimal {
    type FromStrRadixErr = <bigdecimal::BigDecimal as bigdecimal::Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(Self { number: BigDecimal::from_str_radix(str, radix)?, uncertainty: BigDecimal::zero() })
    }
}

impl Zero for SciBigDecimal {
    fn zero() -> Self {
        Self { number: BigDecimal::zero(), uncertainty: BigDecimal::zero() }
    }

    fn is_zero(&self) -> bool {
        self.number.is_zero()
    }
}

impl One for SciBigDecimal {
    fn one() -> Self {
        Self { number: BigDecimal::one(), uncertainty: BigDecimal::zero() }
    }
}

impl PartialEq for SciBigDecimal {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.number == other.number
    }
}

impl Eq for SciBigDecimal {}

impl PartialOrd for SciBigDecimal {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SciBigDecimal {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.number.cmp(&other.number)
    }
}

impl Add for SciBigDecimal {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let number = self.number + rhs.number;
        let uncertainty = (self.uncertainty.powi(2) + rhs.uncertainty.powi(2)).sqrt().unwrap();
        Self { number, uncertainty }
    }
}

impl Sub for SciBigDecimal {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let number = self.number - rhs.number;
        let uncertainty = (self.uncertainty.powi(2) + rhs.uncertainty.powi(2)).sqrt().unwrap();
        Self { number, uncertainty }
    }
}

impl Mul for SciBigDecimal {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let number = &self.number * &rhs.number;
        let uncertainty = (self.relative_uncertainty().powi(2) + rhs.relative_uncertainty().powi(2)).sqrt().unwrap() * number.abs();
        Self { number, uncertainty }
    }
}

impl Div for SciBigDecimal {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        let number = &self.number / &rhs.number;
        let uncertainty = (self.relative_uncertainty().powi(2) + rhs.relative_uncertainty().powi(2)).sqrt().unwrap() * number.abs();
        Self { number, uncertainty }
    }
}

impl Rem for SciBigDecimal {
    type Output = Self;

    /// Performs the `%` operation.
    ///
    /// NOTE: Uncertainty propagation is not implemented for this method,
    /// and the returned result is exact.
    fn rem(self, rhs: Self) -> Self {
        let number = self.number % rhs.number;
        number.into()
    }
}
