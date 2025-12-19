// SPDX-FileCopyrightText: 2025 Matthew Milner <matterhorn103@proton.me>
// SPDX-License-Identifier: MIT

use std::ops::{Add, Div, Mul, Rem, Sub};

use num_traits::{Num, One, Zero};

use crate::{SciNumeric, error::SciNumError};

pub struct SciNum<T> {
    number: T,
    uncertainty: T,
}

impl<T: Num + Copy> SciNumeric for SciNum<T> {
    type Numeric = T;
    
    fn number(&self) -> Self::Numeric {
        self.number
    }
    
    fn uncertainty(&self) -> Self::Numeric {
        self.uncertainty
    }
}

impl<T: Num> Num for SciNum<T> {
    type FromStrRadixErr = SciNumError;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        todo!()
    }
}

impl<T: Num> PartialEq for SciNum<T> {
    fn eq(&self, other: &Self) -> bool {
        self.number == other.number
    }
}

impl<T: Num> Zero for SciNum<T> {
    fn zero() -> Self {
        todo!()
    }

    fn is_zero(&self) -> bool {
        todo!()
    }
}

impl<T: Num> One for SciNum<T> {
    fn one() -> Self {
        todo!()
    }
}

impl<T: Num> Add for SciNum<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

impl<T: Num> Sub for SciNum<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

impl<T: Num> Mul for SciNum<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

impl<T: Num> Div for SciNum<T> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

impl<T: Num> Rem for SciNum<T> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        todo!()
    }
}
