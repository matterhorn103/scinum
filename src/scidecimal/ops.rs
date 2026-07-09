//! Arithmetic methods and trait implementations for [`SciDecimal`].
//!
//! This module does not include the implementation of `num_traits::Float`, which
//! defines most of the more complicated operations.

use std::{
    cmp::Ordering,
    ops::{Add, Div, Mul, Neg, Rem, Sub},
};

use num_traits::{Float, Inv, Pow, Zero};

use crate::{
    RoundingMode, SciDecimal, SciNum,
    scicast::{SciCast, SciCastFrom},
    uncertainties::uncertainty_fn_generator,
};

/// Arithmetic operations that return exact results with potentially excess precision,
/// useful for intermediate results to avoid rounding errors, but not to be
/// returned to the end user.
impl SciDecimal {
    /// Calculates `self + rhs` without uncertainty, permitting values for the
    /// significand greater than `SciDecimal::MAX_SIGNIFICAND` and up to `u64::MAX`.
    pub(crate) fn unbounded_add(self, rhs: Self) -> Self {
        // TODO If significand would be too large for u64, just round it and
        // increase the exponent instead of panicking

        // Handle NaN
        if self.is_nan() | rhs.is_nan() {
            return Self::NAN;
        }
        // Handle infinities
        match (self.inf_bit(), rhs.inf_bit()) {
            (true, true) => {
                if self.sign_bit() == rhs.sign_bit() {
                    // ∞ + ∞ = ∞, -∞ + -∞ = -∞
                    return self;
                } else {
                    // ∞ - ∞ = NaN
                    return Self::NAN;
                }
            }
            (true, false) => {
                return self;
            }
            (false, true) => {
                return rhs;
            }
            (false, false) => {}
        }
        // Handle zero
        if self.is_zero() {
            return rhs.number();
        } else if rhs.is_zero() {
            return self.number();
        }

        match self.exponent.cmp(&rhs.exponent) {
            // In the simplest case, the exponents are the same
            Ordering::Equal => {
                let number = self.signed_significand() + rhs.signed_significand();
                Self::new(number, self.exponent)
            }
            // Otherwise have to try and set the exponent to the same for both terms
            // Use whichever exponent is smallest
            Ordering::Less => {
                let exp_diff = rhs.exponent - self.exponent;
                let scaled = rhs.increase_precision(exp_diff.try_into().unwrap());
                let number = self.signed_significand() + scaled.signed_significand();
                Self::new(number, self.exponent)
            }
            Ordering::Greater => {
                let exp_diff = self.exponent - rhs.exponent;
                let scaled = self.increase_precision(exp_diff.try_into().unwrap());
                let number = scaled.signed_significand() + rhs.signed_significand();
                Self::new(number, scaled.exponent)
            }
        }
    }

    /// Calculates `self * rhs` without uncertainty, permitting values for the
    /// significand greater than `SciDecimal::MAX_SIGNIFICAND` and up to `u64::MAX`.
    pub(crate) fn unbounded_mul(self, rhs: Self) -> Self {
        // Handle NaN
        if self.is_nan() | rhs.is_nan() {
            return Self::NAN;
        }
        let negative = self.sign_bit() ^ rhs.sign_bit();
        // Handle infinities
        match (self.inf_bit(), rhs.inf_bit()) {
            (true, true) => {
                if negative {
                    return Self::NEG_INFINITY;
                } else {
                    return Self::INFINITY;
                }
            }
            (true, false) => {
                if rhs.is_zero() {
                    return Self::NAN;
                } else if negative {
                    return Self::NEG_INFINITY;
                } else {
                    return Self::INFINITY;
                }
            }
            (false, true) => {
                if self.is_zero() {
                    return Self::NAN;
                } else if negative {
                    return Self::NEG_INFINITY;
                } else {
                    return Self::INFINITY;
                }
            }
            (false, false) => {}
        }
        let (significand, exponent) = match self.significand.checked_mul(rhs.significand) {
            Some(s) => (s, self.exponent + rhs.exponent),
            None => {
                // Even though we have ~3 spare decimal digits of precision in
                // `u64` above our max allowed significand, multiplication can
                // still result in overflow as
                // `(10_u64.pow(16) - 1) * (10_u64.pow(16) - 1) == 99999999999999980000000000000001`
                // which is too large for `u64`:
                // `99999999999999980000000000000001 / (u64::MAX as u128) == 5421010862427`
                // but `u128` has plenty of space:
                // `u128::MAX / 99999999999999980000000000000001 == 3402823`
                // So, convert to `u128`, do mul (which won't ever overflow), then
                // truncate to get back to a significand representable as a `u64`
                // We want to truncate rather than round since we'll have three
                // excess digits left anyway, meaning we'll have to round later
                // before returning to the user - if we round now as well then
                // we'll get cumulative rounding errors!
                let mut too_wide = (self.significand as u128) * (rhs.significand as u128);
                let mut e = self.exponent + rhs.exponent;
                let s: u64 = loop {
                    match u64::try_from(too_wide) {
                        Err(_) => {
                            // Still too wide so divide by 10 to truncate
                            too_wide /= 10;
                            e += 1;
                            continue;
                        }
                        // We have reduced the precision of the significand enough that it
                        // into a u64 again
                        Ok(narrow_enough) => break narrow_enough,
                    }
                };
                (s, e)
            }
        };
        Self {
            uncertainty: 0,
            uncertainty_scale: 0,
            flags: negative as u8,
            exponent,
            significand,
        }
    }

    /// Calculates `self / rhs` without uncertainty, permitting values for the
    /// significand greater than `SciDecimal::MAX_SIGNIFICAND` and up to `u64::MAX`.
    ///
    /// # Special values
    ///
    /// Here, n is any finite, non-zero number.
    ///
    /// - ±0/±0 → `NaN`
    ///
    /// - ±n/±0 → ±∞
    ///
    /// - ±0/±∞ → ±0
    ///
    /// - ±n/±∞ → ±0
    ///
    /// - ±∞/±∞ → `NaN`
    ///
    /// - ±∞/±0 → ±∞
    ///
    /// - ±∞/±n → ±∞
    ///
    /// - Either `self` or `rhs` is `NaN` → `NaN`
    pub(crate) fn unbounded_div(self, rhs: Self) -> Self {
        // Handle NaN
        if self.is_nan() | rhs.is_nan() {
            return Self::NAN;
        }
        let negative = self.sign_bit() ^ rhs.sign_bit();
        // Handle infinities
        match (self.inf_bit(), rhs.inf_bit()) {
            (true, true) => {
                // ∞/∞ is undefined
                return Self::NAN;
            }
            (true, false) => {
                // ∞/n = ∞ for all n, including 0
                if negative {
                    return Self::NEG_INFINITY;
                } else {
                    return Self::INFINITY;
                }
            }
            (false, true) => {
                // n/∞ = 0 for all n, including 0
                if negative {
                    return Self::NEG_ZERO;
                } else {
                    return Self::ZERO;
                }
            }
            (false, false) => {}
        }
        // Handle zeros
        if rhs.is_zero() {
            if self.is_zero() {
                // 0/0 is undefined
                return Self::NAN;
            } else if negative {
                return Self::NEG_INFINITY;
            } else {
                return Self::INFINITY;
            }
        }
        if self.is_zero() {
            // Already checked for rhs being zero
            if negative {
                return Self::NEG_ZERO;
            } else {
                return Self::ZERO;
            }
        }
        // Increase precision of the numerator until the denominator goes into
        // it an exact number of times, or until the maximum precision - of
        // `u64` - is reached
        let mut lhs = self;
        //let mut iterations: u8 = 0;
        // Loop because we only want to increase the precision as much as we
        // absolutely have to
        while !lhs.significand.is_multiple_of(rhs.significand) {
            // iterations += 1;
            // if iterations > 100 {
            //     panic!("{}", iterations)
            // }
            // Crucially, we allow the precision to increase beyond 16 sf up to
            // the maximum of `u64`
            // Only allowing 16 sf means that the max significand is
            // 0b0000000000100011100001101111001001101111110000001111111111111111
            // which fits into 54 bits
            // This gives us ~10 bits of spare precision to use (~3 sig figs)
            match lhs.increase_precision_unbounded_checked(1) {
                Some(new) => {
                    lhs = new;
                }
                None => {
                    // Max precision was already reached last iteration
                    break;
                }
            }
        }
        let significand = lhs.significand / rhs.significand;
        let exponent = lhs.exponent - rhs.exponent;
        Self {
            uncertainty: 0,
            uncertainty_scale: 0,
            flags: negative as u8,
            exponent,
            significand,
        }
    }

    /// Calculates `self.powi(rhs)` without uncertainty, permitting values for the
    /// significand greater than `SciDecimal::MAX_SIGNIFICAND` and up to `u64::MAX`.
    pub(crate) fn unbounded_powi(self, n: i32) -> Self {
        if !self.is_normal() {
            todo!("Special values are not yet handled correctly by this method!")
        }
        if n.is_negative() {
            self.powi(n.abs()).inv()
        } else {
            let number = self.signed_significand().pow(n.try_into().unwrap());
            let exponent = self.exponent * i16::try_from(n).unwrap();
            Self::new(number, exponent)
        }
    }

    /// Calculates `self.pow(rhs)` without uncertainty, permitting values for the
    /// significand greater than `SciDecimal::MAX_SIGNIFICAND` and up to `u64::MAX`.
    pub(crate) fn unbounded_powf(self, rhs: Self) -> Self {
        if !(self.is_normal() && rhs.is_normal()) {
            todo!("Special values are not yet handled correctly by this method!")
        }
        todo!()
    }
}

impl Add for SciDecimal {
    type Output = Self;

    /// Performs the `+` operation.
    ///
    /// # Special values
    ///
    /// - ±0: no special behaviour
    ///
    /// - ±∞: if one number is an infinity, that infinity is returned; otherwise:
    ///   - ∞ + ∞ → ∞
    ///   - -∞ + -∞ → -∞
    ///   - ∞ - ∞ → `NaN`
    ///   - -∞ + ∞ → `NaN`
    ///
    /// - `NaN`: if either number is `NaN`, returns `NaN`
    fn add(self, rhs: Self) -> Self {
        let exact = self.unbounded_add(rhs);
        let result = if self.is_exact() && rhs.is_exact() {
            exact
        } else if !exact.is_finite() {
            // Uncertainty is infinity or NaN by definition anyway
            exact
        } else {
            let uncertainty =
                ((self.uncertainty().pow(2.into())) + rhs.uncertainty().pow(2.into())).sqrt();
            exact.with_uncertainty(uncertainty)
        };
        if result.significand > Self::MAX_SIGNIFICAND {
            result.round_sf(16, RoundingMode::HalfUp)
        } else {
            result
        }
    }
}

impl Add for &SciDecimal {
    type Output = SciDecimal;

    fn add(self, rhs: Self) -> SciDecimal {
        *self + *rhs
    }
}

impl Sub for SciDecimal {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let rhs = -rhs;
        self + rhs
    }
}

impl Sub for &SciDecimal {
    type Output = SciDecimal;

    fn sub(self, rhs: Self) -> SciDecimal {
        *self - *rhs
    }
}

impl Mul for SciDecimal {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let exact = self.unbounded_mul(rhs);
        let result = if self.is_exact() && rhs.is_exact() {
            exact
        } else if !exact.is_finite() {
            // Uncertainty is infinity or NaN by definition anyway
            exact
        } else {
            let uncertainty =
                (self.relative_uncertainty().powi(2) + rhs.relative_uncertainty().powi(2)).sqrt()
                    * exact.abs();
            exact.with_uncertainty(uncertainty)
        };
        if result.significand > Self::MAX_SIGNIFICAND {
            result.round_sf(16, RoundingMode::HalfUp)
        } else {
            result
        }
    }
}

impl Mul for &SciDecimal {
    type Output = SciDecimal;

    fn mul(self, rhs: Self) -> SciDecimal {
        *self * *rhs
    }
}

impl Div for SciDecimal {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        let exact = self.unbounded_div(rhs);
        let result = if self.is_exact() && rhs.is_exact() {
            exact
        } else if !exact.is_finite() {
            // Uncertainty is infinity or NaN by definition anyway
            exact
        } else if rhs.is_infinite() {
            // Special case where result is zero due to division by infinity
            // Uncertainty must also be exactly zero, but calculation by normal
            // method fails because the relative uncertainty of `rhs` is `NaN`
            exact
        } else {
            let uncertainty =
                (self.relative_uncertainty().powi(2) + rhs.relative_uncertainty().powi(2)).sqrt()
                    * exact.abs();
            exact.with_uncertainty(uncertainty)
        };
        if result.significand > Self::MAX_SIGNIFICAND {
            result.round_sf(16, RoundingMode::HalfUp)
        } else {
            result
        }
    }
}

impl Div for &SciDecimal {
    type Output = SciDecimal;

    fn div(self, rhs: Self) -> SciDecimal {
        *self / *rhs
    }
}

impl Rem for SciDecimal {
    type Output = Self;

    /// Performs the `%` operation.
    ///
    /// WARNING: Uncertainty propagation is not yet implemented for this method,
    /// and the returned result will be exact.
    fn rem(self, rhs: Self) -> Self {
        // Handle NaN
        if self.is_nan() | rhs.is_nan() {
            return Self::NAN;
        }
        // Handle infinities
        if self.inf_bit() {
            // Can't find remainder of infinity
            return Self::NAN;
        } else if rhs.inf_bit() {
            return self;
        }
        // Handle zeros
        if rhs.is_zero() {
            // n % 0 is undefined just like n / 0
            return Self::NAN;
        }
        // TODO implement natively, not via Decimal
        dbg!(&self);
        dbg!(self.to_string());
        let number = rust_decimal::Decimal::cast_from(self.number())
            % rust_decimal::Decimal::cast_from(rhs.number());
        // Don't calculate uncertainty as the remainder function is discontinuous,
        // making it tricky
        number.cast()
    }
}

impl Rem for &SciDecimal {
    type Output = SciDecimal;

    /// Performs the `%` operation.
    ///
    /// WARNING: Uncertainty propagation is not yet implemented for this method,
    /// and the returned result will be exact.
    fn rem(self, rhs: Self) -> SciDecimal {
        *self % *rhs
    }
}

impl Pow<Self> for SciDecimal {
    type Output = Self;

    /// Raise the `SciDecimal` to a `SciDecimal` power.
    fn pow(self, rhs: Self) -> Self {
        let exact = if rhs.is_exact()
            && rhs.exponent.is_zero()
            && (rhs.exponent <= i8::MAX.into() && rhs.exponent >= i8::MIN.into())
        {
            let n = rhs.signed_significand();
            self.unbounded_powi(
                n.try_into()
                    .expect("n has already been checked and should fit into even an i8"),
            )
        } else {
            self.unbounded_powf(rhs)
        };
        let result = if self.is_exact() && rhs.is_exact() {
            exact
        } else if !exact.is_finite() {
            // Uncertainty is infinity or NaN by definition anyway
            exact
        } else {
            // for c = a^b,
            //      σ_c = |c| sqrt( ((b/a)σ_a)^2 + (ln(a)⋅σ_b)^2 + 2⋅b⋅ln(a)⋅σ_ab/a )
            // if σ_ab = 0,
            //      σ_c = |c| sqrt( ((b/a)σ_a)^2 + (ln(a)⋅σ_b)^2 )
            todo!();
        };
        if result.significand > Self::MAX_SIGNIFICAND {
            result.round_sf(16, RoundingMode::HalfUp)
        } else {
            result
        }
    }
}

impl Pow<Self> for &SciDecimal {
    type Output = SciDecimal;

    fn pow(self, rhs: Self) -> SciDecimal {
        (*self).pow(*rhs)
    }
}

impl Neg for SciDecimal {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            flags: self.flags ^ 0x01,
            ..self
        }
    }
}

impl Neg for &SciDecimal {
    type Output = SciDecimal;

    #[inline]
    fn neg(self) -> SciDecimal {
        SciDecimal {
            flags: self.flags ^ 0x01,
            ..*self
        }
    }
}

impl Inv for SciDecimal {
    type Output = Self;

    #[inline]
    fn inv(self) -> Self {
        Self::ONE / self
    }
}

impl Inv for &SciDecimal {
    type Output = SciDecimal;

    #[inline]
    fn inv(self) -> SciDecimal {
        SciDecimal::ONE / *self
    }
}

/// Methods to get correlated uncertainties.
impl SciDecimal {
    /// Function that calculates a result and its uncertainty for any non-linear
    /// differentiable function f(a, b).
    ///
    /// `f` is a function that gives the *exact* result of f(a, b), where a is `self`.
    ///
    /// `partderiv_a` and `partderiv_b` are the partial derivatives of f w.r.t. a and b.
    ///
    /// `ρ_ab` is the correlation between a and b, which should be 0 (uncorrelated),
    /// 1 (completely correlated) or a value between the two.
    fn calculate_with_uncertainty<F, A, B>(
        self,
        b: Self,
        f: F,
        partderiv_a: A,
        partderiv_b: B,
        ρ_ab: Self,
    ) -> Self
    where
        F: Fn(Self, Self) -> Self,
        A: Fn(Self, Self) -> Self,
        B: Fn(Self, Self) -> Self,
    {
        let exact = f(self, b);
        let uncertainty_fn = uncertainty_fn_generator(partderiv_a, partderiv_b);
        let uncertainty = uncertainty_fn(self, b, self.uncertainty(), b.uncertainty(), ρ_ab);
        exact.with_uncertainty(uncertainty)
    }

    /// Calculates the sum of two values with correlated uncertainties.
    ///
    /// `correlation` must be 0 (uncorrelated), 1 (completely correlated) or a
    /// value between the two.
    #[allow(unused_variables)]
    pub fn correlated_add(self, rhs: Self, correlation: Self) -> Self {
        if correlation < Self::ZERO || correlation > Self::ONE {
            panic!("Correlation must be between 0 and 1!")
        }
        self.calculate_with_uncertainty(
            rhs,
            Self::add,
            |a, b| Self::ONE,
            |a, b| Self::ONE,
            correlation,
        )
    }
}
/*
#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::sci;

    use super::*;

    /// Returns an iterator over all combinations of `2.5e5`, `0`, `inf`, their
    /// negative counterparts, and `NaN`.
    fn combos() -> Vec<(SciDecimal, SciDecimal)> {
        let vals = vec![
            sci!(2.5e5),
            sci!(-2.5e5),
            SciDecimal::ZERO,
            SciDecimal::NEG_ZERO,
            SciDecimal::INFINITY,
            SciDecimal::NEG_INFINITY,
            SciDecimal::NAN,
        ];
        vals.iter()
            .cloned()
            .cartesian_product(vals.iter().cloned())
            .collect()
    }

    #[test]
    fn check_combos_cast_f64() {
        for (val, _) in combos() {
            assert_eq!(val.to_plain_string(), f64::cast_from(val).to_string());
        }
    }

    #[test]
    fn add_exact() {
        let n1 = SciDecimal::new(40, 0);
        let n2 = sci!(5.1);
        let result = n1 + n2;
        assert_eq!(result, sci!(45.1));
    }

    #[test]
    fn add_with_uncertainty() {
        let n1 = SciDecimal::new_with_uncertainty(20, 2, 0);
        let n2 = SciDecimal::new_with_uncertainty(30, 5, 0);
        let result = n1 + n2;
        assert_eq!(result.number(), sci!(50));
        //assert_eq!(
        //    Decimal::try_from(result.uncertainty()).unwrap().round_dp(5),
        //    dec!(5.3851648071345).round_dp(5)
        //);
    }

    #[test]
#[rustfmt::skip]
fn add_special() {
    let p = sci!(2.5e5);
    let n = sci!(-2.5e5);
    let nan = SciDecimal::NAN;
    let inf = SciDecimal::INFINITY;
    let ninf = SciDecimal::NEG_INFINITY;
    let zero = SciDecimal::ZERO;
    let nzero = SciDecimal::NEG_ZERO;
    // Check positive zero is always created when summing to zero
    assert_eq!( (p      + n     ),  zero);
    assert_eq!( (n      + p     ),  zero);
    // NaN
    assert!(    (nan    + nan   )   .is_nan());
    assert!(    (nan    + p     )   .is_nan());
    assert!(    (nan    + n     )   .is_nan());
    assert!(    (nan    + inf   )   .is_nan());
    assert!(    (nan    + ninf  )   .is_nan());
    assert!(    (nan    + zero  )   .is_nan());
    assert!(    (nan    + nzero )   .is_nan());
    assert!(    (p      + nan   )   .is_nan());
    assert!(    (n      + nan   )   .is_nan());
    assert!(    (inf    + nan   )   .is_nan());
    assert!(    (ninf   + nan   )   .is_nan());
    assert!(    (zero   + nan   )   .is_nan());
    assert!(    (nzero  + nan   )   .is_nan());
    // Infinities
    assert_eq!( (inf    + inf   ),  inf);
    assert_eq!( (ninf   + ninf  ),  ninf);
    assert!(    (inf    + ninf  )   .is_nan());
    assert!(    (ninf   + inf   )   .is_nan());
    assert_eq!( (inf    + p     ),  inf);
    assert_eq!( (inf    + n     ),  inf);
    assert_eq!( (inf    + zero  ),  inf);
    assert_eq!( (inf    + nzero ),  inf);
    assert_eq!( (p      + inf   ),  inf);
    assert_eq!( (n      + inf   ),  inf);
    assert_eq!( (zero   + inf   ),  inf);
    assert_eq!( (nzero  + inf   ),  inf);
    assert_eq!( (ninf   + p     ),  ninf);
    assert_eq!( (ninf   + n     ),  ninf);
    assert_eq!( (ninf   + zero  ),  ninf);
    assert_eq!( (ninf   + nzero ),  ninf);
    assert_eq!( (p      + ninf  ),  ninf);
    assert_eq!( (n      + ninf  ),  ninf);
    assert_eq!( (zero   + ninf  ),  ninf);
    assert_eq!( (nzero  + ninf  ),  ninf);
    // Zeros
    assert_eq!( (zero   + zero  ),  zero);
    assert_eq!( (nzero  + nzero ),  nzero);
    assert_eq!( (zero   + nzero ),  zero);
    assert_eq!( (nzero  + zero  ),  zero);
    assert_eq!( (zero   + p     ),  p);
    assert_eq!( (zero   + n     ),  n);
    assert_eq!( (p      + zero  ),  p);
    assert_eq!( (n      + zero  ),  n);
    assert_eq!( (nzero  + p     ),  p);
    assert_eq!( (nzero  + n     ),  n);
    assert_eq!( (p      + nzero ),  p);
    assert_eq!( (n      + nzero ),  n);
}

    #[test]
    fn add_validate_vs_f64() {
        for (a, b) in combos() {
            let result_sci = a + b;
            let result_f64 = f64::cast_from(a) + f64::cast_from(b);
            assert_eq!(result_sci.to_plain_string(), result_f64.to_string());
        }
    }

    #[test]
    fn sub_exact() {
        let n1 = SciDecimal::new(20, 0);
        let n2 = SciDecimal::new(30, 0);
        assert_eq!(n1 - n2, sci!(-10));
    }

    #[test]
    fn sub_with_uncertainty() {
        let n1 = SciDecimal::new_with_uncertainty(20, 2, 0);
        let n2 = SciDecimal::new_with_uncertainty(30, 5, 0);
        let result = n1 - n2;
        assert_eq!(result, sci!(-10));
        assert_eq!(
            rust_decimal::Decimal::cast_from(result.uncertainty()).round_dp(5),
            rust_decimal_macros::dec!(5.3851648071345).round_dp(5)
        );
    }

    #[test]
#[rustfmt::skip]
fn sub_special() {
    let p = sci!(2.5e5);
    let n = sci!(-2.5e5);
    let nan = SciDecimal::NAN;
    let inf = SciDecimal::INFINITY;
    let ninf = SciDecimal::NEG_INFINITY;
    let zero = SciDecimal::ZERO;
    let nzero = SciDecimal::NEG_ZERO;
    // Check positive zero is always created when summing to zero
    assert_eq!( (p      - p     ),  zero);
    assert_eq!( (n      - n     ),  zero);
    // NaN
    assert!(    (nan    - nan   )   .is_nan());
    assert!(    (nan    - p     )   .is_nan());
    assert!(    (nan    - n     )   .is_nan());
    assert!(    (nan    - inf   )   .is_nan());
    assert!(    (nan    - ninf  )   .is_nan());
    assert!(    (nan    - zero  )   .is_nan());
    assert!(    (nan    - nzero )   .is_nan());
    assert!(    (p      - nan   )   .is_nan());
    assert!(    (n      - nan   )   .is_nan());
    assert!(    (inf    - nan   )   .is_nan());
    assert!(    (ninf   - nan   )   .is_nan());
    assert!(    (zero   - nan   )   .is_nan());
    assert!(    (nzero  - nan   )   .is_nan());
    // Infinities
    assert!(    (inf    - inf   )   .is_nan());
    assert!(    (ninf   - ninf  )   .is_nan());
    assert_eq!( (inf    - ninf  ),  inf);
    assert_eq!( (ninf   - inf   ),  ninf);
    assert_eq!( (inf    - p     ),  inf);
    assert_eq!( (inf    - n     ),  inf);
    assert_eq!( (inf    - zero  ),  inf);
    assert_eq!( (inf    - nzero ),  inf);
    assert_eq!( (p      - inf   ),  ninf);
    assert_eq!( (n      - inf   ),  ninf);
    assert_eq!( (zero   - inf   ),  ninf);
    assert_eq!( (nzero  - inf   ),  ninf);
    assert_eq!( (ninf   - p     ),  ninf);
    assert_eq!( (ninf   - n     ),  ninf);
    assert_eq!( (ninf   - zero  ),  ninf);
    assert_eq!( (ninf   - nzero ),  ninf);
    assert_eq!( (p      - ninf  ),  inf);
    assert_eq!( (n      - ninf  ),  inf);
    assert_eq!( (zero   - ninf  ),  inf);
    assert_eq!( (nzero  - ninf  ),  inf);
    // Zeros
    assert_eq!( (zero   - zero  ),  zero);
    assert_eq!( (nzero  - nzero ),  zero);
    assert_eq!( (zero   - nzero ),  zero);
    assert_eq!( (nzero  - zero  ),  nzero);
    assert_eq!( (zero   - p     ),  n);
    assert_eq!( (zero   - n     ),  p);
    assert_eq!( (p      - zero  ),  p);
    assert_eq!( (n      - zero  ),  n);
    assert_eq!( (nzero  - p     ),  n);
    assert_eq!( (nzero  - n     ),  p);
    assert_eq!( (p      - nzero ),  p);
    assert_eq!( (n      - nzero ),  n);
}

    #[test]
    fn mul_exact() {
        let n1 = SciDecimal::new(20, 0);
        let n2 = SciDecimal::new(30, 0);
        assert_eq!(n1 * n2, sci!(600));
        // Large values to confirm rounding behaviour
        // Max significand is 9_999_999_999_999_999
        // First, where the true significand doesn't even fit into a u64
        let n1 = SciDecimal::new(99_999_999_999_999, 0);
        let n2 = SciDecimal::new(89_999_999_999_999, 0);
        // Result = 8999999999999810000000000001 ≈ 8_999_999_999_999_810 * 10^12
        assert_eq!(n1 * n2, SciDecimal::new(8_999_999_999_999_810, 12));
        // Then check where the true significand just exceeds 16 sf but not u64::MAX
        let n1 = SciDecimal::new(999_999_999, 0);
        let n2 = SciDecimal::new(899_999_999, 0);
        // Result = 899_999_998_100_000_001 ≈ 8_999_999_981_000_000 * 10^2
        assert_eq!(n1 * n2, SciDecimal::new(8_999_999_981_000_000, 2));
    }

    #[test]
    fn mul_with_uncertainty() {
        let n1 = SciDecimal::new_with_uncertainty(20, 2, 0);
        let n2 = SciDecimal::new_with_uncertainty(30, 5, 0);
        let result = n1 * n2;
        assert_eq!(result.number(), sci!(600));
        assert_eq!(
            rust_decimal::Decimal::cast_from(result.uncertainty()).round_dp(5),
            rust_decimal_macros::dec!(116.619037896906).round_dp(5)
        );
        let ft = sci!(0.3048);
        let square_ft = ft * ft;
        assert_eq!(square_ft, sci!(0.09290304));
    }

    #[test]
#[rustfmt::skip]
fn mul_special() {
    let p = sci!(2.5e5);
    let n = sci!(-2.5e5);
    let nan = SciDecimal::NAN;
    let inf = SciDecimal::INFINITY;
    let ninf = SciDecimal::NEG_INFINITY;
    let zero = SciDecimal::ZERO;
    let nzero = SciDecimal::NEG_ZERO;
    // NaN
    assert!(    (nan    * nan   )   .is_nan());
    assert!(    (nan    * p     )   .is_nan());
    assert!(    (nan    * n     )   .is_nan());
    assert!(    (nan    * inf   )   .is_nan());
    assert!(    (nan    * ninf  )   .is_nan());
    assert!(    (nan    * zero  )   .is_nan());
    assert!(    (nan    * nzero )   .is_nan());
    assert!(    (p      * nan   )   .is_nan());
    assert!(    (n      * nan   )   .is_nan());
    assert!(    (inf    * nan   )   .is_nan());
    assert!(    (ninf   * nan   )   .is_nan());
    assert!(    (zero   * nan   )   .is_nan());
    assert!(    (nzero  * nan   )   .is_nan());
    // Infinities
    assert_eq!( (inf    * inf   ),  inf);
    assert_eq!( (ninf   * ninf  ),  inf);
    assert_eq!( (inf    * ninf  ),  ninf);
    assert_eq!( (ninf   * inf   ),  ninf);
    assert_eq!( (inf    * p     ),  inf);
    assert_eq!( (inf    * n     ),  ninf);
    assert!(    (inf    * zero  )   .is_nan());
    assert!(    (inf    * nzero )   .is_nan());
    assert_eq!( (p      * inf   ),  inf);
    assert_eq!( (n      * inf   ),  ninf);
    assert!(    (zero   * inf   )   .is_nan());
    assert!(    (nzero  * inf   )   .is_nan());
    assert_eq!( (ninf   * p     ),  ninf);
    assert_eq!( (ninf   * n     ),  inf);
    assert!(    (ninf   * zero  )   .is_nan());
    assert!(    (ninf   * nzero )   .is_nan());
    assert_eq!( (p      * ninf  ),  ninf);
    assert_eq!( (n      * ninf  ),  inf);
    assert!(    (zero   * ninf  )   .is_nan());
    assert!(    (nzero  * ninf  )   .is_nan());
    // Zeros
    assert_eq!( (zero   * zero  ),  zero);
    assert_eq!( (nzero  * nzero ),  zero);
    assert_eq!( (zero   * nzero ),  nzero);
    assert_eq!( (nzero  * zero  ),  nzero);
    assert_eq!( (zero   * p     ),  zero);
    assert_eq!( (zero   * n     ),  nzero);
    assert_eq!( (p      * zero  ),  zero);
    assert_eq!( (n      * zero  ),  nzero);
    assert_eq!( (nzero  * p     ),  nzero);
    assert_eq!( (nzero  * n     ),  zero);
    assert_eq!( (p      * nzero ),  nzero);
    assert_eq!( (n      * nzero ),  zero);
}

    #[test]
    fn div_exact() {
        // Non-recurring result with same exponent
        assert_eq!(
            SciDecimal::new(60, 0) / SciDecimal::new(30, 0),
            SciDecimal::new(2, 0),
        );
        // Non-recurring result with different exponent
        assert_eq!(
            SciDecimal::new(30, 0) / SciDecimal::new(60, 0),
            SciDecimal::new(5, -1),
        );
        // Identical recurring results with different pairs of starting numbers
        assert_eq!(
            SciDecimal::new(30, 0) / SciDecimal::new(60, 0),
            SciDecimal::new(3, 6) / SciDecimal::new(6, 6),
        );
        // Recurring results to confirm rounding behaviour
        assert_eq!(
            (SciDecimal::new(1, 0) / SciDecimal::new(3, 0)),
            SciDecimal::new(3333333333333333, -16),
        );
        assert_eq!(
            (SciDecimal::new(2, 0) / SciDecimal::new(3, 0)),
            SciDecimal::new(6666666666666667, -16),
        );
        assert_eq!(
            (SciDecimal::new(1, 0) / SciDecimal::new(9, 0)),
            SciDecimal::new(1111111111111111, -16),
        );
    }

    #[test]
    fn div_with_uncertainty() {
        let n1 = SciDecimal::new_with_uncertainty(20, 2, 0);
        let n2 = SciDecimal::new_with_uncertainty(30, 5, 0);
        let result = n1 / n2;
        assert_eq!(
            rust_decimal::Decimal::cast_from(result.uncertainty()).round_dp(10),
            rust_decimal_macros::dec!(0.6666666667).round_dp(10)
        );
        assert_eq!(
            rust_decimal::Decimal::cast_from(result.uncertainty()).round_dp(5),
            rust_decimal_macros::dec!(0.129576708774340).round_dp(5)
        );
    }

    #[test]
    fn div_with_uncertainty_reversed() {
        let n1 = SciDecimal::new_with_uncertainty(20, 2, 0);
        let n2 = SciDecimal::new_with_uncertainty(30, 5, 0);
        let result = n2 / n1;
        assert_eq!(result, sci!(1.5));
        assert_eq!(
            rust_decimal::Decimal::cast_from(result.uncertainty()).round_dp(5),
            rust_decimal_macros::dec!(0.2915475947422).round_dp(5)
        );
    }

    #[test]
#[rustfmt::skip]
fn div_special() {
    let p = sci!(2.5e5);
    let n = sci!(-2.5e5);
    let nan = SciDecimal::NAN;
    let inf = SciDecimal::INFINITY;
    let ninf = SciDecimal::NEG_INFINITY;
    let zero = SciDecimal::ZERO;
    let nzero = SciDecimal::NEG_ZERO;
    // NaN
    assert!(    (nan    / nan   )   .is_nan());
    assert!(    (nan    / p     )   .is_nan());
    assert!(    (nan    / n     )   .is_nan());
    assert!(    (nan    / inf   )   .is_nan());
    assert!(    (nan    / ninf  )   .is_nan());
    assert!(    (nan    / zero  )   .is_nan());
    assert!(    (nan    / nzero )   .is_nan());
    assert!(    (p      / nan   )   .is_nan());
    assert!(    (n      / nan   )   .is_nan());
    assert!(    (inf    / nan   )   .is_nan());
    assert!(    (ninf   / nan   )   .is_nan());
    assert!(    (zero   / nan   )   .is_nan());
    assert!(    (nzero  / nan   )   .is_nan());
    // Infinities
    assert!(    (inf    / inf  )    .is_nan());
    assert!(    (ninf   / ninf   )  .is_nan());
    assert!(    (inf    / ninf  )   .is_nan());
    assert!(    (ninf   / inf   )   .is_nan());
    assert_eq!( (inf    / p     ),  inf);
    assert_eq!( (inf    / n     ),  ninf);
    assert_eq!( (inf    / zero  ),  inf);
    assert_eq!( (inf    / nzero ),  ninf);
    assert_eq!( (p      / inf   ),  zero);
    assert_eq!( (n      / inf   ),  nzero);
    assert_eq!( (zero   / inf   ),  zero);
    assert_eq!( (nzero  / inf   ),  nzero);
    assert_eq!( (ninf   / p     ),  ninf);
    assert_eq!( (ninf   / n     ),  inf);
    assert_eq!( (ninf   / zero  ),  ninf);
    assert_eq!( (ninf   / nzero ),  inf);
    assert_eq!( (p      / ninf  ),  nzero);
    assert_eq!( (n      / ninf  ),  zero);
    assert_eq!( (zero   / ninf  ),  nzero);
    assert_eq!( (nzero  / ninf  ),  zero);
    // Zeros
    assert!(    (zero   / zero )    .is_nan());
    assert!(    (nzero  / nzero)    .is_nan());
    assert!(    (zero   / nzero)    .is_nan());
    assert!(    (nzero  / zero )    .is_nan());
    assert_eq!( (zero   / p     ),  zero);
    assert_eq!( (zero   / n     ),  nzero);
    assert_eq!( (p      / zero  ),  zero);
    assert_eq!( (n      / zero  ),  nzero);
    assert_eq!( (nzero  / p     ),  nzero);
    assert_eq!( (nzero  / n     ),  zero);
    assert_eq!( (p      / nzero ),  nzero);
    assert_eq!( (n      / nzero ),  zero);
}

    #[test]
#[rustfmt::skip]
fn rem_special() {
    let p = sci!(2.5e5);
    let n = sci!(-2.5e5);
    let nan = SciDecimal::NAN;
    let inf = SciDecimal::INFINITY;
    let ninf = SciDecimal::NEG_INFINITY;
    let zero = SciDecimal::ZERO;
    let nzero = SciDecimal::NEG_ZERO;
    // Check zero has the sign of the dividend
    assert_eq!( (p      % p     ),  zero);
    assert_eq!( (n      % n     ),  nzero);
    assert_eq!( (p      % n     ),  zero);
    assert_eq!( (n      % p     ),  nzero);
    // NaN
    assert!(    (nan    % nan   )   .is_nan());
    assert!(    (nan    % p     )   .is_nan());
    assert!(    (nan    % n     )   .is_nan());
    assert!(    (nan    % inf   )   .is_nan());
    assert!(    (nan    % ninf  )   .is_nan());
    assert!(    (nan    % zero  )   .is_nan());
    assert!(    (nan    % nzero )   .is_nan());
    assert!(    (p      % nan   )   .is_nan());
    assert!(    (n      % nan   )   .is_nan());
    assert!(    (inf    % nan   )   .is_nan());
    assert!(    (ninf   % nan   )   .is_nan());
    assert!(    (zero   % nan   )   .is_nan());
    assert!(    (nzero  % nan   )   .is_nan());
    // Infinities
    assert!(    (inf    % inf   )   .is_nan());
    assert!(    (ninf   % ninf  )   .is_nan());
    assert!(    (inf    % ninf  )   .is_nan());
    assert!(    (ninf   % inf   )   .is_nan());
    assert!(    (inf    % p     )   .is_nan());
    assert!(    (inf    % n     )   .is_nan());
    assert!(    (inf    % zero  )   .is_nan());
    assert!(    (inf    % nzero )   .is_nan());
    assert_eq!( (p      % inf   ),  p);
    assert_eq!( (n      % inf   ),  n);
    assert_eq!( (zero   % inf   ),  zero);
    assert_eq!( (nzero  % inf   ),  nzero);
    assert!(    (ninf   % p     )   .is_nan());
    assert!(    (ninf   % n     )   .is_nan());
    assert!(    (ninf   % zero  )   .is_nan());
    assert!(    (ninf   % nzero )   .is_nan());
    assert_eq!( (p      % ninf   ), p);
    assert_eq!( (n      % ninf   ), n);
    assert_eq!( (zero   % ninf   ), zero);
    assert_eq!( (nzero  % ninf   ), nzero);
    // Zeros
    assert!(    (zero   % zero )   .is_nan());
    assert!(    (nzero  % nzero)   .is_nan());
    assert!(    (zero   % nzero)   .is_nan());
    assert!(    (nzero  % zero )   .is_nan());
    assert_eq!( (zero   % p     ),  zero);
    assert_eq!( (zero   % n     ),  zero);
    assert!(    (p      % zero  )   .is_nan());
    assert!(    (n      % zero  )   .is_nan());
    assert_eq!( (nzero  % p     ),  nzero);
    assert_eq!( (nzero  % n     ),  nzero);
    assert!(    (p      % nzero  )   .is_nan());
    assert!(    (n      % nzero  )   .is_nan());
}

    #[test]
    fn powi_exact() {
        let n = SciDecimal::new(4, 0);
        assert_eq!(n.powi(2), sci!(16));
        assert_eq!(n.powi(3), sci!(64));
        assert_eq!(n.powi(-1), sci!(0.25));
        assert_eq!(n.powi(-2), sci!(0.0625));
    }

    #[test]
    fn powi_with_uncertainty() {
        let n = SciDecimal::new_with_uncertainty(20, 2, 0);
        let result = n.powi(2);
        assert_eq!(result.number(), sci!(400));
        // Currently fails, calculates an uncertainty of 8000
        assert_eq!(result.uncertainty(), sci!(80));
    }

    #[test]
    fn inv() {
        assert_eq!(SciDecimal::new(4, 0).inv(), SciDecimal::new(25, -2));
        assert_eq!(SciDecimal::new(5, -1).inv(), SciDecimal::new(2, 0));
    }

    #[test]
    fn neg() {
        let n_pos = SciDecimal::new(4, 0);
        let n_neg = n_pos.neg();
        assert_eq!(n_neg, SciDecimal::new(-4, 0));
        assert!(n_neg.sign_bit());
        assert_eq!(n_neg.significand, 4);
        let n_roundtrip = n_neg.neg();
        assert!(!n_roundtrip.sign_bit());
        assert_eq!(n_roundtrip, n_pos);
    }
}
*/
