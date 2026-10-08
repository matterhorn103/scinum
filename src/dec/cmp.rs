//! Comparison and ordering trait implementations for [`SciDecimal`].

use num_traits::{Float, Zero};

use crate::{SciDecimal, SciNum};

impl PartialEq for SciDecimal {
    fn eq(&self, other: &Self) -> bool {
        // NaN is never equal to anything, even itself
        if self.is_nan() | other.is_nan() {
            false
        // +0 == +0, but also +0 == -0
        } else if self.is_zero() && other.is_zero() {
            true
        } else if self.is_zero() || other.is_zero() {
            false
        // Can't be equal if sign is different, so short circuit if so
        } else if self.sign_bit() != other.sign_bit() {
            false
        // ∞ == ∞, -∞ == -∞, +∞ != -∞ but we already checked the signs are the same
        } else if self.inf_bit() & other.inf_bit() {
            true
        } else if self.inf_bit() | other.inf_bit() {
            false
        } else if self.exponent == other.exponent {
            self.significand == other.significand
        // Might be the same value but to different precision
        } else if self.significand.is_multiple_of(other.significand) {
            let factor = self.significand / other.significand;
            // 0 counts as a multiple of 10
            if factor.is_multiple_of(10) {
                let order_diff = factor.ilog10();
                self.exponent + order_diff as i16 == other.exponent
            } else {
                false
            }
        } else if other.significand.is_multiple_of(self.significand) {
            let factor = other.significand / self.significand;
            if factor.is_multiple_of(10) {
                let order_diff = factor.ilog10();
                other.exponent + order_diff as i16 == self.exponent
            } else {
                false
            }
        } else {
            false
        }
    }
}

impl PartialOrd for SciDecimal {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;

        // NaN can't be compared
        if self.is_nan() | other.is_nan() {
            return None;
        }
        if self.is_zero() {
            if other.is_zero() {
                // Zeros are equal regardless of sign
                return Some(Ordering::Equal);
            } else if other.sign_bit() {
                return Some(Ordering::Greater);
            } else {
                return Some(Ordering::Less);
            }
        } else if other.is_zero() {
            // Checked for both being zero already
            if self.sign_bit() {
                return Some(Ordering::Less);
            } else {
                return Some(Ordering::Greater);
            }
        }
        // Different signs are easily ordered
        if self.sign_bit() != other.sign_bit() {
            return Some(if self.sign_bit() {
                Ordering::Less
            } else {
                Ordering::Greater
            });
        }
        // Infinities
        match (self.inf_bit(), other.inf_bit()) {
            (true, true) => {
                // Must be same sign because we already compared signs
                return Some(Ordering::Equal);
            }
            (true, false) => {
                return Some(if self.sign_bit() {
                    Ordering::Less
                } else {
                    Ordering::Greater
                });
            }
            (false, true) => {
                return Some(if other.sign_bit() {
                    Ordering::Greater
                } else {
                    Ordering::Less
                });
            }
            (false, false) => {}
        }
        // Same sign, both finite, neither is zero
        let lhs_magnitude = self.precision_most_significant_fig();
        let rhs_magnitude = other.precision_most_significant_fig();
        let ordering = if lhs_magnitude != rhs_magnitude {
            lhs_magnitude.cmp(&rhs_magnitude)
        } else if self.exponent == other.exponent {
            // Easy to compare them if the exponents are the same
            self.significand.cmp(&other.significand)
        } else {
            // Otherwise have to set to same precision
            let mut lhs_sig = self.significand as u128;
            let mut rhs_sig = other.significand as u128;
            let exp_diff = self.exponent - other.exponent;
            if exp_diff.is_positive() {
                // e.g. self = 3e6, other = 3072e3
                // Have to make self into 3000e3 to be able to compare
                lhs_sig *= 10_u128.pow(exp_diff.unsigned_abs().into());
                lhs_sig.cmp(&rhs_sig)
            } else {
                // e.g. self = 3072e3, other = 3e6
                // Have to make other into 3000e3 to be able to compare
                rhs_sig *= 10_u128.pow(exp_diff.unsigned_abs().into());
                lhs_sig.cmp(&rhs_sig)
            }
        };
        // If both are negative then the order is actually the reverse
        Some(if self.sign_bit() {
            ordering.reverse()
        } else {
            ordering
        })
    }
}

/*
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eq() {
        // Basic case
        assert_eq!(SciDecimal::new(3, 0), SciDecimal::new(3, 0));
        // Not equal, basic case
        assert_ne!(SciDecimal::new(3, 0), SciDecimal::new(4, 0));
        // Both zero
        assert_eq!(SciDecimal::new(0, 0), SciDecimal::new(0, 0));
        // Both zero, one is negative zero
        assert_eq!(SciDecimal::new(0, 0), SciDecimal::new(-0, 0));
        // Opposite sign but same significand
        assert_ne!(SciDecimal::new(3, 0), SciDecimal::new(-3, 0));
        // Same value but different precision
        assert_eq!(SciDecimal::new(200, 3), SciDecimal::new(2, 5));
        // Same value but different precision, small numbers
        assert_eq!(SciDecimal::new(200, 3), SciDecimal::new(2, 5));
    }
}
*/
