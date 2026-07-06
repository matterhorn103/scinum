//! Comparison and ordering trait implementations for [`SciFloat`].

use crate::SciFloat;

impl PartialEq for SciFloat {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.number == other.number
    }
}

impl PartialOrd for SciFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.number.partial_cmp(&other.number)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eq() {
        // Basic case
        assert_eq!(SciFloat::new(3.0), SciFloat::new(3.0));
        // Not equal, basic case
        assert_ne!(SciFloat::new(3.0), SciFloat::new(4.0));
        // Both zero
        assert_eq!(SciFloat::new(0.0), SciFloat::new(0.0));
        // Both zero, one is negative zero
        assert_eq!(SciFloat::new(0.0), SciFloat::new(-0.0));
        // Opposite sign but same significand
        assert_ne!(SciFloat::new(3.0), SciFloat::new(-3.0));
        // Same value but different precision
        assert_eq!(SciFloat::new(200e3), SciFloat::new(2e5));
        // How is this different than the previous one?
        // Same value but different precision, small numbers
        //assert_eq!(SciFloat::new(200, 3), SciFloat::new(2, 5));
    }
}
