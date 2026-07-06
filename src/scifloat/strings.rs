//! String-related methods and trait implementations for [`SciFloat`].

use std::{fmt::Display, str::FromStr};

use num_traits::Num;

use crate::{SciFloat, SciNumError};

impl Display for SciFloat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} +/- {}", self.number, self.uncertainty)
    }
}

impl FromStr for SciFloat {
    type Err = SciNumError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let number = s.parse();
        match number {
            Ok(num) => Ok(SciFloat::new(num)),
            Err(_) => Err(SciNumError::Parse(s.into())),
        }
    }
}

impl Num for SciFloat {
    type FromStrRadixErr = <f64 as Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(Self {
            number: f64::from_str_radix(str, radix)?,
            uncertainty: 0.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    //#[test]
    //fn debug() {
    //    let n = SciFloat::new_with_uncertainty(20, 2, 0);
    //    assert_eq!(format!("{n:?}"), "SciFloat { number: 20, uncertainty: 2 }");
    //}

    #[test]
    fn display() {
        // Numbers with up to five places either side of the decimal point should
        // be displayed using normal notation
        // Integers should display without any decimal point at all
        assert_eq!(SciFloat::new(20.0).to_string(), "20 +/- 0");
        assert_eq!(SciFloat::new(-20.0).to_string(), "-20 +/- 0");
        assert_eq!(SciFloat::new(99999.0).to_string(), "99999 +/- 0");
        assert_eq!(SciFloat::new(10000.0).to_string(), "10000 +/- 0");
        assert_eq!(SciFloat::new(1000.0).to_string(), "1000 +/- 0");
        assert_eq!(SciFloat::new(100.0).to_string(), "100 +/- 0");
        assert_eq!(SciFloat::new(10.0).to_string(), "10 +/- 0");
        assert_eq!(SciFloat::new(1.0).to_string(), "1 +/- 0");
        assert_eq!(SciFloat::new(0.1).to_string(), "0.1 +/- 0");
        assert_eq!(SciFloat::new(0.01).to_string(), "0.01 +/- 0");
        assert_eq!(SciFloat::new(0.001).to_string(), "0.001 +/- 0");
        assert_eq!(SciFloat::new(0.0001).to_string(), "0.0001 +/- 0");
        assert_eq!(SciFloat::new(0.00001).to_string(), "0.00001 +/- 0");
        assert_eq!(SciFloat::new(0.0325).to_string(), "0.0325 +/- 0");
        assert_eq!(SciFloat::new(-0.0325).to_string(), "-0.0325 +/- 0");
        assert_eq!(SciFloat::new(85.13).to_string(), "85.13 +/- 0");
        assert_eq!(SciFloat::new(81700.0).to_string(), "81700 +/- 0");

        assert_eq!(
            SciFloat::new_with_uncertainty(20.0, 2.0).to_string(),
            "20 +/- 2"
        );
        assert_eq!(
            SciFloat::new_with_uncertainty(10000.0, 15.0).to_string(),
            "10000 +/- 15"
        );
        assert_eq!(
            SciFloat::new_with_uncertainty(86.75309, 42.0).to_string(),
            "86.75309 +/- 42"
        );
        assert_eq!(
            SciFloat::new_with_uncertainty(-86.75309, 42.0).to_string(),
            "-86.75309 +/- 42"
        );
    }

    #[test]
    fn from_str() {
        // Integer
        assert_eq!(SciFloat::from_str("42").unwrap(), SciFloat::new(42.0));
        // Decimal
        assert_eq!(SciFloat::from_str("0.0859").unwrap(), SciFloat::new(859e-4));
        // Decimal without integral part before decimal point
        assert_eq!(SciFloat::from_str(".0859").unwrap(), SciFloat::new(859e-4));
        // Negative decimal
        assert_eq!(SciFloat::from_str("-3.12").unwrap(), SciFloat::new(-312e-2));
        // Scientific notation
        assert_eq!(SciFloat::from_str("1.5e8").unwrap(), SciFloat::new(15e7));
        // Scientific notation with negative exponent
        assert_eq!(SciFloat::from_str("2e-5").unwrap(), SciFloat::new(2e-5));
        // Negative number with positive exponent
        assert_eq!(
            SciFloat::from_str("-6.022e6").unwrap(),
            SciFloat::new(-6022e3)
        );
        // Large exponents
        assert_eq!(SciFloat::from_str("1.5e18").unwrap(), SciFloat::new(15e17));
        assert_eq!(
            SciFloat::from_str("-6.022e23").unwrap(),
            SciFloat::new(-6022e20)
        );
        // Capital E for exponent
        assert_eq!(SciFloat::from_str("1.5E8").unwrap(), SciFloat::new(15e7));
        // 16 significant figures must always be fine
        assert_eq!(
            SciFloat::from_str("0.5293040185492948").unwrap(),
            SciFloat::new(5293040185492948e-16)
        );
        // Make sure incorrectly formatted strings fail
        assert!(SciFloat::from_str("not a number").is_err());
        assert!(SciFloat::from_str("x.482").is_err());
        assert!(SciFloat::from_str("52.x").is_err());
        assert!(SciFloat::from_str("-2.42F-4").is_err());
    }
}
