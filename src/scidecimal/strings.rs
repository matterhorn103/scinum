//! String-related methods and trait implementations for [`SciDecimal`].

use core::fmt;
use std::str::FromStr;

use bigdecimal::BigDecimal;
use num_traits::{Num, Zero};
use regex::Regex;

use crate::{RoundingMode, SciDecimal, SciNum, SciNumError, scicast::SciCast};

impl SciDecimal {
    pub fn to_plain_string(&self) -> String {
        // Handle NaN
        if self.nan {
            return String::from("NaN");
        }
        // Get sign character
        let sign = if self.negative {
            String::from("-")
        } else {
            String::new()
        };
        // Handle infinities
        if self.inf {
            return format!("{sign}inf");
        }
        // Handle zeros
        if self.is_zero() {
            // TODO Have this display the uncertainty properly once they can be
            // displayed with +/-
            return format!("{}0", sign);
        }
        let significand = self.significand;
        let uncertainty = if self.is_exact() {
            String::new()
        } else {
            format!("({})", self.uncertainty)
        };
        // Integers
        if self.precision() > 0 {
            // Need to add appropriate zeros as padding
            // e.g. 25(2)e4 needs to become 250000(20000)
            let zeros = "0".repeat(self.precision() as usize);
            let uncertainty = if !uncertainty.is_empty() {
                uncertainty.replace(")", &zeros) + ")"
            } else {
                uncertainty
            };
            format!("{sign}{significand}{zeros}{uncertainty}")
        // Numbers with both integral and fractional parts
        } else if self.precision_most_significant_fig() >= 0 {
            // 3.1 has precision = -1, sigfigs = 2
            // 42.764 has precision = -3, sigfigs = 5
            // 3.02 has precision = -2, sigfigs = 3
            let int_figs = self.sf() as u16 - self.precision().unsigned_abs();
            let mut int = significand.to_string();
            let frac = int.split_off(int_figs.into());
            format!("{sign}{int}.{frac}{uncertainty}")
        // Numbers with only a fractional part
        } else {
            // 0.005 needs to have two zeros, precision = -3, sigfigs = 1
            let zeros = // (-3).abs() - 1 = 2
                "0".repeat((self.precision().unsigned_abs() - self.sf() as u16).into());
            format!("{sign}0.{zeros}{significand}{uncertainty}")
        }
    }

    pub fn to_scientific_string(&self) -> String {
        // Handle NaN
        if self.nan {
            return String::from("NaN");
        }
        // Get sign character
        let sign = if self.negative {
            String::from("-")
        } else {
            String::new()
        };
        // Handle infinities
        if self.inf {
            return format!("{}inf", sign);
        }
        let uncertainty = if self.is_exact() {
            String::new()
        } else {
            format!("({})", self.uncertainty)
        };
        let (int, zeros, frac, _, exp) = self.scientific_parts().unwrap();
        let zeros = "0".repeat(zeros.into());
        // Fractional part might not have any places at all (e.g. 2e6)
        if frac == 0 {
            format!("{int}{uncertainty}e{exp}")
        } else {
            format!("{int}.{zeros}{frac}{uncertainty}e{exp}")
        }
    }
}

impl fmt::Display for SciDecimal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Numbers with up to five places either side of the decimal point should
        // be displayed using normal notation:
        // - 0.0325 = 3.25e-2 = (325, -4) => "0.0325"
        // - 85.130 = 8.5130e1 = (85130, -3) => "85.130"
        // If this is exceeded, display in scientific notation:
        // - 0.000325 = 3.25e-4 = (325, -6) => "3.25e-4"
        // - 8174036 = 8.174036e6 = (8174036, 0) => "8.174036e6"
        // Scientific notation should also be used if there are insignificant zeros
        // before the decimal point, so that the precision is indicated:
        // - 81700 with 3 sf = 8.17e4 = (817, 2) => "8.17e4"
        // - 81700 with 5 sf = 8.1700e4 = (81700, 0) => "81700"
        if self.precision() <= 0
            && self.precision() >= -5
            && self.precision_most_significant_fig() <= 4
        {
            write!(f, "{}", self.to_plain_string())
        // Otherwise, use scientific notation
        } else {
            write!(f, "{}", self.to_scientific_string())
        }
    }
}

impl FromStr for SciDecimal {
    type Err = SciNumError;

    /// Parses a string and attempts to create a corresponding `SciDecimal`.
    ///
    /// A correctly formed string will *always* return a `SciDecimal`:
    ///
    /// - Excess precision is rounded to 16 significant figures (according to
    ///   [`RoundingMode::HalfEven`]).
    ///
    /// - If the absolute magnitude of the number is too *large* to be represented,
    ///   an infinity with the appropriate sign is returned.
    ///
    /// - If the absolute magnitude of the number is too *small* to be represented,
    ///   a zero with the appropriate sign is returned.
    ///
    /// In this way the behaviour is like that of [`SciCast::cast`], and indeed
    /// this method is used to effect the casts from several types.
    ///
    /// Fails if the string cannot be parsed at all.
    ///
    /// # Special values
    ///
    /// `inf` and `+inf` are parsed to positive infinity, `-inf` and `−inf` (with
    /// a hyphen-minus or a proper minus sign) are parsed to negative infinity.
    ///
    /// Likewise, `0` and `+0` are parsed to positive zero, `-0` and `−0` to negative.
    ///
    /// All variations of `NaN` are parsed case-insensitively to `NaN`, with any
    /// sign ignored.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // TODO Support special values
        // TODO Allow underscores to group digits
        let re =
            Regex::new(r"^(-)?(\d+)?(?:[.,](\d+))?(?:\((\d+)\))?(?:[eE]([+-]?\d+))?$").unwrap();
        let caps = re.captures(s).ok_or(SciNumError::Parse(s.into()))?;
        // Example given with "6.971e-7"
        let negative = caps.get(1).is_some(); // false
        let int = caps.get(2).map_or("", |m| m.as_str()); // "6"
        let frac = caps.get(3).map_or("", |m| m.as_str()); // "971"
        let frac_places = frac.len(); // 3
        let significand_string = int.to_owned() + frac;
        let mut significand_slice = &*significand_string;
        // Remove any leading zeros
        significand_slice = significand_slice.trim_start_matches('0');
        // But if the significand is just zero, it will have been removed
        if significand_slice.is_empty() {
            significand_slice = "0";
        };
        // If the precision in the string is too high, get it down to 19 sf
        // We'll round to 16 sf at the end
        let truncated_places = if significand_slice.len() > 19 {
            let excess_places = significand_slice.len() - 19;
            // OK to re-slice like this because our regex will only ever return
            // a string of 1-byte chars
            significand_slice = &significand_slice[..19];
            excess_places
        } else {
            0
        };
        let significand =
            u64::from_str(significand_slice).map_err(|_e| SciNumError::Parse(s.into()))?; // "6971"
        let uncertainty = caps
            .get(4)
            .map_or(Ok(0), |m| u32::from_str(m.as_str()))
            .map_err(|_e| SciNumError::Parse(s.into()))?; // 0
        let exponent = caps
            .get(5)
            .map_or(Ok(0), |m| i16::from_str(m.as_str()))
            .map_err(|_e| SciNumError::Parse(s.into()))?
            - frac_places as i16
            + truncated_places as i16; // -7
        // "6.971e-7" should be represented as (6971, -10)
        let num = Self {
            uncertainty,
            uncertainty_scale: 0,
            uncertainty_inf: false,
            uncertainty_nan: false,
            nan: false,
            inf: false,
            negative,
            exponent,
            significand,
        };
        if num.sf() > 16 {
            Ok(num.round_sf(16, RoundingMode::HalfUp))
        } else {
            Ok(num)
        }
    }
}

impl Num for SciDecimal {
    type FromStrRadixErr = SciNumError;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, SciNumError> {
        // For now, just make use of the BigDecimal implementation
        let dec = BigDecimal::from_str_radix(str, radix)
            .or(Err(SciNumError::Parse(format!("Couldn't parse {}", str))))?;
        Ok(dec.cast())
    }
}

#[macro_export]
macro_rules! sci {
    ($s:expr) => {
        <SciDecimal as std::str::FromStr>::from_str(stringify!($s)).unwrap()
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    //#[test]
    //fn debug() {
    //    let n = SciDecimal::new_with_uncertainty(20, 2, 0);
    //    assert_eq!(format!("{n:?}"), "SciDecimal { number: 20, uncertainty: 2 }");
    //}

    #[test]
    fn to_plain_string() {
        assert_eq!(
            SciDecimal::from_str("25e4").unwrap().to_plain_string(),
            "250000"
        );
        assert_eq!(
            SciDecimal::from_str("25(2)e4").unwrap().to_plain_string(),
            "250000(20000)"
        );
    }

    #[test]
    fn display() {
        // NaN and infinity should match the native `f64`
        assert_eq!(SciDecimal::NAN.to_string(), f64::NAN.to_string()); // "NaN"
        assert_eq!(SciDecimal::INFINITY.to_string(), f64::INFINITY.to_string()); // "inf"
        assert_eq!(
            SciDecimal::NEG_INFINITY.to_string(),
            f64::NEG_INFINITY.to_string()
        ); // "-inf"
        // As should +/- zero, as long as there's no uncertainty
        assert_eq!(SciDecimal::ZERO.to_string(), (0.0).to_string()); // "0"
        assert_eq!(SciDecimal::NEG_ZERO.to_string(), (-0.0).to_string()); // "-0"
        // Numbers with up to five places either side of the decimal point should
        // be displayed using normal notation
        // Integers should display without any decimal point at all
        assert_eq!(SciDecimal::new(20, 0).to_string(), "20");
        assert_eq!(SciDecimal::new(-20, 0).to_string(), "-20");
        assert_eq!(SciDecimal::new(99999, 0).to_string(), "99999");
        assert_eq!(SciDecimal::new(10000, 0).to_string(), "10000");
        assert_eq!(SciDecimal::new(1000, 0).to_string(), "1000");
        assert_eq!(SciDecimal::new(100, 0).to_string(), "100");
        assert_eq!(SciDecimal::new(10, 0).to_string(), "10");
        assert_eq!(SciDecimal::new(1, 0).to_string(), "1");
        assert_eq!(SciDecimal::new(1, -1).to_string(), "0.1");
        assert_eq!(SciDecimal::new(1, -2).to_string(), "0.01");
        assert_eq!(SciDecimal::new(1, -3).to_string(), "0.001");
        assert_eq!(SciDecimal::new(1, -4).to_string(), "0.0001");
        assert_eq!(SciDecimal::new(1, -5).to_string(), "0.00001");
        assert_eq!(sci!(0.00001).to_string(), "0.00001");
        assert_eq!(SciDecimal::new(325, -4).to_string(), "0.0325");
        assert_eq!(SciDecimal::new(-325, -4).to_string(), "-0.0325");
        assert_eq!(SciDecimal::new(85130, -3).to_string(), "85.130");
        assert_eq!(sci!(25691.29854).to_string(), "25691.29854");
        // If the maximum number of places (5) is exceeded, use scientific notation
        assert_eq!(SciDecimal::new(1295891, 0).to_string(), "1.295891e6");
        assert_eq!(SciDecimal::new(325, -6).to_string(), "3.25e-4"); // Not 0.000325
        assert_eq!(SciDecimal::new(-325, -6).to_string(), "-3.25e-4");
        assert_eq!(SciDecimal::new(8174036, 0).to_string(), "8.174036e6");
        assert_eq!(sci!(0.000000432).to_string(), "4.32e-7");
        // Importantly, explicit zeros should be treated as significant
        assert_eq!(SciDecimal::new(1295800, 0).to_string(), "1.295800e6");
        // Scientific notation should also be used if there are insignificant zeros
        // before the decimal point, even when the maximum number of places (5)
        // is not exceeded, so that the precision is indicated
        // 81700 with 3 sf = 8.17e4 = (817, 2) => "8.17e4"
        assert_eq!(SciDecimal::new(817, 2).to_string(), "8.17e4");
        // 81700 with 5 sf = 8.1700e4 = (81700, 0) => "81700"
        assert_eq!(SciDecimal::new(81700, 0).to_string(), "81700");

        // Check uncertainty formatting
        assert_eq!(
            SciDecimal::new_with_uncertainty(20, 2, 0).to_string(),
            "20(2)"
        );
        // TODO: More uncertainty display tests
    }

    #[test]
    fn from_str() {
        // Integer
        assert_eq!(SciDecimal::from_str("42").unwrap(), SciDecimal::new(42, 0));
        // Zero
        assert_eq!(SciDecimal::from_str("0").unwrap(), SciDecimal::ZERO);
        // Decimal
        assert_eq!(
            SciDecimal::from_str("0.0859").unwrap(),
            SciDecimal::new(859, -4)
        );
        // Decimal without integral part before decimal point
        assert_eq!(
            SciDecimal::from_str(".0859").unwrap(),
            SciDecimal::new(859, -4)
        );
        // Negative decimal
        assert_eq!(
            SciDecimal::from_str("-3.14").unwrap(),
            SciDecimal::new(-314, -2)
        );
        // Small number but not scientific notation
        assert_eq!(
            SciDecimal::from_str("0.0000000000000000000000000022250738585072").unwrap(),
            SciDecimal::new(22250738585072, -40)
        );
        // Scientific notation
        assert_eq!(
            SciDecimal::from_str("1.5e8").unwrap(),
            SciDecimal::new(15, 7)
        );
        // Scientific notation with negative exponent
        assert_eq!(
            SciDecimal::from_str("2e-5").unwrap(),
            SciDecimal::new(2, -5)
        );
        // Negative number with positive exponent
        assert_eq!(
            SciDecimal::from_str("-6.022e6").unwrap(),
            SciDecimal::new(-6022, 3)
        );
        // Large exponents
        assert_eq!(
            SciDecimal::from_str("1.5e18").unwrap(),
            SciDecimal::new(15, 17)
        );
        assert_eq!(
            SciDecimal::from_str("-6.022e23").unwrap(),
            SciDecimal::new(-6022, 20)
        );
        // Capital E for exponent
        assert_eq!(
            SciDecimal::from_str("1.5E8").unwrap(),
            SciDecimal::new(15, 7)
        );
        // 16 significant figures must always be fine
        assert_eq!(
            SciDecimal::from_str("0.5293040185492948").unwrap(),
            SciDecimal::new(5293040185492948, -16)
        );
        // Excess precision should be silently truncated to 16 sf
        // TODO: maybe in future should be rounded rather than truncated?
        assert_eq!(
            SciDecimal::from_str("0.529304018549294841").unwrap(),
            SciDecimal::new(5293040185492948, -16)
        );
        // Make sure incorrectly formatted strings fail
        assert!(SciDecimal::from_str("not a number").is_err());
        assert!(SciDecimal::from_str("x.482").is_err());
        assert!(SciDecimal::from_str("52.x").is_err());
        assert!(SciDecimal::from_str("-2.42F-4").is_err());
    }

    #[test]
    fn sci_macro() {
        // Integer
        assert_eq!(sci!(42), SciDecimal::new(42, 0));
        // Negative float
        assert_eq!(
            sci!(-3.14),
            SciDecimal::from_scientific_parts(-3, 0, 14, 0, 0)
        );
        // Scientific notation
        assert_eq!(sci!(1.5e8), SciDecimal::new(15, 7));
        // Scientific notation with large exponent
        assert_eq!(sci!(1.5e10), SciDecimal::new(15, 9));
        // Scientific notation with negative exponent
        assert_eq!(sci!(2e-5), SciDecimal::new(2, -5));
        // Negative number with positive exponent
        assert_eq!(sci!(-6.022e6), SciDecimal::new(-6022, 3));
        // Negative number with large exponent
        assert_eq!(sci!(-6.022e23), SciDecimal::new(-6022, 20));
        // Capital E for exponent
        assert_eq!(sci!(1.5E8), SciDecimal::new(15, 7));
    }
}
