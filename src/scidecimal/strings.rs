//! String-related methods and trait implementations for [`SciDecimal`].

use core::fmt;
use std::str::FromStr;

use bigdecimal::BigDecimal;
use num_traits::{Num, Zero};

use crate::parse::*;
use crate::{RoundingMode, SciDecimal, SciNum, SciNumError, scicast::SciCast};

impl FromStr for SciDecimal {
    type Err = SciNumError;

    /// Parses a string and attempts to create a corresponding `SciDecimal`.
    ///
    /// A correctly formed string, whether in positional or scientific/engineering
    /// notation, will *always* return a `SciDecimal`:
    ///
    /// - Excess precision is rounded to 16 significant figures (according to
    ///   [`RoundingMode::HalfEven`]).
    ///
    /// - If the absolute magnitude of the number is too *large* to be represented, an
    ///   infinity with the appropriate sign is returned.
    ///
    /// - If the absolute magnitude of the number is too *small* to be represented, a
    ///   zero with the appropriate sign is returned.
    ///
    /// In this way the behaviour is like that of [`SciCast::cast`], and indeed this
    /// method is used to effect the casts from several types.
    ///
    /// Fails if the string cannot be parsed at all.
    ///
    /// # Special values
    ///
    /// `0` and `+0` are parsed to positive zero, `-0` and `−0` to negative
    /// zero.
    ///
    /// Several variations of infinities and `NaN` are permitted and they are parsed
    /// case-insensitively.
    ///
    /// `inf`, `infinity`, and `∞` are parsed to positive or negative infinity
    /// according to any preceding sign.
    ///
    /// All variations of `NaN` are parsed case-insensitively to `NaN`, with any sign
    /// sign ignored.
    ///
    /// # Valid syntax
    ///
    /// Any string valid for [Rust's native float types](https://doc.rust-lang.org/std/primitive.f32.html#impl-FromStr-for-f32)
    /// or for decimal floats per the [IBM Decimal Arithmetic Specification](https://speleotrove.com/decimal/daconvs.html)
    /// will be parsed correctly.
    ///
    /// The parser also supports some additional syntax extensions, both for convenience
    /// and to support uncertainties.
    ///
    /// ## Notation of uncertainty
    ///
    /// Uncertainties can be written using one of two notations:
    ///
    /// 1. As a suffixed second number separated from the first by either `"+/-"` or
    ///    `"±"` (U+00B1 ± PLUS-MINUS SIGN) e.g. `"47.2+/-0.6"` or `"7.29e5±2.1e4"`
    /// 2. Using in-line shorthand notation in parentheses e.g. `"47.2(6)"` to mean the
    ///    same as `"47.2+/-0.6"` or `"7.29(21)e5"` to mean `"7.29e5±2.1e4"`
    ///
    /// The former notation supports uncertainties with a different quantum ("precision"
    /// as used in scientific fields) to the number e.g. `"47.2284719503+/-0.6"`, as
    /// well as infinity and `NaN` uncertainty e.g. `"47.2+/-NaN"`.
    ///
    /// Note that the uncertainty of infinities and `NaN` is defined to be `NaN` and a
    /// string encoding of these values with any uncertainty is considered invalid, even
    /// if the notated uncertainty is `NaN` e.g. inputs of `"Infinity+/-4e7"`,
    /// `"-inf+/-NaN"`, and `"NaN±NaN"` will all return an error.
    ///
    /// ## Valid characters
    ///
    /// As usual for Rust's string primitives, the input string is expected to be valid
    /// UTF-8.
    ///
    /// `scinum` supports the use of some additional characters in comparison to most
    /// floating point implementations:
    ///
    /// - A decimal comma may be used instead of a decimal point
    /// - U+2212 − MINUS SIGN is parsed as a minus sign, just like U+002D - HYPHEN-MINUS
    /// - U+221E ∞ INFINITY is parsed as an infinity
    /// - U+00B1 ± PLUS-MINUS SIGN can be used for (full-form) uncertainties
    ///
    /// Note that none of these characters are ever used in `SciDecimal`'s _output_
    /// unless explicitly requested, to avoid compatibility issues.
    ///
    /// Like Rust's built-in integer types, but unlike the built-in floats, underscores
    /// are permitted within runs of digits to separate them into groups; they are
    /// simply ignored when parsing. At least one digit 0–9 is required.
    ///
    /// ## Grammar
    ///
    /// Strings are parsed **case-insensitively** according to the following formal
    /// grammar:
    ///
    /// ```
    /// NumericString  ::= Sign? ( Inf | NaN | Number | UncertainNumber )
    /// UncertainNumber ::= Decimal ShortUncert Exp? | Number FullUncert
    /// Number ::= Decimal Exp?
    /// Decimal ::= ( Digits | Digits Separator Digits? | Digits? Separator Digits )
    /// ShortUncert ::= '(' Digits ')'
    /// FullUncert ::= ( '+/-' | '±' ) ( Inf | NaN | Number )
    /// Exp    ::= 'e' Sign? Digits
    /// Digits ::= ( '_' | Digit )+ Digit ( '_' | Digit )+
    /// Digit  ::= [0-9]
    /// Sign   ::= [+-−]
    /// Separator ::= [.,]
    /// Inf ::= 'infinity' | 'inf' | '∞'
    /// NaN ::= 's'? 'nan' Digit*
    /// ```
    ///
    /// ### Exceptions
    ///
    /// Current known exceptions to the above grammar:
    ///
    /// 1. Only one set of `Digits` is required to contain an actual digit 0–9, and
    ///    the other may simply be an underscore e.g. `"_.5"` is parsed as `".5"`
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // The grammar for `f32` allows:
        //
        // Float  ::= Sign? ( 'inf' | 'infinity' | 'nan' | Number )
        // Number ::= ( Digit+ |
        //              Digit+ '.' Digit* |
        //              Digit* '.' Digit+ ) Exp?
        // Exp    ::= 'e' Sign? Digit+
        // Sign   ::= [+-]
        // Digit  ::= [0-9]
        //
        // The IBM decimal spec includes a formal grammar for numeric strings, which,
        // adapting the syntax to match the Rust one, looks like:
        //
        // sign ::= [+-]
        // digit ::= [0-9]
        // indicator ::= 'e'
        // digits ::= digit+
        // decimal-part ::= digits '.' digits? | '.'? digits
        // exponent-part ::= indicator sign? digits
        // infinity ::= 'infinity' | 'inf'
        // nan ::= 'nan' digits? | 'snan' digits?
        // numeric-value ::= decimal-part exponent-part? | infinity
        // numeric-string ::= sign? numeric-value | sign? nan
        //
        // We support essentially the same as f32 but with uncertainties on top, plus a
        // few extra characters (see the docstring for the supported grammar).
        if s.is_empty() {
            return Err(SciNumError::Parse("Attempted to parse empty string".into()));
        }
        let mut bytes = s.as_bytes();
        let sign = parse_sign(&mut bytes)?;
        match bytes.first() {
            Some(0x53) | Some(0x73) | Some(0x4E) | Some(0x6E) => {
                // S or s or N or n (sNaN or NaN)
                return match parse_nan(&mut bytes) {
                    Ok(_) => Ok(SciDecimal::NAN), // We don't support signalling NaN or payloads, so we drop them
                    Err(_) => Err(SciNumError::Parse(s.into())),
                };
            }
            Some(0x49) | Some(0x69) | Some(0xE2) => {
                // I or i or ∞
                return parse_inf(&mut bytes).map(|_| {
                    if sign {
                        SciDecimal::NEG_INFINITY
                    } else {
                        SciDecimal::INFINITY
                    }
                });
            }
            None => return Err(SciNumError::Parse(s.into())),
            _ => (),
        }
        // Not a special value so try normal parsing
        let (significand, mut exponent) = parse_decimal_19_places(&mut bytes)?;
        let uncert_digits: Option<u32> = if bytes.first() == Some(&0x28) {
            // ( for short-form uncertainty
            Some(parse_short_uncertainty(&mut bytes)?)
        } else {
            None
        };
        // Check for scientific notation
        if bytes.first() == Some(&0x45) || bytes.first() == Some(&0x65) {
            // e or E
            let _ = bytes.split_off_first();
            let e = parse_exponent(&mut bytes)?;
            exponent += e;
        }
        // Now we have only three remaining possibilities:
        // 1. The whole string has been parsed
        // 2. There remains a full-form uncertainty to parse
        // 3. There remains something else, which means the string was invalid
        // In any case we have definitely finished parsing the number
        let mut num = {
            SciDecimal {
                uncertainty: uncert_digits.unwrap_or_default(),
                uncertainty_scale: 0,
                flags: sign as u8,
                exponent,
                significand,
            }
        };
        let uncert = match bytes.first() {
            Some(0x2B) => {
                // +/- for long-form uncertainty
                _ = bytes.split_off(..3);
                Some(SciDecimal::from_str(
                    str::from_utf8(bytes).expect("Was originally a str"),
                )?)
            }
            Some(0xC2) => {
                // ± for long-form uncertainty
                _ = bytes.split_off(..2);
                Some(SciDecimal::from_str(
                    str::from_utf8(bytes).expect("Was originally a str"),
                )?)
            }
            Some(_) => return Err(SciNumError::Parse(s.into())), // Remaining characters so string invalid
            None => None,
        };
        if let Some(u) = uncert {
            num = num.with_uncertainty(u);
        };
        // Might have up 19 sf of precision
        if num.sf() > 16 {
            Ok(num.round_sf(16, RoundingMode::HalfEven))
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

impl fmt::Display for SciDecimal {
    /// Formats the number using the given formatter.
    ///
    /// Scientific notation is used if:
    /// - positional notation would require more than five leading zeros in the
    ///   fractional part (e.g. `"0.000005"` is fine, but `"5e-7"` would be returned
    ///   rather than `"0.0000005"`)
    /// - positional notation would result in insignificant zeros being shown before
    ///   the decimal point, erroneously implying a greater number of significant
    ///   figures than the number possesses (e.g. `"3000"` implies 4 significant
    ///   figures, so while `3000e0` returns `"3000"`, `3e3` returns `"3e3"`)
    ///
    /// /// # Example
    ///
    /// ```
    /// # use scinum::SciDecimal;
    /// #
    /// assert_eq!(SciDecimal::new(325, -4).to_string(), "0.0325");
    /// assert_eq!(SciDecimal::new(85130, -3).to_string(), "85.130");
    /// assert_eq!(SciDecimal::new(325, -9).to_string(), "3.25e-7");
    /// assert_eq!(
    ///     SciDecimal::from_scientific_parts(3, 25, 0, 2, -7).to_string(),
    ///     "3.25e-7",
    /// );
    /// assert_eq!(SciDecimal::new(8174036, 0).to_string(), "8174036");
    /// // 81700 with 3 sf
    /// assert_eq!(SciDecimal::new(817, 2).to_string(), "8.17e4");
    /// // 81700 with 5 sf
    /// assert_eq!(SciDecimal::new(81700, 0).to_string(), "81700");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Handle NaN
        if self.is_nan() {
            return write!(f, "NaN");
        }
        // Handle infinities
        if self.inf_bit() {
            return write!(
                f,
                "{}inf",
                self.sign_bit()
                    .then_some(String::from("-"))
                    .unwrap_or_default() // i.e. no extra character if not negative
            );
        }
        if self.precision_most_significant_fig() >= -6 && self.precision() <= 0 {
            return write!(f, "{}", self.to_positional_string());
        // Otherwise, use scientific notation
        } else {
            return write!(f, "{}", self.to_scientific_string());
        }
    }
}

impl SciDecimal {
    pub fn to_scientific_string(&self) -> String {
        // Handle NaN
        if self.is_nan() {
            return String::from("NaN");
        }
        // Get sign character
        let sign = if self.sign_bit() {
            String::from("-")
        } else {
            String::new()
        };
        // Handle infinities
        if self.inf_bit() {
            return format!("{}inf", sign);
        }
        let uncertainty = if self.is_exact() {
            String::new()
        } else {
            format!("({})", self.uncertainty)
        };
        let (int, frac, _, places, exp) = self.to_scientific_parts().unwrap();
        // Fractional part might not have any places at all (e.g. 2e6)
        if frac == 0 {
            format!("{int}{uncertainty}e{exp}")
        } else {
            let p = places as usize;
            format!("{int}.{frac:p$}{uncertainty}e{exp}")
        }
    }

    pub fn to_positional_string(&self) -> String {
        // Handle NaN
        if self.is_nan() {
            return String::from("NaN");
        }
        // Get sign character
        let sign = if self.sign_bit() {
            String::from("-")
        } else {
            String::new()
        };
        // Handle infinities
        if self.inf_bit() {
            return format!("{sign}inf");
        }
        // Handle zeros
        if self.is_zero() {
            // TODO Have this display the uncertainty properly once they can be
            // displayed with +/-
            return format!("{sign}0");
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

    #[test]
    fn from_str_integer() {
        // Integer
        assert_eq!(SciDecimal::from_str("42").unwrap(), SciDecimal::new(42, 0));
        // Zero
        assert_eq!(SciDecimal::from_str("0").unwrap(), SciDecimal::ZERO);
    }

    #[test]
    fn from_str_fraction() {
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
        // Integer with decimal point but no fractional part after decimal point
        assert_eq!(
            SciDecimal::from_str("017.").unwrap(),
            SciDecimal::new(17, 0)
        );
        // Negative decimal
        assert_eq!(
            SciDecimal::from_str("-3.14").unwrap(),
            SciDecimal::new(-314, -2)
        );
        // Positive decimal but with explicit plus sign
        assert_eq!(
            SciDecimal::from_str("+3.14").unwrap(),
            SciDecimal::new(314, -2)
        );
        // Small number but not scientific notation
        assert_eq!(
            SciDecimal::from_str("0.0000000000000000000000000022250738585072").unwrap(),
            SciDecimal::new(22250738585072, -40)
        );
    }

    #[test]
    fn from_str_scientific() {
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
        // Scientific notation with positive exponent with explicit plus sign
        assert_eq!(SciDecimal::from_str("2e+5").unwrap(), SciDecimal::new(2, 5));
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
    }

    #[test]
    fn from_str_rounding() {
        // 16 significant figures must always be fine
        assert_eq!(
            SciDecimal::from_str("0.5293040185492948").unwrap(),
            SciDecimal::new(5293040185492948, -16)
        );
        // Excess precision should be rounded to 16 sf
        assert_eq!(
            SciDecimal::from_str("0.529304018549294841").unwrap(),
            SciDecimal::new(5293040185492948, -16)
        );
        // Rounding should use HalfToEven strategy
        assert_eq!(
            SciDecimal::from_str("0.52930401854929485").unwrap(),
            SciDecimal::new(5293040185492948, -16)
        );
        assert_eq!(
            SciDecimal::from_str("0.52930401854929475").unwrap(),
            SciDecimal::new(5293040185492948, -16)
        );
    }

    #[test]
    fn from_str_special() {
        // NaNs
        assert!(SciDecimal::from_str("NaN").unwrap().is_nan());
        assert!(SciDecimal::from_str("nan").unwrap().is_nan());
        // Signalling NaN
        assert!(SciDecimal::from_str("sNaN").unwrap().is_nan());
        // Infinities, all cases, all signs
        assert_eq!(SciDecimal::from_str("inf").unwrap(), SciDecimal::INFINITY);
        assert_eq!(SciDecimal::from_str("+inf").unwrap(), SciDecimal::INFINITY);
        assert_eq!(
            SciDecimal::from_str("-inf").unwrap(),
            SciDecimal::NEG_INFINITY
        );
        assert_eq!(SciDecimal::from_str("Inf").unwrap(), SciDecimal::INFINITY);
        assert_eq!(SciDecimal::from_str("+Inf").unwrap(), SciDecimal::INFINITY);
        assert_eq!(
            SciDecimal::from_str("-Inf").unwrap(),
            SciDecimal::NEG_INFINITY
        );
        assert_eq!(
            SciDecimal::from_str("infinity").unwrap(),
            SciDecimal::INFINITY
        );
        assert_eq!(
            SciDecimal::from_str("+infinity").unwrap(),
            SciDecimal::INFINITY
        );
        assert_eq!(
            SciDecimal::from_str("-infinity").unwrap(),
            SciDecimal::NEG_INFINITY
        );
        assert_eq!(
            SciDecimal::from_str("Infinity").unwrap(),
            SciDecimal::INFINITY
        );
        assert_eq!(
            SciDecimal::from_str("+Infinity").unwrap(),
            SciDecimal::INFINITY
        );
        assert_eq!(
            SciDecimal::from_str("-Infinity").unwrap(),
            SciDecimal::NEG_INFINITY
        );
    }

    #[test]
    fn from_str_diagnostic_nan() {
        // NaN with diagnostic info
        assert!(SciDecimal::from_str("NaN8275").unwrap().is_nan());
        // Signalling NaN with diagnostic info
        assert!(SciDecimal::from_str("sNaN8275").unwrap().is_nan());
    }

    #[test]
    fn from_str_additional_chars() {
        // Decimal comma is accepted
        assert_eq!(
            SciDecimal::from_str("3,14").unwrap(),
            SciDecimal::new(314, -2)
        );
        // Underscore separator is accepted in all parts of the number
        // In integral part
        assert_eq!(
            SciDecimal::from_str("9_876_543.21").unwrap(),
            SciDecimal::new(987654321, -2)
        );
        // In fractional part
        assert_eq!(
            SciDecimal::from_str("9.876_543_21").unwrap(),
            SciDecimal::new(987654321, -8)
        );
        // In shorthand uncertainty
        assert_eq!(
            SciDecimal::from_str("9.876_54(3_21)").unwrap(),
            SciDecimal::new_with_uncertainty(987654, 321, -5)
        );
        // Unicode symbol for infinity works
        assert_eq!(SciDecimal::from_str("∞").unwrap(), SciDecimal::INFINITY);
        assert_eq!(SciDecimal::from_str("+∞").unwrap(), SciDecimal::INFINITY);
        assert_eq!(
            SciDecimal::from_str("-∞").unwrap(),
            SciDecimal::NEG_INFINITY
        );
        // Negative integer with Unicode minus
        assert_eq!(
            SciDecimal::from_str("−42").unwrap(),
            SciDecimal::new(-42, 0)
        );
        // Negative decimal with Unicode minus
        assert_eq!(
            SciDecimal::from_str("−3.14").unwrap(),
            SciDecimal::new(-314, -2)
        );
        // Scientific notation with negative exponent with Unicode minus sign
        assert_eq!(
            SciDecimal::from_str("2e−5").unwrap(),
            SciDecimal::new(2, -5)
        );
        // Infinity with Unicode minus
        assert_eq!(
            SciDecimal::from_str("−inf").unwrap(),
            SciDecimal::NEG_INFINITY
        );
        assert_eq!(
            SciDecimal::from_str("−Inf").unwrap(),
            SciDecimal::NEG_INFINITY
        );
        assert_eq!(
            SciDecimal::from_str("−infinity").unwrap(),
            SciDecimal::NEG_INFINITY
        );
        assert_eq!(
            SciDecimal::from_str("−Infinity").unwrap(),
            SciDecimal::NEG_INFINITY
        );
        assert_eq!(
            SciDecimal::from_str("−∞").unwrap(),
            SciDecimal::NEG_INFINITY
        );
    }

    #[test]
    fn from_str_malformed() {
        // Make sure incorrectly formatted strings fail
        assert!(SciDecimal::from_str("").is_err());
        assert!(SciDecimal::from_str("-").is_err());
        assert!(SciDecimal::from_str(".").is_err());
        assert!(SciDecimal::from_str("-.").is_err());
        assert!(SciDecimal::from_str("_").is_err());
        // These pass even though they don't match the formal grammar, because
        // they still contain at least one digit
        //assert!(SciDecimal::from_str("_.5").is_err());
        //assert!(SciDecimal::from_str("5._").is_err());
        assert!(SciDecimal::from_str("-_").is_err());
        assert!(SciDecimal::from_str("not a number").is_err());
        assert!(SciDecimal::from_str("x.482").is_err());
        assert!(SciDecimal::from_str("52.x").is_err());
        assert!(SciDecimal::from_str("-2.42F-4").is_err());
        assert!(SciDecimal::from_str("NaNInf").is_err());
        assert!(SciDecimal::from_str("(5)").is_err());
        assert!(SciDecimal::from_str("e10").is_err());
        assert!(SciDecimal::from_str("-inf5").is_err());
    }

    #[test]
    fn to_positional_string() {
        assert_eq!(
            SciDecimal::from_str("25e4").unwrap().to_positional_string(),
            "250000"
        );
        assert_eq!(
            SciDecimal::from_str("25(2)e4")
                .unwrap()
                .to_positional_string(),
            "250000(20000)"
        );
    }
}
/*
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
*/
