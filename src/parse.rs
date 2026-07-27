//! Methods for parsing floats with uncertainties from strings.

use std::num::NonZeroU8;

use lexical::{NumberFormatBuilder, ParseIntegerOptions};

use crate::SciNumError;

/// Parses a sign from the beginning of a slice of UTF-8 bytes, returning `true` if
/// the sign is negative and `false` if positive or a sign is absent.
pub(crate) fn parse_sign(s: &mut &[u8]) -> Result<bool, SciNumError> {
    const PLUS_SIGN: u8 = '+' as u8;
    const HYPHEN_MINUS: u8 = '-' as u8;
    const MINUS_SIGN: &str = "−"; // Three bytes
    const MINUS_FIRST: u8 = MINUS_SIGN.as_bytes()[0];
    match *s
        .first()
        .ok_or(SciNumError::Parse("Bytes slice is empty".into()))?
    {
        PLUS_SIGN | HYPHEN_MINUS => Ok(s
            .split_off_first()
            .expect("Already confirmed to have at least one item")
            == &HYPHEN_MINUS),
        MINUS_FIRST => {
            if &s[..3] == MINUS_SIGN.as_bytes() {
                _ = s.split_off(..3);
                Ok(true)
            } else {
                // Don't remove anything, assume it's some different valid string
                // (e.g. an infinity sign starts with the same continuation byte)
                Ok(false)
            }
        }
        _ => Ok(false),
    }
}

/// Parses a decimal number, returning up to 19 significant digits as a `u64` along
/// with the exponent of the number's quantum.
///
/// The exponent of a number formatted with scientific notation is _not_ parsed; parsing
/// stops at the first character that is not a digit or a decimal separator (which may
/// only occur once).
///
/// Both U+002E . FULL STOP and U+002C , COMMA are parsed as a decimal separator.
///
/// # Panics
///
/// Panics if the encoded number somehow has more than 2<sup>15</sup> significant
/// figures.
pub(crate) fn parse_decimal_19_places(s: &mut &[u8]) -> Result<(u64, i16), SciNumError> {
    // Grammar allows three options:
    //
    //     Decimal ::= ( Digits | Digits Separator Digits? | Digits? Separator Digits )
    //
    // where:
    //
    //     Digits ::= ( '_' | Digit )+ Digit ( '_' | Digit )+
    //     Digit  ::= [0-9]
    //
    // i.e. a valid decimal string has to have at least one digit either before or after
    // any decimal separator.
    const FULL_STOP: u8 = '.' as u8;
    const COMMA: u8 = ',' as u8;
    const UNDERSCORE: u8 = '_' as u8;
    let mut at_least_one_digit = false;
    let mut significant_places: usize = 0;
    let mut frac_places: usize = 0;
    let mut int_digits: Vec<u8> = Vec::new();
    // Stop at first non-digit or decimal separator
    loop {
        match s.first() {
            Some(&UNDERSCORE) => {
                // Digits separator, ignore (drop)
                _ = s.split_off_first();
                continue;
            }
            Some(0x31..=0x39) => {
                // Non-zero digit
                at_least_one_digit = true;
                significant_places += 1;
                int_digits.push(*s.split_off_first().unwrap());
                continue;
            }
            Some(0x30) => {
                // Zero
                at_least_one_digit = true;
                if significant_places != 0 {
                    significant_places += 1;
                    int_digits.push(*s.split_off_first().unwrap());
                } else {
                    _ = s.split_off_first();
                    continue;
                }
            }
            Some(&FULL_STOP) | Some(&COMMA) => {
                // Decimal separator, stop
                _ = s.split_off_first();
                break;
            }
            None => {
                // Reached the end of the string before any decimal separator,
                // implying that the number is an integer
                // Confirm that we had at least one actual digit at some point
                if !at_least_one_digit {
                    return Err(SciNumError::Parse(
                        "Decimal string contains no digits".into(),
                    ));
                };
                break;
            }
            _ => break, // Any other character, stop
        }
    }
    // As it stands the exponent is either zero (for a non-zero integer part) or it
    // will end up being negative anyway
    let mut exp: i16 = 0;
    let mut frac_digits: Vec<u8> = Vec::new();
    // Stop at first non-digit, significance tracking continues
    loop {
        match s.first() {
            Some(&UNDERSCORE) => {
                // Digits separator, ignore (drop)
                _ = s.split_off_first();
                continue;
            }
            Some(0x31..=0x39) => {
                // Non-zero digit
                at_least_one_digit = true;
                exp -= 1;
                significant_places += 1;
                frac_places += 1;
                frac_digits.push(*s.split_off_first().unwrap());
            }
            Some(0x30) => {
                // Zero
                at_least_one_digit = true;
                if significant_places != 0 {
                    exp -= 1;
                    significant_places += 1;
                    frac_places += 1;
                    frac_digits.push(*s.split_off_first().unwrap());
                } else {
                    // Every fractional digit decreases the exponent, even if it's
                    // not significant
                    exp -= 1;
                    _ = s.split_off_first();
                    continue;
                }
            }
            _ => {
                // Either the end of the string or the end of the decimal part
                // Confirm that we had at least one actual digit at some point
                if !at_least_one_digit {
                    return Err(SciNumError::Parse(
                        "Decimal string contains no digits".into(),
                    ));
                }
                break;
            }
        }
    }
    dbg!(&int_digits);
    let int: u64 = if !int_digits.is_empty() {
        lexical::parse(int_digits).expect("Already checked each character explicitly")
    } else {
        0
    };
    let mut frac: u64 = if !frac_digits.is_empty() {
        lexical::parse(frac_digits).expect("Already checked each character explicitly")
    } else {
        0
    };
    if significant_places > 19 {
        let excess_places = significant_places - 19;
        frac /= 10_u64.pow(
            excess_places
                .try_into()
                .expect("Number of places parsed into a u64 must fit into a u32"),
        );
        frac_places -= excess_places;
        exp += i16::try_from(excess_places).expect("Will never truncate more than 2^15 places");
    };
    let significand = if frac_places != 0 {
        int * 10_u64.pow(
            frac_places
                .try_into()
                .expect("Number of places parsed into a u64 must fit into a u32"),
        ) + frac
    } else {
        int
    };
    Ok((significand, exp))
}

/// Parses an exponent, including any sign, and returns it as a signed integer.
pub(crate) fn parse_exponent(s: &mut &[u8]) -> Result<i16, SciNumError> {
    let neg = parse_sign(s)?;
    let (exp, digits): (i16, usize) = lexical::parse_partial(*s)
        .or(Err(SciNumError::Parse("Failed to parse exponent".into())))?;
    _ = s.split_off(..digits);
    if neg { Ok(-exp) } else { Ok(exp) }
}

/// Parses an uncertainty in shorthand form and returns the significant digits as a
/// `u32`.
///
/// Underscores are permitted as digit separators.
pub(crate) fn parse_short_uncertainty(s: &mut &[u8]) -> Result<u32, SciNumError> {
    if *s
        .split_off_first()
        .ok_or(SciNumError::Parse("Bytes slice is empty".into()))?
        != '(' as u8
    {
        return Err(SciNumError::Parse(
            "Uncertainty is missing opening parenthesis".into(),
        ));
    }
    // Allow underscores as digit separators, allow anywhere within string of digits
    const FORMAT: u128 = NumberFormatBuilder::new()
        .digit_separator(NonZeroU8::new(b'_'))
        .internal_digit_separator(true)
        .leading_digit_separator(true)
        .trailing_digit_separator(true)
        .consecutive_digit_separator(true)
        .build_strict();
    // Just use default options though
    const OPTIONS: ParseIntegerOptions = ParseIntegerOptions::new();
    let (uncert, digits): (u32, usize) =
        lexical::parse_partial_with_options::<u32, _, FORMAT>(*s, &OPTIONS).or(Err(
            SciNumError::Parse("Failed to parse uncertainty".into()),
        ))?;
    _ = s.split_off(..digits);
    // Must have a closing parenthesis too
    let next = s.split_off_first().ok_or(SciNumError::Parse(
        "Uncertainty is missing closing parenthesis".into(),
    ))?;
    if *next != ')' as u8 {
        Err(SciNumError::Parse(
            "Uncertainty is missing closing parenthesis".into(),
        ))
    } else {
        Ok(uncert)
    }
}

/// Returns whether a slice of UTF-8 bytes is the correct form for an infinity.
///
/// Unlike the other parsing functions in this module, the whole slice is compared,
/// so a correct infinity followed by trailing characters is considered invalid.
pub(crate) fn parse_inf(s: &mut &[u8]) -> Result<(), SciNumError> {
    // Apply the same trick as in `core` https://doc.rust-lang.org/src/core/num/imp/dec2flt/parse.rs.html
    // Max valid length is 8 so work on u64
    let mut register: u64;

    // All valid strings are either of length 8 or 3.
    if s.len() == 8 {
        register = s
            .iter()
            .enumerate()
            .map(|x| (*(x.1) as u64) << (x.0 * 8))
            .fold(0_u64, |acc, x| acc | x);
    } else if s.len() == 3 {
        let a = s[0] as u64;
        let b = s[1] as u64;
        let c = s[2] as u64;
        register = (c << 16) | (b << 8) | a;
    } else {
        return Err(SciNumError::Parse(
            "An infinity string should always consist of either 3 or 8 UTF-8 bytes".into(),
        ));
    }

    // u64 values corresponding to relevant cases
    const INF_3: u64 = 0x464E49; // "INF"
    const INF_8: u64 = 0x5954494E49464E49; // "INFINITY"
    const INF_SYMBOL: u64 = 0x9E88E2; // "∞"

    // First check for infinity symbol
    if register == INF_SYMBOL {
        return Ok(());
    }

    // Clear out the bits which turn ASCII uppercase characters into
    // lowercase characters. The resulting string is all uppercase.
    register &= 0xDFDFDFDFDFDFDFDF;
    match register {
        INF_3 | INF_8 => Ok(()),
        _ => Err(SciNumError::Parse("Invalid infinity string".into())),
    }
}

/// Parses a `NaN`, returning whether the `NaN` is signalling as well as any
/// diagnostic payload.
pub(crate) fn parse_nan(s: &mut &[u8]) -> Result<(bool, Option<u64>), SciNumError> {
    let mut signalling: bool = false;
    match s.first() {
        Some(0x53) | Some(0x73) => {
            signalling = true;
            _ = s.split_off_first();
        }
        _ => (),
    }
    // NaN string itself must always be three characters
    let nan = str::from_utf8(s.split_off(..3).ok_or(SciNumError::Parse(
        "A NaN string should always consist of at least three UTF-8 bytes".into(),
    ))?)
    .or(Err(SciNumError::Parse("Invalid NaN string".into())))?;
    if nan.to_lowercase() != "nan" {
        return Err(SciNumError::Parse("Invalid NaN string".into()));
    }
    let payload: Option<u64> = if s.is_empty() {
        None
    } else {
        Some(lexical::parse(s).or(Err(SciNumError::Parse("Invalid NaN payload".into())))?)
    };
    Ok((signalling, payload))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sign() {
        let mut s = "-3".as_bytes();
        let sign = parse_sign(&mut s).unwrap();
        assert_eq!(sign, true);
        // Confirm that s has been modified in-place
        assert_eq!(s, "3".as_bytes());
        // Test a variety of strings
        assert_eq!(parse_sign(&mut "7.0".as_bytes()).unwrap(), false);
        assert_eq!(parse_sign(&mut "+7.0".as_bytes()).unwrap(), false);
        assert_eq!(parse_sign(&mut "-7.0".as_bytes()).unwrap(), true);
        // Anything other than a negative symbol is implied positive
        assert_eq!(parse_sign(&mut "a7.0".as_bytes()).unwrap(), false);
    }

    #[test]
    fn test_parse_exp() {
        // Exponent will always either be at end or before uncertainty
        let mut s = "12".as_bytes();
        assert_eq!(parse_exponent(&mut s).unwrap(), 12);
        assert_eq!(s, &[]);
        let mut s = "12+/-3".as_bytes();
        assert_eq!(parse_exponent(&mut s).unwrap(), 12);
        assert_eq!(s, "+/-3".as_bytes());
        // Negative exponent
        let mut s = "-3".as_bytes();
        assert_eq!(parse_exponent(&mut s).unwrap(), -3);
        assert_eq!(s, &[]);
        let mut s = "-3+/-42e-5".as_bytes();
        assert_eq!(parse_exponent(&mut s).unwrap(), -3);
        assert_eq!(s, "+/-42e-5".as_bytes());
        // Leading insignificant zero
        assert_eq!(parse_exponent(&mut "012".as_bytes()).unwrap(), 12);
    }

    #[test]
    fn test_parse_decimal() {
        let mut s = "12.59".as_bytes();
        // Decimal
        assert_eq!(parse_decimal_19_places(&mut s).unwrap(), (1259, -2));
        assert_eq!(s, &[]);
        // Decimal with zero integral part
        let mut s = ".0859".as_bytes();
        assert_eq!(parse_decimal_19_places(&mut s).unwrap(), (859, -4));
        assert_eq!(s, &[]);
        // Integer with decimal point but no fractional part after decimal point
        assert_eq!(
            parse_decimal_19_places(&mut "17.".as_bytes()).unwrap(),
            (17, 0)
        );
        // With a leading zero
        assert_eq!(
            parse_decimal_19_places(&mut "017.3".as_bytes()).unwrap(),
            (173, -1)
        );
        // Small number but not scientific notation
        assert_eq!(
            parse_decimal_19_places(&mut "0.0000000000000000000000000022250738585072".as_bytes())
                .unwrap(),
            (22250738585072, -40)
        );
    }

    #[test]
    fn test_parse_inf() {
        assert!(parse_inf(&mut "inf".as_bytes()).is_ok());
        assert!(parse_inf(&mut "Inf".as_bytes()).is_ok());
        assert!(parse_inf(&mut "INF".as_bytes()).is_ok());
        assert!(parse_inf(&mut "infinity".as_bytes()).is_ok());
        assert!(parse_inf(&mut "Infinity".as_bytes()).is_ok());
        assert!(parse_inf(&mut "INFINITY".as_bytes()).is_ok());
        assert!(parse_inf(&mut "∞".as_bytes()).is_ok());
        assert!(parse_inf(&mut "enf".as_bytes()).is_err());
        assert!(parse_inf(&mut "a∞".as_bytes()).is_err());
    }
}
