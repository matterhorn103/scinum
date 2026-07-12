//! Runs a subset of the test suite of IBM's `decNumber` package.
//!
//! The original language-agnostic test files are stored in `dectest` and are parsed
//! at runtime.
//! Only the tests for `decimal64` (called `decDouble` in `decNumber`) have been
//! included.
//!
//! As `SciDecimal` has a much larger exponent range than `decimal64`, and no
//! subnormal numbers, some tests are excluded from running.
//! Tests to be excluded are indicated within the test function; the original test
//! files have not been changed.
//!
//! The documentation for the tests can be found at
//! https://speleotrove.com/decimal/dectest.pdf
//!
//! The lines in a testcase file may be:
//! - empty
//! - comments (start with --)
//! - directives (in the form `keyword: value`)
//! - tests (in the form `id operation operands -> result conditions`), with between
//!   one and three space-separated operands, and zero or more conditions
//!
//! All text is ASCII.
//!
//! Operands or results may be singly or doubly quoted.
//! In some cases, operands are deliberately written with incorrect syntax (to test
//! rejection).
//!
//! If an operand or result token contains a #, special syntax rules apply – see the
//! dectest docs.

use std::{collections::HashSet, str::FromStr};

use scinum::SciDecimal;

#[test]
fn dectest_base() {
    let mut total: usize = 0;
    let mut passed: usize = 0;
    let mut skipped: usize = 0;
    let mut failed: usize = 0;
    let mut exclusions: HashSet<String> = HashSet::new();
    exclusions.extend((301..=412).map(|n| format!("ddbas{}", n))); // Engineering notation tests
    let testfile = include_str!("dectest/ddBase.decTest");
    for line in testfile.lines() {
        // Each test has format:
        // ddbas130 toSci "0.000E-1"  -> '0.0000'
        // with a single operand, where the operation is either toSci or toEng,
        // and the operand/result may or may not be quotes
        if line.is_empty() || line.starts_with("--") {
            // Empty or comment line
            continue;
        } else if line.contains(":") {
            // Directive
            continue;
        } else {
            total += 1;
        }
        let mut split = line.split_ascii_whitespace();
        let id = split.next().unwrap();
        if exclusions.contains(id) {
            skipped += 1;
            continue;
        }
        let operation = split.next().unwrap();
        let operand = SciDecimal::from_str(split.next().unwrap()).unwrap_or(SciDecimal::NAN); // Tests expect NaN when parsing fails
        split.next();
        let result = split.next().unwrap();
        let eq = match operation.to_lowercase().as_str() {
            "tosci" => operand.to_string() == result,
            //"toeng" => operand.to_engineering_string() == result,
            _ => false,
        };
        if eq {
            passed += 1;
        } else {
            failed += 1;
            dbg!(line);
        };
    }
    println!("Total: {total} Passed: {passed} Skipped: {skipped} Failed: {failed}");
    assert_eq!(failed, 0);
}
