//! The `scinum` crate is designed to provide numeric types appropriate for scientific applications.
//!
//! The numeric types in `scinum` implement the [`SciNum`] trait, meaning that they:
//!
//! 1. Have an associated uncertainty that is tracked correctly across arithmetic operations
//! 2. Provide a complete set of rounding methods (to _n_ significant figures, decimal places etc.) with all rounding modes including "traditional" rounding
//!
//! In addition, the types in `scinum` offer the following:
//!
//! 3. IEEE 754 floating point arithmetic including signed zeros, infinities, and NaN
//! 4. Operator overloading for common arithmetic operations
//! 5. Implementations of `num_traits::Num` and `num_traits::Float`, providing further mathematical functions
//! 6. Appropriate string representations, in scientific notation and including the uncertainty, via `fmt::Display`
//!
//! ## Available types
//!
//! The crate provides two numeric types: [`SciDecimal`] and [`SciFloat`].
//! Both are implementations of [IEEE 754 floating point arithmetic](https://en.wikipedia.org/wiki/IEEE_754); `SciDecimal` is a _decimal_ float (effectively `decimal64` + an uncertainty), while `SciFloat` is a more traditional _binary_ float (`binary64` + an uncertainty).
//!
//! `SciFloat` simply wraps a pair of `f64` values – a number and an uncertainty.
//! It is intended as a drop-in replacement for `f64` that trades a doubled size for the convenience of uncertainty arithmetic, while maintaining a performance as close to a raw `f64` as possible.
//!
//! `SciDecimal` is the real star of `scinum`.
//! It is an implementation of the IEEE 754 specification for `decimal64`, with added uncertainty, for an overall width of 128 bits (the same as `SciFloat`).
//! All values representable by `f64`/`binary64` and `decimal64` can be represented by `SciDecimal` with no loss of precision.
//!
//! Decimal floating point has a number of advantages over the traditional binary floating point, as outlined nicely by the [docs for Python's `decimal` module](https://docs.python.org/3/library/decimal.html).
//! Primarily, unlike binary floats, they have the advantage of being able to represent and manipulate base-10 decimal numbers exactly, avoiding [counterintuitive arithmetic results, incorrect rounding behaviour, and accumulating rounding errors](https://en.wikipedia.org/wiki/Floating-point_arithmetic#Accuracy_problems).
//! What also makes decimal floats particularly appropriate for a `SciNum` type is that they have significant figures in a meaningful way, while binary floats do not.
//! `binary64` cannot distinguish between `12500`, with 5 significant figures, and `1.25e4`, with 3 significant figures, nor can it express the greater precision of `0.6300` vs `0.63`, and `1.30 * 1.20` gives `1.56` rather than `1.5600`.
//! `decimal64` can, so when using `SciDecimal`, the precision can be specified appropriately, and significant figures are maintained across calculations, just as scientists are taught they should be.
//! Unless performance is critical, the user is thus encouraged to use `SciDecimal`.
//!
//! `SciDecimal` is designed with performance in mind and the goal is for it to cost as little as possible to choose it over `SciFloat`.
//! However, it is at the end of the day a _software_ implementation of floating point arithmetic, whereas most modern CPUs offer _hardware_ floating point units for optimized calculations with `f64`, so there will inevitably be some cost.
//! In many applications, that cost is worth it.
//!
//! Write code generic over the `SciNum` trait to allow interchangeable use of `SciDecimal` and `SciFloat`.
//! As both types are `Float`, either can be used anywhere that is generic over any float type.
//!
//! `From` is implemented for lossless conversion to `SciDecimal` from `f64`.
//! Conversion _to_ `f64` is potentially lossy and so only `TryFrom` is implemented; `SciDecimal::to_f64()` is provided for lossy, saturating conversion.
//! `From` is implemented for conversion between `SciFloat` and `f64` in both directions, as dropping the uncertainty is considered entirely expected by the user and doesn't qualify the conversion as lossy.
//!
//! `TryFrom` is also implemented for/with `rust_decimal::Decimal` and `bigdecimal::BigDecimal`; the different precisions and exponent ranges and those formats' lack of support for special float values (`−0`, `±∞`, `NaN`) makes conversion potentially lossy or impossible.
//!
//! Finally, a `sci!()` macro is provided for convenient, literal-like creation of `SciDecimal` instances.
//!
//! ## Usage
//!
//! An exact [`SciDecimal`] (one with an uncertainty of 0) is easily created using [`SciDecimal::new()`]:
//!
//! ```rust
//! use scinum::SciDecimal;
//!
//! let n = SciDecimal::new(251, -3); // i.e. 251e-3
//! assert_eq!(n.to_string(), "0.251");
//! ```
//!
//! A `SciDecimal` with uncertainty can be created with [`SciDecimal::new_with_uncertainty()`]:
//!
//! ```rust
//! let m = SciDecimal::new_with_uncertainty(251, 3, -3);
//! assert_eq!(m.to_string(), "0.251(3)");
//! ```
//!
//! or from an existing `SciDecimal` with [`SciDecimal::with_uncertainty()`] (note that `SciDecimal` is immutable):
//!
//! ```rust
//! let n = SciDecimal::new(251, -3);
//! let m = n.with_uncertainty(SciDecimal::new(3, -3));
//! assert_eq!(n.to_string(), "0.251");
//! assert_eq!(m.to_string(), "0.251(3)");
//! assert_eq!(m, SciDecimal::new_with_uncertainty(251, 3, -3));
//! assert_eq!(n, m);
//! ```
//!
//! Note that two `SciDecimal` or `SciFloat` instances with the same number but different uncertainties are considered equal.
//!
//! Instantiation is even easier with the [`sci!`] macro:
//!
//! ```rust
//! let n = sci!(2.51e-3);
//! assert_eq!(n.to_string(), "0.251");
//! let m = sci!(0.251(3));
//! assert_eq!(m.to_string(), "0.251(3)");
//! ```
//!
//! [`SciFloat`] provides a similar API, but instantiation just uses an `f64` or two:
//!
//! ```rust
//! use scinum::SciFloat;
//!
//! let f = SciFloat::new(0.251);
//! assert_eq!(f.to_string(), "0.251");
//! let g = SciFloat::new_with_uncertainty(0.251, 0.003);
//! assert_eq!(f.to_string(), "0.251(3)");
//! ```
//!
//! ## Decimal implementation
//!
//! Just like `SciFloat` is intended to behave just like `f64`, `SciDecimal` is intended to behave just as if it were an implementation of [`decimal64` from the IEEE 754 standard](https://en.wikipedia.org/wiki/Decimal64_floating-point_format).
//!
//! Tracking the associated uncertainty is only possible by increasing the size, and so `SciFloat` and `SciDecimal` are both 128 bits wide.
//! `SciFloat` is simply a pair of `binary64` values.
//! `SciDecimal` takes a different approach, however, and is not simply a wrapper for two equally-sized `decimal64` values.
//!
//! An implementation of IEEE 754's `decimal64` format covers the full range of values representable by `binary64` (and a bit more), a feat which is achievable by clever use of a multi-purpose "combination field".
//! The complexity of the design makes implementation a challenge and reduces the efficiency of a software implementation.
//!
//! `SciDecimal` has a binary integer decimal (BID) encoding, but in a way that differs from the BID encoding specified by IEEE 754, taking advantage of the extra bits available to adopt a simpler design.
//! A `SciDecimal` has no combination field, and yet covers the full range of values representable by the `decimal64` interchange format.
//! This is done by expanding the representation of the number beyond 64 bits, by using a simple `u64` for the significand, an `i16` exponent, and a sign bit, as well as flags for special values.
//!
//! This much simpler bit pattern allows for more efficient calculations.
//! It also means that a full 16 decimal digits of precision can be assured across the entire representable range, with no need for `decimal64`'s "subnormal" numbers.
//! It also has the convenient side effect of massively increasing the exponent range, and thus the range of representable values – `decimal64` has a range (for _normal_ numbers) from `1e−383` to `1.0e+385`, while `SciDecimal` spans from `1e−32768` to `1e+32783`.
//! Though such numbers are ridiculously small and large, they may see occasional need in scientific fields, and the extra range is useful particularly for representation of intermediate results and for avoiding saturating behaviour.
//!
//! The exponent is (for now at least) unbiased, as the different bit layout diminishes the advantages of using a bias, and so using a signed integer reduces complexity.
//!
//! The significand of the uncertainty is encoded by a `u32` significand.
//! 9 significant figures are more than enough for an uncertainty, of which typically only 1 or 2 are quoted (but of course the storage of more is necessary to avoid rounding errors).
//! Then, taking advantage of decimal floating point's built-in tracking of precision, the exponent of the number is re-used as the exponent of the uncertainty; the possibility that the precisions of the number and uncertainty differ is accounted for by an extra `uncertainty_scale: i8`.
//!
//! ## Comparison to other decimal implementations
//!
//! The decision to implement a custom IEEE 754 decimal float was motivated by the lack of one in the Rust ecosystem.
//!
//! Rust has several crates that offer fixed-point decimal types, among them [`bigdecimal`](https://crates.io/crates/bigdecimal), [`fastnum`](https://crates.io/crates/fastnum), and [`arrow`](https://crates.io/crates/arrow).
//! Fixed-point decimal is ideal for use in finance applications, but not for scientific calculations.
//! [`rust_decimal`](https://crates.io/crates/rust_decimal) is the popular choice for floating point arithmetic in Rust.
//! However, the range is severely limited, to approximately ±1e±28, making it inappropriate for scientific contexts, which often deal with far larger and smaller exponents.
//!
//! [`decimal-rs`]((https://crates.io/crates/decimal-rs)) is no longer maintained, and its `Decimal` type is 160 bits wide, so a tuple of `(number, uncertainty)` would be 2.5 times bigger than `SciDecimal`.
//!
//! [`decimal`]((https://crates.io/crates/decimal)) does indeed provide IEEE 754 decimal floating point arithmetic by using the `decNumber` C library, but it only offers `decimal128`, not `decimal64`, so `struct SciDecimal { number: decimal, uncertainty: decimal }` would be twice the size of `SciDecimal`.
//!
//! ## Implementation status
//!
//! The long-term goal is to fully implement the IEEE 754 standard for both `SciDecimal` and `SciFloat`.
//!
//! However, the crate is an ongoing work in progress and lots remains to be implemented.
//! Many methods are `todo!()`, many operations are implemented but only return exact quantities (i.e. the uncertainty is dropped), and many others that are implemented have non-ideal behaviour for edge cases (e.g. truncating instead of rounding).
//!
//! Documentation and test coverage is not currently complete.
//!
//! The operations of `SciDecimal` have also not yet been optimized for performance.

mod error;
mod integer;
mod rounding;
mod scicast;
mod scidecimal;
mod scifloat;
mod scinum;
mod uncertainties;

pub use error::SciNumError;
pub use rounding::RoundingMode;
pub use scicast::{CheckedSciCast, CheckedSciCastFrom, SciCast, SciCastFrom};
pub use scidecimal::scidecimal::SciDecimal;
pub use scifloat::scifloat::SciFloat;
pub use scinum::SciNum;
