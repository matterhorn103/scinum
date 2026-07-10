# scinum

An implementation of decimal floating point arithmetic, and float types that carry and propagate uncertainty.

The `SciDecimal` type is a Rust implementation of decimal floating point similar to IEEE 754's `decimal64`, but with a 128-bit width that accommodates an uncertainty and simplifies the layout.
It supports `NaN` and infinities, can express up to 16 decimal digits of precision, covers a range from ±10<sup>−32768</sup> to ±10<sup>32783</sup>, and implements `num_traits::Num`, `num_traits::Float` and standard arithmetic.

Uncertainties are optional and `SciDecimal` is entirely viable for use simply as a decimal alternative to `f64` or as an alternative to other Rust decimal crates.

`SciFloat` simply wraps a pair of `f64` values, representing a number and its uncertainty, again for an overall width of 128 bits.
It is intended as a drop-in replacement for `f64` that trades a doubled size for the convenience of uncertainty arithmetic, without requiring the inevitable performance compromise of `SciDecimal`.

## Features

Both `SciDecimal` and `SciFloat`:

1. Can have associated uncertainty that is propagated correctly across arithmetic operations
2. Offer a complete set of rounding methods (to _n_ significant figures, decimal places etc.) with all rounding modes including "traditional" rounding
3. Behave according to IEEE 754 and have signed zeros, infinities, and `NaN`
4. Overload operators for the common arithmetic operations
5. Implement `num_traits::Num` and `num_traits::Float` to provide further mathematical functions
6. Implement the `SciNum` trait, allowing code to be generic over either type
7. Have appropriate string representations, including any uncertainty, with the choice of positional or scientific notation

## Usage

An exact `SciDecimal` (one with an uncertainty of 0) is easily created using `new()`:

```rust
use scinum::SciDecimal;

let n = SciDecimal::new(251, -3); // i.e. 251e-3
assert_eq!(n.to_string(), "0.251");
```

A `SciDecimal` with uncertainty can be created with the appropriate method:

```rust
let m = SciDecimal::new_with_uncertainty(251, 3, -3);
assert_eq!(m.to_string(), "0.251(3)");
```

or from an existing `SciDecimal` (note that `SciDecimal` is immutable):

```rust
let n = SciDecimal::new(251, -3);
let m = n.with_uncertainty(SciDecimal::new(3, -3));
assert_eq!(n.to_string(), "0.251");
assert_eq!(m.to_string(), "0.251(3)");
assert_eq!(m, SciDecimal::new_with_uncertainty(251, 3, -3));
assert_eq!(n, m);
```

Note that two `SciDecimal`s or `SciFloat`s with the same number but different uncertainties are considered equal.

A `sci!()` macro is provided for convenient, literal-like creation of a `SciDecimal`:

```rust
let n = sci!(2.51e-3);
assert_eq!(n.to_string(), "0.251");
let m = sci!(0.251(3));
assert_eq!(m.to_string(), "0.251(3)");
```

`SciFloat` provides a similar API, but instantiation just uses an `f64` or two:

```rust
use scinum::SciFloat;

let f = SciFloat::new(0.251);
assert_eq!(f.to_string(), "0.251");
let g = SciFloat::new_with_uncertainty(0.251, 0.003);
assert_eq!(f.to_string(), "0.251(3)");
```

## Decimal implementation

Just like `SciFloat` is intended to behave just like `f64`, `SciDecimal` is intended to behave just as if it were an implementation of [`decimal64` from the IEEE 754 standard](https://en.wikipedia.org/wiki/Decimal64_floating-point_format), and all possible `decimal64` values can be represented.

Tracking the associated uncertainty is only possible by increasing the size, and so `SciFloat` and `SciDecimal` are both 128 bits wide.
`SciFloat` is simply a pair of `binary64` values.
`SciDecimal` takes a different approach, however, and is not simply a wrapper for two equally-sized `decimal64` values.

An implementation of IEEE 754's `decimal64` format covers roughly the same range of values representable by `binary64`, a feat which is achievable by clever use of a multi-purpose "combination field".
The complexity of the design makes implementation a challenge and reduces the efficiency of a software implementation.

`SciDecimal` has a binary integer decimal (BID) encoding, but in a way that differs from the BID encoding specified by IEEE 754, taking advantage of the extra bits available to adopt a simpler design.
A `SciDecimal` has no combination field, and yet covers the full range of values representable by the `decimal64` interchange format.
This is done by expanding the representation of the number beyond 64 bits, by using a simple `u64` for the significand, an `i16` exponent, and a sign bit, as well as flags for special values.

This much simpler bit pattern allows for more efficient calculations.
It also means that a full 16 decimal digits of precision can be assured across the entire representable range, with no need for `decimal64`'s "subnormal" numbers.
It also has the convenient side effect of massively increasing the exponent range, and thus the range of representable values – `decimal64` has a range (for _normal_ numbers) from `1e−383` to `1.0e+385`, while `SciDecimal` spans from `1e-32768` to `1e+32783`.
Though such numbers are ridiculously small and large, they may see occasional need in scientific fields, and the extra range is useful particularly for representation of intermediate results and for avoiding saturating behaviour.

The exponent is (for now at least) unbiased, as the different bit layout diminishes the advantages of using a bias, and so using a signed integer reduces complexity.

The significand of the uncertainty is encoded by a `u32` significand.
9 significant figures are more than enough for an uncertainty, of which typically only 1 or 2 are quoted (but of course the storage of more is necessary to avoid rounding errors).
Then, taking advantage of decimal floating point's built-in tracking of precision, the exponent of the number is re-used as the exponent of the uncertainty; the possibility that the precisions of the number and uncertainty differ is accounted for by an extra `uncertainty_scale: i8`.

## Comparison to other decimal implementations

The decision to implement a custom IEEE 754 decimal float was motivated by the lack of one in the Rust ecosystem.

Rust has several crates that offer fixed-point decimal types, among them [`bigdecimal`](https://crates.io/crates/bigdecimal), [`fastnum`](https://crates.io/crates/fastnum), and [`arrow`](https://crates.io/crates/arrow).
Fixed-point decimal is ideal for use in finance applications, but not for scientific calculations.

[`rust_decimal`](https://crates.io/crates/rust_decimal) is the popular choice for floating point arithmetic in Rust.
However, the range is severely limited, to approximately ±1e±28, making it inappropriate for scientific contexts, which often deal with far larger and smaller exponents.
It is also missing various features of IEEE 754 floats, in particular support for infinity or `NaN`.

[`decimal`]((https://crates.io/crates/decimal)) does indeed provide IEEE 754 decimal floating point arithmetic by using the `decNumber` C library, but is unmaintained.

## Implementation status

The long-term goal is to fully implement the IEEE 754 standard for both `SciDecimal` and `SciFloat`.

However, the crate is an ongoing work in progress and lots remains to be implemented.
Many methods are `todo!()`, many operations are implemented but only return exact quantities (i.e. the uncertainty is dropped), and many others that are implemented have non-ideal behaviour for edge cases (e.g. truncating instead of rounding).

Documentation and test coverage is not currently complete.

The operations of `SciDecimal` have also not yet been optimized for performance.

## Future work

Beyond simply completing implementation of the current scope, goals include:
- conversion to and from further foreign decimal types including Arrow
- Serde de-/serialization
- Python bindings and a PyPI package release
- complex numbers with associated uncertainty, using either binary or decimal floating point arithmetic, or both

A related longer term wish would be for a performant, safe, fully spec-compliant, Rust-native implementation of the IEEE 754 decimal floating point interchange formats, but that would be a huge undertaking and much beyond the scope of this project.

## Contributing

Contributions to improve operation coverage, robustness, and performance are very welcome, and should be made by opening a PR on GitHub.

The codebase follows normal Rust standards and conventions for e.g. formatting.

## Downstream usage

`scinum` was originally written to provide numeric types for [`quanstants`](https://github.com/matterhorn103/quanstants), a Python units and quantities library written in Rust for performance.

## License

Licensed under either of:

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
