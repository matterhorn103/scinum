/// A trait for the implementation of infallible but potentially lossy conversions.
///
/// These casts follow the semantics of [numeric casting with the `as` keyword](https://doc.rust-lang.org/reference/expressions/operator-expr.html#numeric-cast):
///
/// - Casting never fails or panics.
///
/// - The output value is the closest possible to the input value representable
///   by the target type.
///
/// - If both types support uncertainties, any uncertainty is preserved; otherwise
///   any uncertainty is dropped and the output is always exact.
///
/// - Loss of precision is acceptable, with rounding as follows:
///
///   - Casting from a fractional type (including floating and fixed point types)
///   to an integer type rounds towards zero i.e. [`scinum::RoundingMode::Down`].
///
///   - Casting to a fractional type, whether from another fractional type or from
///   an integer type, rounds to nearest with ties to even i.e. [`scinum::RoundingMode::HalfEven`].
///
/// - Saturation is acceptable, as follows:
///
///   - A value too small (i.e. too close to zero) to be represented by the
///     target type returns `0` with the appropriate sign.
///
///   - A value too large to be represented by the target type returns the saturated
///     maximum value of the type, or infinity if the type has an infinity.
///
///   - A value too low (i.e. too negative) to be represented by the target type
///     returns the saturated minimum value of the type, or negative infinity if
///     the type has it.
///
/// - Special float values are preserved if the target type also supports them;
///   if not, alternatives are used as follows:
///
///   - `-0` becomes `0` if zero is unsigned in the target type
///
///   - `inf` becomes the saturated maximum value of the target type
///
///   - `-inf` becomes the saturated minimum value of the target type, which will
///     be `0` for types with no negative numbers
///
///   - `NaN` becomes `0`
///
/// WARNING: If the target type cannot represent non-finite numbers (`inf` and/or
/// `NaN`) then `cast()` converts them to finite numbers, with the potentially
/// misleading implication that an arithmetic result was finite. This could cause
/// very real problems in some situations, so care should be exercised. If such
/// conversions would be problematic, use [`CheckedSciCast::checked_cast`], which
/// returns `None` if a non-finite value would become finite.
///
/// As such, `CheckedSciCast` should also be implemented for types that cannot
/// represent `inf` and/or `NaN`.
///
/// # Comparison to similar traits
///
/// Given the rules above, `SciCast` serves a different role to `From`, `TryFrom`,
/// and `ToPrimitive`/`NumCast`, which each make a different set of promises:
///
/// - `from()` and `into()` in `std` are also infallible must be lossless.
///
/// - `try_from()` and `try_into()` in `std` are fallible, but must also be
///   lossless i.e. a lossy conversion must fail.
///
/// - The methods of `ToPrimitive` and `FromPrimitive` and thus also `NumCast::from()`
///   *may* be lossy, in the sense that loss of precision is acceptable, but they
///   are not infallible conversions.
///
/// In summary:
///
/// | Conversion                    | Return type   | Infallible?   | Rounding?     | Saturating?   | Non-finite -> finite? |
/// | ----------------------------- | ------------- | ------------- | ------------- | ------------- | --------------------- |
/// | Numeric casting with `as`     | `T`           | Yes           | Yes           | Yes           | Yes                   |
/// | `std::convert::From`          | `T`           | Yes           | No            | No            | No                    |
/// | `std::convert::TryFrom`       | `Result<T>`   | No            | No            | No            | No                    |
/// | `num::{To, From}Primitive`    | `Option<T>`   | No            | Yes           | Yes           | ?                     |
/// | `SciCast`                     | `T`           | Yes           | Yes           | Yes           | Sometimes†            |
/// | `CheckedSciCast`              | `Option<T>`   | No            | Yes           | Sometimes‡    | No                    |
///
/// † Only if the target type cannot represent the non-finite value (i.e. it
/// does not support infinities or `NaN` or both)
///
/// ‡ Only to `inf` or `0`
pub trait SciCast<T> {
    /// Converts `self` to a `T` infallibly, but potentially lossily, dropping
    /// any uncertainty, rounding and saturating as necessary.
    fn cast(self) -> T;
}

/// A trait for the implementation of infallible but potentially lossy conversions
/// in a fashion similar to [`SciCast`] but without saturating behaviour and no
/// silent conversion of infinity or `NaN` to finite values.
///
/// The conversion by [`SciCast::cast()`] of `inf` and `NaN` to finite numbers,
/// and the potentially misleading implication that an arithmetic result was
/// finite, could cause very real problems in some situations. This trait helps
/// with this by providing a `checked_cast()` method that returns `None` if a
/// non-finite value would become finite.
///
/// These casts are made according to the following rules:
///
/// - Casting never panics.
///
/// - Casting only fails if saturation would occur or a non-finite value would
///   become finite, in which case `None` is returned.
///
/// - The output value is the closest possible to the input value representable
///   by the target type.
///
/// - If both types support uncertainties, any uncertainty is preserved; otherwise
///   any uncertainty is dropped and the output is always exact.
///
/// - Loss of precision *is* acceptable, with rounding as follows:
///
///   - Casting from a fractional type (including floating and fixed point types)
///   to an integer type rounds towards zero i.e. [`scinum::RoundingMode::Down`].
///
///   - Casting to a fractional type, whether from another fractional type or from
///   an integer type, rounds to nearest with ties to even i.e. [`scinum::RoundingMode::HalfEven`].
///
/// - Saturation is *not* acceptable:
///
///   - A value too small (i.e. too close to zero) to be represented by the
///     target type returns `None`.
///
///   - A value too large to be represented by the target type returns `None`
///
///   - A value too low (i.e. too negative) to be represented by the target type
///     returns `None`.
///
/// - Special float values are preserved if the target type also supports them;
///   otherwise:
///
///   - `-0` becomes `0` if zero is unsigned in the target type
///
///   - `inf`, `-inf` and `NaN` return `None`
///
/// # Comparison to similar traits
///
/// Given the rules above, `SciCast` serves a different role to `From`, `TryFrom`,
/// and `ToPrimitive`/`NumCast`, which each make a different set of promises:
///
/// - `from()` and `into()` in `std` are also infallible must be lossless.
///
/// - `try_from()` and `try_into()` in `std` are fallible, but must also be
///   lossless i.e. a lossy conversion must fail.
///
/// - The methods of `ToPrimitive` and `FromPrimitive` and thus also `NumCast::from()`
///   *may* be lossy, in the sense that loss of precision is acceptable, but they
///   are not infallible conversions.
///
/// In summary:
///
/// | Conversion                    | Return type   | Infallible?   | Rounding?     | Saturating?   | Non-finite -> finite? |
/// | ----------------------------- | ------------- | ------------- | ------------- | ------------- | --------------------- |
/// | Numeric casting with `as`     | `T`           | Yes           | Yes           | Yes           | Yes                   |
/// | `std::convert::From`          | `T`           | Yes           | No            | No            | No                    |
/// | `std::convert::TryFrom`       | `Result<T>`   | No            | No            | No            | No                    |
/// | `num::{To, From}Primitive`    | `Option<T>`   | No            | Yes           | Yes           | ?                     |
/// | `SciCast`                     | `T`           | Yes           | Yes           | Yes           | Sometimes†            |
/// | `CheckedSciCast`              | `Option<T>`   | No            | Yes           | Sometimes‡    | No                    |
///
/// † Only if the target type cannot represent the non-finite value (i.e. it
/// does not support infinities or `NaN` or both)
///
/// ‡ Only to `inf` or `0`
pub trait CheckedSciCast<T> {
    /// Converts `self` to a `T` similarly to [`SciCast::cast`], dropping
    /// uncertainty and rounding as necessary, but with saturating behaviour
    /// only in limited cases and no silent conversion of infinity or `NaN` to
    /// finite values.
    fn checked_cast(self) -> Option<T>;
}

/// A companion trait to [`SciCast`] to form a bidirectional pair analogous to
/// `From` and `Into`.
pub trait SciCastFrom<N>: Sized
where
    N: SciCast<Self>,
{
    fn cast_from(n: N) -> Self {
        n.cast()
    }
}

/// A companion trait to [`CheckedSciCast`] to form a bidirectional pair analogous to
/// `TryFrom` and `TryInto`.
pub trait CheckedSciCastFrom<N>: Sized
where
    N: CheckedSciCast<Self>,
{
    fn checked_cast_from(n: N) -> Option<Self> {
        n.checked_cast()
    }
}

// Blanket implementations so that something that is a target for a cast method
// gets a method automatically implemented to cast it from the source type.
// Note that while this is the same idea as with from/into, it works the opposite
// way round: the source type should implement `SciCast` and the target type gets
// `SciCastFrom` for free, whereas with from/into, it's the target type that
// should implement `From` while the source type gets `Into` for free.
impl<T, N> SciCastFrom<N> for T where N: SciCast<T> {}

impl<T, N> CheckedSciCastFrom<N> for T where N: CheckedSciCast<T> {}
