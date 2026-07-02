//! Functions for calculating uncertainties.

use num_traits::Float;

// From Wikipedia https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulae:
//
// For any non-linear differentiable function of two variables f(a, b), where σ_a
// and σ_b are the standard deviations of a and b, and the result c has standard
// deviation σ_c:
// σ_c² = |δf/δa|²⋅σ_a² + |δf/δb|²⋅σ_b² + 2⋅(δf/δa)⋅(δf/δb)⋅σ_ab
// where the covariance between a and b, σ_ab, is given by:
// σ_ab = ρ_ab⋅σ_a⋅σ_b
// where ρ_ab is the correlation between a and b.

// We can't do the calculus programmatically. But we can write a function that
// takes two closures:
// 1. A closure for δf/δa
// 2. A closure for δf/δb
//
// and returns a new closure for the uncertainty that takes:
// 1. The value of a
// 2. The value of b
// 3. The uncertainty σ_a
// 4. The uncertainty σ_b
// 5. The correlation ρ_ab

/// Returns a function that calculates the uncertainty σ_c in the result C of
/// a non-linear differentiable function f(a, b).
///
/// `partderiv_a` and `partderiv_b` are the partial derivatives of f(a, b) with
/// respect to a and b.
///
/// The returned function is of the form `uncertainty(a, b, σ_a, σ_b, ρ_ab)`
/// with five arguments:
/// - `a` and `b` are the values of the two input parameters
/// - `σ_a` and `σ_b` are the uncertainties (standard deviations) of the input parameters
/// - `ρ_ab` is the correlation between a and b
pub(crate) fn uncertainty_fn_generator<N, A, B>(
    partderiv_a: A,
    partderiv_b: B,
) -> impl Fn(N, N, N, N, N) -> N
where
    N: Float,
    A: Fn(N, N) -> N,
    B: Fn(N, N) -> N,
{
    move |a, b, σ_a, σ_b, ρ_ab| {
        let dfda = partderiv_a(a, b);
        let dfdb = partderiv_b(a, b);
        let term_a = dfda.abs().powi(2) * σ_a.powi(2);
        let term_b = dfdb.abs().powi(2) * σ_b.powi(2);
        let term_ρ = if ρ_ab.is_zero() {
            N::zero()
        } else {
            let σ_ab = ρ_ab * σ_a * σ_b;
            N::from(2).expect("All floats can represent 2") * dfda * dfdb * σ_ab
        };
        let variance = term_a + term_b + term_ρ;
        let stddev = variance.sqrt();
        stddev
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add() {
        // f(a, b) = a + b
        let dfda = |a: f64, b: f64| 1_f64;
        let dfdb = |a: f64, b: f64| 1_f64;
        let uncert_fn = uncertainty_fn_generator(dfda, dfdb);
        // For our example:
        // a = 20 +/- 2
        // b = 30 +/- 5
        // ρ = 0
        // Should give:
        // c = a + b = 50 +/- 5.385164807134504
        let uncert = uncert_fn(20_f64, 30_f64, 2_f64, 5_f64, 0_f64);
        assert_eq!(uncert.to_string(), "5.385164807134504");
    }

    #[test]
    fn mul() {
        // f(a, b) = a * b
        let dfda = |a: f64, b: f64| b;
        let dfdb = |a: f64, b: f64| a;
        let uncert_fn = uncertainty_fn_generator(dfda, dfdb);
        // For our example:
        // a = 20 +/- 2
        // b = 30 +/- 5
        // ρ = 0
        // Should give:
        // c = a * b = 600 +/- 116.61903789690601
        let uncert = uncert_fn(20_f64, 30_f64, 2_f64, 5_f64, 0_f64);
        assert_eq!(uncert.to_string(), "116.61903789690601");
    }
}
