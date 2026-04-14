// src/gradcheck.rs — Numerical Gradient Verification
//
// Compares the analytical gradients produced by autodiff with numerical
// finite-difference approximations.  This is the same technique used by
// PyTorch's `torch.autograd.gradcheck` and JAX's `jax.test_util.check_grads`.
//
// Algorithm (central differences):
//   ∂f/∂xᵢ ≈ (f(x + hᵢ) − f(x − hᵢ)) / (2h)
//
// where hᵢ = eps * max(|xᵢ|, 1).  Central differences give O(h²) accuracy
// vs O(h) for forward differences.
//
// Usage:
//   let result = gradcheck(&func, &inputs, GradCheckOpts::default())?;
//   assert!(result.passed, "{}", result.report());

use crate::autodiff;
use crate::errors::DslError;
use crate::interp::{Interpreter, Val};
use crate::ir::{IrFunction, Terminator, Value};

// ─────────────────────────────────────────────────────────────────────────────
// Options
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GradCheckOpts {
    /// Finite difference step size
    pub eps:     f64,
    /// Absolute tolerance for gradient comparison
    pub atol:    f64,
    /// Relative tolerance for gradient comparison
    pub rtol:    f64,
}

impl Default for GradCheckOpts {
    fn default() -> Self {
        GradCheckOpts { eps: 1e-5, atol: 1e-4, rtol: 1e-3 }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Result
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct GradCheckResult {
    pub passed:      bool,
    pub param_names: Vec<String>,
    /// analytical gradient from autodiff
    pub analytic:    Vec<f64>,
    /// numerical gradient from finite differences
    pub numeric:     Vec<f64>,
    /// per-component absolute error
    pub abs_errors:  Vec<f64>,
    pub max_abs_err: f64,
}

impl GradCheckResult {
    pub fn report(&self) -> String {
        let mut s = format!(
            "GradCheck: {}\n  {:>12}  {:>14}  {:>14}  {:>12}\n",
            if self.passed { "PASSED ✓" } else { "FAILED ✗" },
            "param", "analytic", "numeric", "abs_error"
        );
        for i in 0..self.param_names.len() {
            s.push_str(&format!(
                "  {:>12}  {:>14.8}  {:>14.8}  {:>12.2e}\n",
                self.param_names[i],
                self.analytic.get(i).copied().unwrap_or(f64::NAN),
                self.numeric.get(i).copied().unwrap_or(f64::NAN),
                self.abs_errors.get(i).copied().unwrap_or(f64::NAN),
            ));
        }
        s.push_str(&format!("  max_abs_err = {:.2e}", self.max_abs_err));
        s
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main check function
// ─────────────────────────────────────────────────────────────────────────────

/// Verify that autodiff of `func` matches finite-difference gradients at `inputs`.
pub fn gradcheck(
    func:   &IrFunction,
    inputs: &[f64],
    opts:   &GradCheckOpts,
) -> Result<GradCheckResult, DslError> {
    let n = func.params.len();
    if inputs.len() != n {
        return Err(DslError::IrError(format!(
            "gradcheck: {} inputs given but function has {} params", inputs.len(), n)));
    }

    // ── Analytical gradients via autodiff ────────────────────────────────────
    let grad_fn = autodiff::differentiate(func)?;
    let mut interp = Interpreter::new();
    let grad_out = interp.call(&grad_fn, inputs)?;

    // grad_fn returns Tensor[primal, ∂f/∂x0, ∂f/∂x1, ...]
    let grad_tensor = grad_out.as_tensor();
    // skip index 0 (primal value)
    let analytic: Vec<f64> = (1..=n)
        .map(|i| grad_tensor.get(i).copied().unwrap_or(0.0))
        .collect();

    // ── Numerical gradients via central differences ───────────────────────────
    let mut numeric = Vec::with_capacity(n);
    for i in 0..n {
        let h = opts.eps * inputs[i].abs().max(1.0);

        let mut x_plus  = inputs.to_vec();
        let mut x_minus = inputs.to_vec();
        x_plus[i]  += h;
        x_minus[i] -= h;

        let f_plus  = interp.call(func, &x_plus)?.as_f64();
        let f_minus = interp.call(func, &x_minus)?.as_f64();

        numeric.push((f_plus - f_minus) / (2.0 * h));
    }

    // ── Compare ───────────────────────────────────────────────────────────────
    let abs_errors: Vec<f64> = analytic.iter().zip(numeric.iter())
        .map(|(a, n)| (a - n).abs())
        .collect();

    let max_abs_err = abs_errors.iter().cloned().fold(0.0_f64, f64::max);

    let passed = analytic.iter().zip(numeric.iter()).all(|(a, n)| {
        let abs_err = (a - n).abs();
        let rel_err = abs_err / (n.abs().max(1e-8));
        abs_err <= opts.atol || rel_err <= opts.rtol
    });

    Ok(GradCheckResult {
        passed,
        param_names: func.params.iter().map(|(n, _)| n.clone()).collect(),
        analytic,
        numeric,
        abs_errors,
        max_abs_err,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience: check all scalar-returning functions in a module
// ─────────────────────────────────────────────────────────────────────────────

pub struct ModuleGradCheckResult {
    pub func_name: String,
    pub result:    Result<GradCheckResult, DslError>,
}

pub fn gradcheck_module(
    module:      &crate::ir::IrModule,
    input_fn:    impl Fn(&IrFunction) -> Vec<f64>,
    opts:        &GradCheckOpts,
) -> Vec<ModuleGradCheckResult> {
    use crate::ast::TypeAnnotation;
    module.functions.iter()
        .filter(|f| matches!(f.ret_ty, TypeAnnotation::F64 | TypeAnnotation::F32))
        .filter(|f| !f.name.ends_with("__grad"))   // skip already-generated grad fns
        .map(|f| {
            let inputs = input_fn(f);
            ModuleGradCheckResult {
                func_name: f.name.clone(),
                result:    gradcheck(f, &inputs, opts),
            }
        })
        .collect()
}
