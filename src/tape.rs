// src/tape.rs — Trace-Based Reverse-Mode Automatic Differentiation
//
// Problem with the static SSA autodiff (src/autodiff.rs):
//   It only processes the entry BasicBlock.  Functions containing loops have
//   phi nodes and back-edges that the static pass cannot follow.
//
// Solution — execution trace + reverse replay:
//   1. Run the function with a recording interpreter that stores every
//      instruction execution as a TapeEntry (the "Wengert list").
//      Each loop iteration creates separate entries → loops are unrolled
//      implicitly by the forward execution.
//
//   2. Seed: adj[return_register] = 1.0
//
//   3. Walk the TapeEntry list in REVERSE.  For each entry, use the
//      concrete register values captured at record time to apply the VJP
//      rule.  Accumulate adjoints for operand registers.
//
//   4. Collect adj[param_load_register] for each function parameter.
//
// The output is (primal_f64, Vec<f64>) — the function value and gradient
// at the specific input point.  Unlike the static autodiff, the result is
// NOT a new IrFunction; it is a concrete numeric gradient.
//
// Comparison:
//   static autodiff  →  general IrFunction, only straight-line code
//   tape AD          →  concrete Vec<f64>,  handles arbitrary control flow

use std::collections::HashMap;
use crate::errors::DslError;
use crate::ir::*;
use crate::interp::Interpreter;

// ─────────────────────────────────────────────────────────────────────────────
// Tape entry
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TapeEntry {
    /// The instruction that was executed
    pub instr: Instruction,
    /// Snapshot of all register values AT THE TIME this instruction ran.
    /// Used during the reverse sweep to recover operand values.
    pub regs: HashMap<usize, f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Recording interpreter
// ─────────────────────────────────────────────────────────────────────────────

/// Runs `func` with `inputs`, recording every instruction execution.
/// Returns (tape, return_value, param_name → load_register).
pub fn record(
    func:   &IrFunction,
    inputs: &[f64],
) -> Result<(Vec<TapeEntry>, f64, HashMap<String, usize>), DslError> {
    // Delegate actual execution to the standard interpreter; we intercept
    // by wrapping it in a custom execution loop that records each step.
    let mut regs: HashMap<usize, f64>   = HashMap::new();
    let mut mem:  HashMap<String, f64>  = HashMap::new();
    let mut tape: Vec<TapeEntry>        = Vec::new();
    let mut param_loads: HashMap<String, usize> = HashMap::new();
    let mut prev_block: Option<usize>   = None;

    // Bind parameters
    for ((name, _), val) in func.params.iter().zip(inputs.iter()) {
        mem.insert(name.clone(), *val);
    }

    let max_iters = 10_000_000usize;
    let mut iters = 0usize;
    let mut current = 0usize;

    loop {
        let block = func.blocks.get(current)
            .ok_or_else(|| DslError::IrError(format!("Block {} OOB", current)))?;

        for instr in &block.instructions {
            // Snapshot registers BEFORE executing
            let snap = regs.clone();

            // Execute instruction (simplified — full version in interp.rs)
            exec_instr(instr, &mut regs, &mut mem, prev_block, &mut param_loads)?;

            // Record
            tape.push(TapeEntry { instr: instr.clone(), regs: snap });
        }

        iters += 1;
        if iters > max_iters {
            return Err(DslError::IrError("tape: exceeded max iterations".into()));
        }

        let next = match &block.terminator {
            Terminator::Return(v) => {
                let rv = eval(&regs, v);
                return Ok((tape, rv, param_loads));
            }
            Terminator::Jump(b) => *b,
            Terminator::CondJump { cond, true_block, false_block } => {
                if eval(&regs, cond) != 0.0 { *true_block } else { *false_block }
            }
            Terminator::Unreachable =>
                return Err(DslError::IrError("unreachable".into())),
        };
        prev_block = Some(current);
        current = next;
    }
}

fn exec_instr(
    instr:       &Instruction,
    regs:        &mut HashMap<usize, f64>,
    mem:         &mut HashMap<String, f64>,
    prev_block:  Option<usize>,
    param_loads: &mut HashMap<String, usize>,
) -> Result<(), DslError> {
    match instr {
        Instruction::Add { dest, lhs, rhs } => set(regs, dest, eval(regs, lhs) + eval(regs, rhs)),
        Instruction::Sub { dest, lhs, rhs } => set(regs, dest, eval(regs, lhs) - eval(regs, rhs)),
        Instruction::Mul { dest, lhs, rhs } => set(regs, dest, eval(regs, lhs) * eval(regs, rhs)),
        Instruction::Div { dest, lhs, rhs } => {
            let r = eval(regs, rhs);
            if r == 0.0 { return Err(DslError::IrError("division by zero".into())); }
            set(regs, dest, eval(regs, lhs) / r);
        }
        Instruction::Mod { dest, lhs, rhs } => set(regs, dest, eval(regs, lhs) % eval(regs, rhs)),
        Instruction::Pow { dest, base, exp } => set(regs, dest, eval(regs, base).powf(eval(regs, exp))),
        Instruction::Neg { dest, src } => set(regs, dest, -eval(regs, src)),
        Instruction::Lt  { dest, lhs, rhs } => set(regs, dest, if eval(regs, lhs) <  eval(regs, rhs) { 1.0 } else { 0.0 }),
        Instruction::Gt  { dest, lhs, rhs } => set(regs, dest, if eval(regs, lhs) >  eval(regs, rhs) { 1.0 } else { 0.0 }),
        Instruction::Le  { dest, lhs, rhs } => set(regs, dest, if eval(regs, lhs) <= eval(regs, rhs) { 1.0 } else { 0.0 }),
        Instruction::Ge  { dest, lhs, rhs } => set(regs, dest, if eval(regs, lhs) >= eval(regs, rhs) { 1.0 } else { 0.0 }),
        Instruction::Eq  { dest, lhs, rhs } => set(regs, dest, if (eval(regs, lhs) - eval(regs, rhs)).abs() < 1e-12 { 1.0 } else { 0.0 }),
        Instruction::Ne  { dest, lhs, rhs } => set(regs, dest, if (eval(regs, lhs) - eval(regs, rhs)).abs() >= 1e-12 { 1.0 } else { 0.0 }),
        Instruction::Load { dest, name } => {
            let v = mem.get(name).copied().unwrap_or(0.0);
            set(regs, dest, v);
            // first load of a param? register it
            if !param_loads.contains_key(name.as_str()) {
                param_loads.insert(name.clone(), dest.0);
            }
        }
        Instruction::Store { src, name } => { mem.insert(name.clone(), eval(regs, src)); }
        Instruction::Call { dest, func, args } => {
            let av: Vec<f64> = args.iter().map(|a| eval(regs, a)).collect();
            set(regs, dest, call_builtin(func, &av)?);
        }
        Instruction::BuildTensor { dest, elems } => {
            // Store first element as scalar for gradient purposes
            let v = elems.first().map(|e| eval(regs, e)).unwrap_or(0.0);
            set(regs, dest, v);
        }
        Instruction::Index { dest, base, idx } => {
            set(regs, dest, eval(regs, base)); // simplified: no true tensor
        }
        Instruction::Phi { dest, incoming } => {
            let v = if let Some(prev) = prev_block {
                incoming.iter()
                    .find(|(_, b)| *b == prev)
                    .map(|(v, _)| eval(regs, v))
                    .unwrap_or(0.0)
            } else {
                incoming.first().map(|(v, _)| eval(regs, v)).unwrap_or(0.0)
            };
            set(regs, dest, v);
        }
        Instruction::Comment(_) => {}
        _ => {}
    }
    Ok(())
}

fn set(regs: &mut HashMap<usize, f64>, dest: &Register, v: f64) {
    regs.insert(dest.0, v);
}
fn eval(regs: &HashMap<usize, f64>, v: &Value) -> f64 {
    match v {
        Value::Reg(Register(id)) => regs.get(id).copied().unwrap_or(0.0),
        Value::ConstF64(f)       => *f,
        Value::ConstI64(i)       => *i as f64,
    }
}
fn call_builtin(name: &str, args: &[f64]) -> Result<f64, DslError> {
    let a = args.get(0).copied().unwrap_or(0.0);
    let b = args.get(1).copied().unwrap_or(0.0);
    Ok(match name {
        "sin" => a.sin(), "cos" => a.cos(), "tan" => a.tan(),
        "exp" => a.exp(), "log" => a.ln(),  "sqrt" => a.sqrt(),
        "abs" => a.abs(), "sign" | "signum" => a.signum(),
        "min" => a.min(b), "max" => a.max(b),
        "asin" => a.asin(), "acos" => a.acos(), "atan" => a.atan(),
        "floor" => a.floor(), "ceil" => a.ceil(), "round" => a.round(),
        other => return Err(DslError::IrError(format!("unknown builtin: {}", other))),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Reverse sweep — VJP over tape
// ─────────────────────────────────────────────────────────────────────────────

/// Compute gradient of `func` at `inputs` using trace-based reverse-mode AD.
/// Returns `(f(inputs), [∂f/∂x₀, ∂f/∂x₁, …])`.
///
/// Adjoints are keyed by **tape-entry index** (not SSA register ID).
/// This handles loops correctly: each iteration of a loop body creates
/// separate tape entries, each with its own adjoint, even though they
/// share the same SSA register IDs in the compiled IR.
pub fn tape_grad(
    func:   &IrFunction,
    inputs: &[f64],
) -> Result<(f64, Vec<f64>), DslError> {
    let (tape, primal, param_loads) = record(func, inputs)?;
    let n = tape.len();

    // ── adj[i] = adjoint of the value produced by tape entry i ───────────────
    let mut adj: Vec<f64> = vec![0.0; n];

    // Find the most recent tape entry before `before` that defined `reg_id`
    let find_prod = |before: usize, reg_id: usize| -> Option<usize> {
        tape[..before].iter().enumerate().rev()
            .find(|(_, e)| matches!(e.instr.dest(), Some(r) if r.0 == reg_id))
            .map(|(i, _)| i)
    };

    // ── Seed ─────────────────────────────────────────────────────────────────
    // Find the return register from the terminator, then find the last tape
    // entry that produced it, and set its adjoint to 1.0.
    let ret_reg = func.blocks.iter().find_map(|b| {
        if let Terminator::Return(Value::Reg(Register(id))) = &b.terminator {
            Some(*id)
        } else { None }
    });

    match ret_reg {
        Some(id) => {
            if let Some(seed_idx) = find_prod(n, id) {
                adj[seed_idx] = 1.0;
            } else {
                // Return value is computed inline (e.g. last Store result); 
                // seed the last entry that stored to memory and whose Load
                // feeds the return.  Fallback: seed the last defining entry.
                if let Some(last) = tape.iter().enumerate().rev()
                    .find(|(_, e)| e.instr.dest().is_some())
                    .map(|(i, _)| i)
                {
                    adj[last] = 1.0;
                }
            }
        }
        None => return Ok((primal, vec![0.0; func.params.len()])),
    }

    // ── adj_mem: adjoint for named memory locations ───────────────────────────
    let mut adj_mem: HashMap<String, f64> = HashMap::new();

    // ── Helper: accumulate adjoint into the producer of a Value ──────────────
    // Defined as inner closure to capture `adj` mutably.
    // We use a helper fn below instead (Rust lifetime rules).

    // ── Reverse sweep ─────────────────────────────────────────────────────────
    for i in (0..n).rev() {
        let entry  = &tape[i];
        let regs   = &entry.regs;

        let snap = |v: &Value| -> f64 {
            match v {
                Value::Reg(Register(id)) => regs.get(id).copied().unwrap_or(0.0),
                Value::ConstF64(f)       => *f,
                Value::ConstI64(i)       => *i as f64,
            }
        };

        // ── Store / Load: propagate through memory ───────────────────────────
        match &entry.instr {
            Instruction::Store { src, name } => {
                let ma = adj_mem.remove(name).unwrap_or(0.0);
                if ma.abs() > 0.0 {
                    if let Value::Reg(Register(id)) = src {
                        if let Some(pi) = find_prod(i, *id) {
                            adj[pi] += ma;
                        }
                    }
                }
                continue;
            }
            Instruction::Load { name, .. } => {
                let ra = adj[i];
                if ra.abs() > 0.0 {
                    *adj_mem.entry(name.clone()).or_insert(0.0) += ra;
                }
                continue;
            }
            _ => {}
        }

        let s = adj[i];
        if s == 0.0 { continue; }

        // Helper: add `delta` to the adjoint of the entry that produced `val`
        // before position i.
        macro_rules! gacc {
            ($val:expr, $delta:expr) => {
                if let Value::Reg(Register(id)) = $val {
                    if let Some(pi) = find_prod(i, *id) {
                        adj[pi] += $delta;
                    }
                }
            };
        }

        match &entry.instr {
            Instruction::Add { lhs, rhs, .. } => {
                gacc!(lhs, s); gacc!(rhs, s);
            }
            Instruction::Sub { lhs, rhs, .. } => {
                gacc!(lhs, s); gacc!(rhs, -s);
            }
            Instruction::Mul { lhs, rhs, .. } => {
                gacc!(lhs, s * snap(rhs));
                gacc!(rhs, s * snap(lhs));
            }
            Instruction::Div { lhs, rhs, .. } => {
                let b = snap(rhs);
                gacc!(lhs, s / b);
                gacc!(rhs, -s * snap(lhs) / (b * b));
            }
            Instruction::Mod { lhs, .. } => { gacc!(lhs, s); }
            Instruction::Pow { base, exp, .. } => {
                let b = snap(base); let e = snap(exp);
                if b != 0.0 { gacc!(base, s * e * b.powf(e - 1.0)); }
                if b > 0.0  { gacc!(exp,  s * b.powf(e) * b.ln()); }
            }
            Instruction::Neg { src, .. } => { gacc!(src, -s); }
            Instruction::Call { func, args, .. } => {
                let a = args.get(0).map(|v| snap(v)).unwrap_or(0.0);
                let g = match func.as_str() {
                    "sin"  => a.cos(),
                    "cos"  => -a.sin(),
                    "tan"  => 1.0 / a.cos().powi(2),
                    "exp"  => a.exp(),
                    "log"  => 1.0 / a,
                    "sqrt" => 1.0 / (2.0 * a.sqrt()),
                    "abs"  => a.signum(),
                    _      => 0.0,
                };
                if let Some(x) = args.get(0) { gacc!(x, s * g); }
            }
            Instruction::Phi { .. } | Instruction::Comment(_) => {}
            _ => {}
        }
    }

    // ── Collect parameter gradients ───────────────────────────────────────────
    // Sum adjoints of ALL tape entries that loaded a given parameter name.
    let grads: Vec<f64> = func.params.iter().map(|(name, _)| {
        tape.iter().enumerate()
            .filter(|(_, e)| matches!(&e.instr, Instruction::Load { name: n, .. } if n == name))
            .map(|(i, _)| adj[i])
            .sum()
    }).collect();

    Ok((primal, grads))
}

fn acc(adj: &mut HashMap<usize, f64>, val: &Value, delta: f64) {
    if let Value::Reg(Register(id)) = val {
        *adj.entry(*id).or_insert(0.0) += delta;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gradient check using tape AD
// ─────────────────────────────────────────────────────────────────────────────

/// Verify tape-based gradients against central finite differences.
pub fn tape_gradcheck(
    func:    &IrFunction,
    inputs:  &[f64],
    eps:     f64,
    atol:    f64,
) -> Result<TapeGradCheckResult, DslError> {
    let (primal, analytic) = tape_grad(func, inputs)?;

    let mut numeric = Vec::with_capacity(inputs.len());
    for i in 0..inputs.len() {
        let h = eps * inputs[i].abs().max(1.0);
        let mut xp = inputs.to_vec(); xp[i] += h;
        let mut xm = inputs.to_vec(); xm[i] -= h;
        let (fp, _) = tape_grad(func, &xp)?;
        let (fm, _) = tape_grad(func, &xm)?;
        numeric.push((fp - fm) / (2.0 * h));
    }

    let abs_errors: Vec<f64> = analytic.iter().zip(numeric.iter())
        .map(|(a, n)| (a - n).abs())
        .collect();
    let max_err = abs_errors.iter().cloned().fold(0.0_f64, f64::max);

    let passed = analytic.iter().zip(numeric.iter()).all(|(a, n)| {
        let e = (a - n).abs();
        e <= atol || e / n.abs().max(1e-8) <= atol * 10.0
    });

    Ok(TapeGradCheckResult {
        primal,
        param_names: func.params.iter().map(|(n,_)| n.clone()).collect(),
        analytic,
        numeric,
        abs_errors,
        max_err,
        passed,
    })
}

#[derive(Debug)]
pub struct TapeGradCheckResult {
    pub primal:      f64,
    pub param_names: Vec<String>,
    pub analytic:    Vec<f64>,
    pub numeric:     Vec<f64>,
    pub abs_errors:  Vec<f64>,
    pub max_err:     f64,
    pub passed:      bool,
}

impl TapeGradCheckResult {
    pub fn report(&self) -> String {
        let mut s = format!(
            "TapeGrad: {}  f={:.6}\n  {:>12}  {:>14}  {:>14}  {:>12}\n",
            if self.passed { "PASSED ✓" } else { "FAILED ✗" },
            self.primal, "param", "analytic", "numeric", "abs_error"
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
        s.push_str(&format!("  max_err = {:.2e}", self.max_err));
        s
    }
}
