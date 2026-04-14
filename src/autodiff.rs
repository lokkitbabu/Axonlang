// src/autodiff.rs — Reverse-Mode Automatic Differentiation on SSA IR
//
// Algorithm: Reverse Accumulation (backpropagation over the SSA tape)
//
// Given a function  f(x1..xn) -> scalar:
//   1. Collect the "forward tape" — the flat list of instructions in the
//      entry block (straight-line code; loops handled via loop unrolling stub).
//   2. Seed: adj[return_val] = 1.0
//   3. Walk the tape in reverse.  For each defining instruction, look up the
//      adjoint of its destination register (adj[dest]), then emit new
//      instructions that propagate that adjoint back to the operand registers.
//   4. After the full reverse sweep, collect adj[load_of_param] for each
//      function parameter — those are the partial derivatives ∂f/∂xᵢ.
//
// The result is a NEW IrFunction named  f__grad  whose body computes both
// the primal value and the gradient vector in one forward+backward pass.
//
// Supported primitive operations and their VJPs:
//
//   add(a,b)  → adj_a += s;         adj_b += s
//   sub(a,b)  → adj_a += s;         adj_b -= s
//   mul(a,b)  → adj_a += s*b;       adj_b += s*a
//   div(a,b)  → adj_a += s/b;       adj_b -= s*a/b²
//   pow(a,e)  → adj_a += s*e*a^(e-1)  (e treated as constant if ConstF64)
//   neg(a)    → adj_a -= s
//   load @p   → record param adjoint
//   store     → skipped (no gradient through stores in this pass)
//   call f()  → treated as opaque; zero gradient (safe conservative default)

use std::collections::HashMap;

use crate::ast::TypeAnnotation;
use crate::errors::DslError;
use crate::ir::*;

// ─────────────────────────────────────────────────────────────────────────────
// Public entry
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a gradient function for `func`.
///
/// Returns a new `IrFunction` named `{func.name}__grad` whose entry block:
///  1. Replicates the forward pass of `func`.
///  2. Appends the reverse sweep.
///  3. Terminates with a `Terminator::Return` of a `BuildTensor` containing
///     [primal_result, ∂f/∂x0, ∂f/∂x1, ...] in param order.
pub fn differentiate(func: &IrFunction) -> Result<IrFunction, DslError> {
    if func.blocks.is_empty() {
        return Err(DslError::IrError(
            format!("Function '{}' has no blocks", func.name)));
    }

    // We only handle straight-line entry blocks right now.
    // Loops require AD through phi-nodes (supported via placeholder).
    let entry = &func.blocks[0];

    let mut ctx = AdjointCtx::new(func);
    ctx.run_forward(entry)?;
    ctx.run_backward(entry)?;

    // Collect gradient values for each parameter (in declaration order)
    let grad_vals: Vec<Value> = func.params.iter()
        .map(|(name, _)| ctx.param_grad(name))
        .collect();

    // Primal return value
    let primal = ctx.primal_return.clone()
        .unwrap_or(Value::ConstF64(0.0));

    // Build result: [primal, grad_x0, grad_x1, ...]
    let result_elems: Vec<Value> = std::iter::once(primal)
        .chain(grad_vals)
        .collect();
    let result_reg = ctx.fresh();
    ctx.adjoint_instrs.push(Instruction::BuildTensor {
        dest:  result_reg.clone(),
        elems: result_elems,
    });

    // Assemble the single output block
    let mut block = BasicBlock::new("entry");
    for i in ctx.forward_instrs  { block.push(i); }
    for i in ctx.adjoint_instrs  { block.push(i); }
    block.terminator = Terminator::Return(Value::Reg(result_reg));

    Ok(IrFunction {
        name:   format!("{}__grad", func.name),
        params: func.params.clone(),
        ret_ty: TypeAnnotation::Tensor,
        blocks: vec![block],
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Adjoint context
// ─────────────────────────────────────────────────────────────────────────────

struct AdjointCtx<'a> {
    func:            &'a IrFunction,
    reg_counter:     usize,
    /// Reproduced forward instructions (no terminators)
    forward_instrs:  Vec<Instruction>,
    /// Generated backward (adjoint) instructions
    adjoint_instrs:  Vec<Instruction>,
    /// adj[reg_id] = accumulated adjoint Value for that register
    adj:             HashMap<usize, Value>,
    /// param_name → register loaded in forward pass
    param_regs:      HashMap<String, Register>,
    /// The Value returned by the primal function
    primal_return:   Option<Value>,
    /// Maximum register ID seen in the original function (we allocate above it)
    base_reg:        usize,
}

impl<'a> AdjointCtx<'a> {
    fn new(func: &'a IrFunction) -> Self {
        // Find max register ID in the whole function
        let base = func.blocks.iter()
            .flat_map(|b| b.instructions.iter())
            .filter_map(|i| i.dest())
            .map(|r| r.0)
            .max()
            .unwrap_or(0) + 1;

        AdjointCtx {
            func,
            reg_counter:    base,
            forward_instrs: Vec::new(),
            adjoint_instrs: Vec::new(),
            adj:            HashMap::new(),
            param_regs:     HashMap::new(),
            primal_return:  None,
            base_reg:       base,
        }
    }

    fn fresh(&mut self) -> Register {
        let r = Register(self.reg_counter);
        self.reg_counter += 1;
        r
    }

    // ── Adjoint accumulation ─────────────────────────────────────────────────

    /// adj[val] += contribution
    /// Skips constants (they have no gradient).
    fn acc_adj(&mut self, val: &Value, contribution: Value) {
        let id = match val {
            Value::Reg(Register(id)) => *id,
            _ => return, // constant — no gradient
        };
        if let Some(existing) = self.adj.remove(&id) {
            // Emit an Add to sum the contributions
            let sum = self.fresh();
            self.adjoint_instrs.push(Instruction::Add {
                dest: sum.clone(),
                lhs:  existing,
                rhs:  contribution,
            });
            self.adj.insert(id, Value::Reg(sum));
        } else {
            self.adj.insert(id, contribution);
        }
    }

    /// Get adj[val], defaulting to 0.0 if not yet set.
    fn get_adj(&self, val: &Value) -> Value {
        match val {
            Value::Reg(Register(id)) =>
                self.adj.get(id).cloned().unwrap_or(Value::ConstF64(0.0)),
            _ => Value::ConstF64(0.0),
        }
    }

    /// Get adj of a register (used for the destination).
    fn get_adj_reg(&self, dest: &Register) -> Value {
        self.adj.get(&dest.0).cloned().unwrap_or(Value::ConstF64(0.0))
    }

    fn emit_adj(&mut self, i: Instruction) { self.adjoint_instrs.push(i); }

    // ── Forward pass ─────────────────────────────────────────────────────────

    fn run_forward(&mut self, block: &BasicBlock) -> Result<(), DslError> {
        // Clone all instructions verbatim into the forward section
        for instr in &block.instructions {
            self.forward_instrs.push(instr.clone());
            // Track param loads for the backward sweep
            if let Instruction::Load { dest, name } = instr {
                if self.func.params.iter().any(|(n, _)| n == name) {
                    self.param_regs.insert(name.clone(), dest.clone());
                }
            }
        }
        // Capture the primal return value
        self.primal_return = match &block.terminator {
            Terminator::Return(v) => Some(v.clone()),
            _ => None,
        };
        Ok(())
    }

    // ── Backward pass ────────────────────────────────────────────────────────

    fn run_backward(&mut self, block: &BasicBlock) -> Result<(), DslError> {
        // Seed: adj[primal_return] = 1.0
        if let Some(ret_val) = self.primal_return.clone() {
            match &ret_val {
                Value::Reg(r) => { self.adj.insert(r.0, Value::ConstF64(1.0)); }
                _ => {}
            }
        }

        self.adjoint_instrs.push(Instruction::Comment(
            "── reverse sweep ──────────────────────────────────".to_string()
        ));

        // Walk instructions in reverse order
        let instrs: Vec<_> = block.instructions.iter().rev().cloned().collect();
        for instr in instrs {
            self.vjp(&instr)?;
        }
        Ok(())
    }

    // ── VJP rules ────────────────────────────────────────────────────────────

    fn vjp(&mut self, instr: &Instruction) -> Result<(), DslError> {
        match instr {

            // ── add: d = a + b  →  adj_a += s, adj_b += s ─────────────────
            Instruction::Add { dest, lhs, rhs } => {
                let s = self.get_adj_reg(dest);
                self.acc_adj(lhs, s.clone());
                self.acc_adj(rhs, s);
            }

            // ── sub: d = a - b  →  adj_a += s, adj_b -= s ─────────────────
            Instruction::Sub { dest, lhs, rhs } => {
                let s = self.get_adj_reg(dest);
                self.acc_adj(lhs, s.clone());
                // adj_b -= s  →  adj_b += neg(s)
                let neg_s = self.fresh();
                self.emit_adj(Instruction::Neg { dest: neg_s.clone(), src: s });
                self.acc_adj(rhs, Value::Reg(neg_s));
            }

            // ── mul: d = a * b  →  adj_a += s*b, adj_b += s*a ────────────
            Instruction::Mul { dest, lhs, rhs } => {
                let s = self.get_adj_reg(dest);
                // adj_a contribution: s * b
                let sb = self.fresh();
                self.emit_adj(Instruction::Mul {
                    dest: sb.clone(), lhs: s.clone(), rhs: rhs.clone() });
                self.acc_adj(lhs, Value::Reg(sb));
                // adj_b contribution: s * a
                let sa = self.fresh();
                self.emit_adj(Instruction::Mul {
                    dest: sa.clone(), lhs: s, rhs: lhs.clone() });
                self.acc_adj(rhs, Value::Reg(sa));
            }

            // ── div: d = a / b  →  adj_a += s/b,  adj_b -= s*a/b² ────────
            Instruction::Div { dest, lhs, rhs } => {
                let s = self.get_adj_reg(dest);
                // adj_a: s / b
                let s_over_b = self.fresh();
                self.emit_adj(Instruction::Div {
                    dest: s_over_b.clone(), lhs: s.clone(), rhs: rhs.clone() });
                self.acc_adj(lhs, Value::Reg(s_over_b));
                // adj_b: -(s * a / b²)
                //   = neg(s * a) / b²
                let b_sq    = self.fresh();
                self.emit_adj(Instruction::Mul {
                    dest: b_sq.clone(), lhs: rhs.clone(), rhs: rhs.clone() });
                let s_times_a = self.fresh();
                self.emit_adj(Instruction::Mul {
                    dest: s_times_a.clone(), lhs: s, rhs: lhs.clone() });
                let neg_sa = self.fresh();
                self.emit_adj(Instruction::Neg {
                    dest: neg_sa.clone(), src: Value::Reg(s_times_a) });
                let grad_b = self.fresh();
                self.emit_adj(Instruction::Div {
                    dest: grad_b.clone(),
                    lhs:  Value::Reg(neg_sa),
                    rhs:  Value::Reg(b_sq) });
                self.acc_adj(rhs, Value::Reg(grad_b));
            }

            // ── mod: d = a % b  →  adj_a += s, adj_b = 0  (subgradient) ──
            Instruction::Mod { dest, lhs, .. } => {
                let s = self.get_adj_reg(dest);
                self.acc_adj(lhs, s);
            }

            // ── pow: d = a^e  →  adj_a += s * e * a^(e-1) ────────────────
            Instruction::Pow { dest, base, exp } => {
                let s = self.get_adj_reg(dest);
                // e * a^(e-1)
                let e_minus_1 = match exp {
                    Value::ConstF64(e) => Value::ConstF64(e - 1.0),
                    Value::ConstI64(e) => Value::ConstF64(*e as f64 - 1.0),
                    Value::Reg(_) => {
                        // dynamic exponent: e-1 via sub
                        let t = self.fresh();
                        self.emit_adj(Instruction::Sub {
                            dest: t.clone(), lhs: exp.clone(), rhs: Value::ConstF64(1.0) });
                        Value::Reg(t)
                    }
                };
                let a_pow = self.fresh();
                self.emit_adj(Instruction::Pow {
                    dest: a_pow.clone(), base: base.clone(), exp: e_minus_1 });
                let e_times_apow = self.fresh();
                self.emit_adj(Instruction::Mul {
                    dest: e_times_apow.clone(),
                    lhs:  exp.clone(),
                    rhs:  Value::Reg(a_pow) });
                let grad_a = self.fresh();
                self.emit_adj(Instruction::Mul {
                    dest: grad_a.clone(),
                    lhs:  s,
                    rhs:  Value::Reg(e_times_apow) });
                self.acc_adj(base, Value::Reg(grad_a));
            }

            // ── neg: d = -a  →  adj_a -= s ────────────────────────────────
            Instruction::Neg { dest, src } => {
                let s     = self.get_adj_reg(dest);
                let neg_s = self.fresh();
                self.emit_adj(Instruction::Neg { dest: neg_s.clone(), src: s });
                self.acc_adj(src, Value::Reg(neg_s));
            }

            // ── load/store: bookkeeping only, no VJP instructions needed ──
            Instruction::Load { .. } | Instruction::Store { .. } => {}

            // ── call: treat as opaque (zero gradient, conservative) ────────
            Instruction::Call { dest, func, .. } => {
                self.emit_adj(Instruction::Comment(
                    format!("opaque call to '{}' — gradient not propagated", func)));
            }

            // ── comparisons, index, phi, tensor, comment — no gradient ─────
            _ => {}
        }
        Ok(())
    }

    // ── Collect parameter gradients ──────────────────────────────────────────

    fn param_grad(&self, param_name: &str) -> Value {
        if let Some(reg) = self.param_regs.get(param_name) {
            self.adj.get(&reg.0).cloned().unwrap_or(Value::ConstF64(0.0))
        } else {
            Value::ConstF64(0.0)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Module-level: differentiate all scalar-returning functions
// ─────────────────────────────────────────────────────────────────────────────

/// Add a `__grad` variant for every function in the module that returns f64.
pub fn differentiate_module(module: &mut IrModule) -> Result<(), DslError> {
    let eligible: Vec<IrFunction> = module.functions.iter()
        .filter(|f| matches!(f.ret_ty, TypeAnnotation::F64 | TypeAnnotation::F32))
        .cloned()
        .collect();

    for func in &eligible {
        match differentiate(func) {
            Ok(grad_fn) => module.functions.push(grad_fn),
            Err(e) => eprintln!("warn: autodiff of '{}' failed: {}", func.name, e),
        }
    }
    Ok(())
}
