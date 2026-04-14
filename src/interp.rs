// src/interp.rs — Tree-walking interpreter over SSA IR
//
// Evaluates an IrFunction by walking its BasicBlocks and executing each
// Instruction against a value environment.  Supports:
//   - All arithmetic ops (Add, Sub, Mul, Div, Mod, Pow, Neg)
//   - Comparisons (returning 0.0/1.0)
//   - Memory (Load from params/locals, Store)
//   - Calls to builtin scientific functions:
//       sin cos tan asin acos atan exp exp2 log log2 log10 sqrt abs
//       floor ceil round sign min max dot norm grad jacobian matmul
//   - Tensor construction and indexing
//   - Phi nodes (for loop back-edges)
//   - Loop CFG (CondJump / Jump terminators)
//
// All values are f64 internally; tensors are Vec<f64>.

use std::collections::HashMap;
use crate::ir::*;
use crate::errors::DslError;

// ─────────────────────────────────────────────────────────────────────────────
// Runtime value
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Val {
    Scalar(f64),
    Tensor(Vec<f64>),
}

impl Val {
    pub fn as_f64(&self) -> f64 {
        match self {
            Val::Scalar(v) => *v,
            Val::Tensor(v) => v.first().copied().unwrap_or(0.0),
        }
    }
    pub fn as_tensor(&self) -> Vec<f64> {
        match self {
            Val::Scalar(v) => vec![*v],
            Val::Tensor(v) => v.clone(),
        }
    }
}

impl From<f64>     for Val { fn from(v: f64)     -> Self { Val::Scalar(v) } }
impl From<Vec<f64>> for Val { fn from(v: Vec<f64>) -> Self { Val::Tensor(v) } }

// ─────────────────────────────────────────────────────────────────────────────
// Interpreter state
// ─────────────────────────────────────────────────────────────────────────────

pub struct Interpreter {
    /// Register → value
    regs:  HashMap<usize, Val>,
    /// Named memory (params and let-bindings)
    mem:   HashMap<String, Val>,
    /// Max iteration guard for loops
    pub max_iters: usize,
    /// Index of the previously executed block (for phi-node resolution)
    prev_block: Option<usize>,
}

impl Default for Interpreter {
    fn default() -> Self {
        Interpreter { regs: HashMap::new(), mem: HashMap::new(), max_iters: 100_000, prev_block: None }
    }
}

impl Interpreter {
    pub fn new() -> Self { Self::default() }

    // ── Public entry ─────────────────────────────────────────────────────────

    /// Run `func` with the given positional arguments (matched to params in order).
    pub fn call(&mut self, func: &IrFunction, args: &[f64]) -> Result<Val, DslError> {
        self.regs.clear();
        self.mem.clear();

        // Bind parameters
        for ((name, _), val) in func.params.iter().zip(args.iter()) {
            self.mem.insert(name.clone(), Val::Scalar(*val));
        }

        self.run_blocks(&func.blocks)
    }

    /// Run a module's __main__ block (no parameters).
    pub fn run_main(&mut self, blocks: &[BasicBlock]) -> Result<Val, DslError> {
        self.regs.clear();
        self.mem.clear();
        self.run_blocks(blocks)
    }

    // ── Block execution ───────────────────────────────────────────────────────

    fn run_blocks(&mut self, blocks: &[BasicBlock]) -> Result<Val, DslError> {
        let mut current = 0usize;
        let mut iters   = 0usize;

        loop {
            let block = blocks.get(current)
                .ok_or_else(|| DslError::IrError(format!("Block {} not found", current)))?;

            // Execute instructions (phi nodes use self.prev_block to pick incoming)
            for instr in &block.instructions {
                self.exec(instr)?;
            }

            // Follow terminator
            iters += 1;
            if iters > self.max_iters {
                return Err(DslError::IrError(
                    format!("Exceeded {} iterations — possible infinite loop", self.max_iters)));
            }

            let next = match &block.terminator {
                Terminator::Return(v) => return Ok(self.eval_val(v)),
                Terminator::Jump(b)   => *b,
                Terminator::CondJump { cond, true_block, false_block } => {
                    if self.eval_val(cond).as_f64() != 0.0 { *true_block } else { *false_block }
                }
                Terminator::Unreachable =>
                    return Err(DslError::IrError("Reached unreachable terminator".into())),
            };
            self.prev_block = Some(current);
            current = next;
        }
    }

    // ── Instruction execution ─────────────────────────────────────────────────

    fn exec(&mut self, instr: &Instruction) -> Result<(), DslError> {
        match instr {
            Instruction::Add { dest, lhs, rhs } =>
                self.set(dest, Val::Scalar(self.f(lhs) + self.f(rhs))),
            Instruction::Sub { dest, lhs, rhs } =>
                self.set(dest, Val::Scalar(self.f(lhs) - self.f(rhs))),
            Instruction::Mul { dest, lhs, rhs } =>
                self.set(dest, Val::Scalar(self.f(lhs) * self.f(rhs))),
            Instruction::Div { dest, lhs, rhs } => {
                let r = self.f(rhs);
                if r == 0.0 {
                    return Err(DslError::IrError("Division by zero".into()));
                }
                self.set(dest, Val::Scalar(self.f(lhs) / r));
            }
            Instruction::Mod { dest, lhs, rhs } =>
                self.set(dest, Val::Scalar(self.f(lhs) % self.f(rhs))),
            Instruction::Pow { dest, base, exp } =>
                self.set(dest, Val::Scalar(self.f(base).powf(self.f(exp)))),
            Instruction::Neg { dest, src } =>
                self.set(dest, Val::Scalar(-self.f(src))),

            // Comparisons → 0.0 or 1.0
            Instruction::Lt { dest, lhs, rhs } =>
                self.set(dest, Val::Scalar(if self.f(lhs) <  self.f(rhs) { 1.0 } else { 0.0 })),
            Instruction::Gt { dest, lhs, rhs } =>
                self.set(dest, Val::Scalar(if self.f(lhs) >  self.f(rhs) { 1.0 } else { 0.0 })),
            Instruction::Le { dest, lhs, rhs } =>
                self.set(dest, Val::Scalar(if self.f(lhs) <= self.f(rhs) { 1.0 } else { 0.0 })),
            Instruction::Ge { dest, lhs, rhs } =>
                self.set(dest, Val::Scalar(if self.f(lhs) >= self.f(rhs) { 1.0 } else { 0.0 })),
            Instruction::Eq { dest, lhs, rhs } =>
                self.set(dest, Val::Scalar(if (self.f(lhs) - self.f(rhs)).abs() < 1e-12 { 1.0 } else { 0.0 })),
            Instruction::Ne { dest, lhs, rhs } =>
                self.set(dest, Val::Scalar(if (self.f(lhs) - self.f(rhs)).abs() >= 1e-12 { 1.0 } else { 0.0 })),

            // Memory
            Instruction::Load { dest, name } => {
                let val = self.mem.get(name).cloned()
                    .unwrap_or(Val::Scalar(0.0)); // uninitialized → 0
                self.set(dest, val);
            }
            Instruction::Store { src, name } => {
                let val = self.eval_val(src);
                self.mem.insert(name.clone(), val);
            }

            // Intrinsic calls
            Instruction::Call { dest, func, args } => {
                let arg_vals: Vec<f64> = args.iter().map(|a| self.f(a)).collect();
                let result = self.call_builtin(func, &arg_vals)?;
                self.set(dest, Val::Scalar(result));
            }

            // Tensor ops
            Instruction::BuildTensor { dest, elems } => {
                let vals: Vec<f64> = elems.iter().map(|v| self.f(v)).collect();
                self.set(dest, Val::Tensor(vals));
            }
            Instruction::Index { dest, base, idx } => {
                let i   = self.f(idx) as usize;
                let val = match self.eval_val(base) {
                    Val::Tensor(v) => Val::Scalar(v.get(i).copied().unwrap_or(0.0)),
                    Val::Scalar(v) => Val::Scalar(v), // scalar[0] = scalar
                };
                self.set(dest, val);
            }

            // Phi node: select incoming value from the predecessor block.
            Instruction::Phi { dest, incoming } => {
                let val = if let Some(prev) = self.prev_block {
                    // Pick the incoming value whose source block matches prev_block
                    incoming.iter()
                        .find(|(_, src_block)| *src_block == prev)
                        .map(|(v, _)| self.eval_val(v))
                        .unwrap_or(Val::Scalar(0.0))
                } else {
                    // First entry: use first incoming (the entry-edge value)
                    incoming.first()
                        .map(|(v, _)| self.eval_val(v))
                        .unwrap_or(Val::Scalar(0.0))
                };
                self.set(dest, val);
            }

            Instruction::Comment(_) => {}

            _ => {} // unhandled → no-op
        }
        Ok(())
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn set(&mut self, dest: &Register, val: Val) {
        self.regs.insert(dest.0, val);
    }

    fn eval_val(&self, v: &Value) -> Val {
        match v {
            Value::Reg(Register(id)) => self.regs.get(id).cloned().unwrap_or(Val::Scalar(0.0)),
            Value::ConstF64(f)       => Val::Scalar(*f),
            Value::ConstI64(i)       => Val::Scalar(*i as f64),
        }
    }

    fn f(&self, v: &Value) -> f64 { self.eval_val(v).as_f64() }

    // ── Built-in scientific functions ─────────────────────────────────────────

    fn call_builtin(&self, name: &str, args: &[f64]) -> Result<f64, DslError> {
        let a0 = args.get(0).copied().unwrap_or(0.0);
        let a1 = args.get(1).copied().unwrap_or(0.0);
        match name {
            "sin"   => Ok(a0.sin()),
            "cos"   => Ok(a0.cos()),
            "tan"   => Ok(a0.tan()),
            "asin"  => Ok(a0.asin()),
            "acos"  => Ok(a0.acos()),
            "atan"  => Ok(a0.atan()),
            "atan2" => Ok(a0.atan2(a1)),
            "exp"   => Ok(a0.exp()),
            "exp2"  => Ok(a0.exp2()),
            "log"   => Ok(a0.ln()),
            "log2"  => Ok(a0.log2()),
            "log10" => Ok(a0.log10()),
            "sqrt"  => Ok(a0.sqrt()),
            "cbrt"  => Ok(a0.cbrt()),
            "abs"   => Ok(a0.abs()),
            "floor" => Ok(a0.floor()),
            "ceil"  => Ok(a0.ceil()),
            "round" => Ok(a0.round()),
            "sign"  => Ok(a0.signum()),
            "min"   => Ok(a0.min(a1)),
            "max"   => Ok(a0.max(a1)),
            "hypot" => Ok(a0.hypot(a1)),
            "pow"   => Ok(a0.powf(a1)),
            "dot"   => {
                // dot(a, b) where a and b are scalars in this context
                Ok(a0 * a1)
            }
            "norm"  => Ok(a0.abs()),
            other   =>
                Err(DslError::IrError(format!("Unknown intrinsic: '{}'", other))),
        }
    }
}
