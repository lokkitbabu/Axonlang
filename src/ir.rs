// src/ir.rs — SSA-style Intermediate Representation + AST Lowering
//
// Design:
//   Each value computed is assigned to a unique virtual register (%0, %1, ...).
//   All assignments are definitions (SSA invariant: defined exactly once).
//   BasicBlocks are sequences of Instructions terminated by a Terminator.
//   Functions own a list of BasicBlocks; the entry block is index 0.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use crate::ast::*;
use crate::errors::DslError;

// ─────────────────────────────────────────────────────────────────────────────
// Core SSA types
// ─────────────────────────────────────────────────────────────────────────────

/// Virtual register: %N
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Register(pub usize);

impl std::fmt::Display for Register {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// An SSA value — either a register or an inline constant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Reg(Register),
    ConstF64(f64),
    ConstI64(i64),
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Reg(r)       => write!(f, "{}", r),
            Value::ConstF64(v)  => write!(f, "{}", v),
            Value::ConstI64(v)  => write!(f, "{}", v),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Instruction set
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Instruction {
    // Arithmetic
    Add  { dest: Register, lhs: Value, rhs: Value },
    Sub  { dest: Register, lhs: Value, rhs: Value },
    Mul  { dest: Register, lhs: Value, rhs: Value },
    Div  { dest: Register, lhs: Value, rhs: Value },
    Mod  { dest: Register, lhs: Value, rhs: Value },
    Pow  { dest: Register, base: Value, exp: Value },
    Neg  { dest: Register, src: Value },

    // Comparisons (result is 0.0 or 1.0)
    Lt   { dest: Register, lhs: Value, rhs: Value },
    Gt   { dest: Register, lhs: Value, rhs: Value },
    Le   { dest: Register, lhs: Value, rhs: Value },
    Ge   { dest: Register, lhs: Value, rhs: Value },
    Eq   { dest: Register, lhs: Value, rhs: Value },
    Ne   { dest: Register, lhs: Value, rhs: Value },

    // Memory
    Load  { dest: Register, name: String },
    Store { src: Value,     name: String },

    // Function / intrinsic calls
    Call  { dest: Register, func: String, args: Vec<Value> },

    // Tensor construction: dest = [v0, v1, ...]
    BuildTensor { dest: Register, elems: Vec<Value> },

    // Tensor index: dest = tensor[idx]
    Index { dest: Register, base: Value, idx: Value },

    // Phi node for SSA after loops/branches: dest = phi[(val, block), ...]
    Phi  { dest: Register, incoming: Vec<(Value, usize)> },

    // No-op / comment (useful for debug)
    Comment(String),
}

impl Instruction {
    /// Return the destination register if this instruction defines one.
    pub fn dest(&self) -> Option<&Register> {
        match self {
            Instruction::Add  { dest, .. } | Instruction::Sub  { dest, .. } |
            Instruction::Mul  { dest, .. } | Instruction::Div  { dest, .. } |
            Instruction::Mod  { dest, .. } | Instruction::Pow  { dest, .. } |
            Instruction::Neg  { dest, .. } | Instruction::Load { dest, .. } |
            Instruction::Call { dest, .. } | Instruction::BuildTensor { dest, .. } |
            Instruction::Index{ dest, .. } | Instruction::Phi { dest, .. } |
            Instruction::Lt   { dest, .. } | Instruction::Gt  { dest, .. } |
            Instruction::Le   { dest, .. } | Instruction::Ge  { dest, .. } |
            Instruction::Eq   { dest, .. } | Instruction::Ne  { dest, .. }
                => Some(dest),
            Instruction::Store { .. } | Instruction::Comment(_) => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Basic blocks and CFG
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Terminator {
    /// Unconditional branch to block
    Jump(usize),
    /// Conditional branch: cond != 0 → true_block, else → false_block
    CondJump { cond: Value, true_block: usize, false_block: usize },
    /// Return a value
    Return(Value),
    /// Unreachable (e.g., after diverging fn)
    Unreachable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BasicBlock {
    pub label:        String,
    pub instructions: Vec<Instruction>,
    pub terminator:   Terminator,
}

impl BasicBlock {
    pub fn new(label: impl Into<String>) -> Self {
        BasicBlock {
            label: label.into(),
            instructions: vec![],
            terminator: Terminator::Unreachable,
        }
    }
    pub fn push(&mut self, i: Instruction) {
        self.instructions.push(i);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// IR function and module
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrFunction {
    pub name:   String,
    pub params: Vec<(String, TypeAnnotation)>,
    pub ret_ty: TypeAnnotation,
    pub blocks: Vec<BasicBlock>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IrModule {
    pub functions: Vec<IrFunction>,
    /// Top-level statements compiled into an implicit __main__ function
    pub main: Vec<BasicBlock>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Lowering context: AST → SSA IR
// ─────────────────────────────────────────────────────────────────────────────

struct LowerCtx {
    /// Next register ID
    reg_counter: usize,
    /// Name → most recent register (SSA renaming)
    env: HashMap<String, Value>,
    /// Current block we're emitting into
    current_block: BasicBlock,
    /// Completed blocks
    blocks: Vec<BasicBlock>,
}

impl LowerCtx {
    fn new(entry_label: &str) -> Self {
        LowerCtx {
            reg_counter: 0,
            env: HashMap::new(),
            current_block: BasicBlock::new(entry_label),
            blocks: vec![],
        }
    }

    fn fresh(&mut self) -> Register {
        let r = Register(self.reg_counter);
        self.reg_counter += 1;
        r
    }

    fn emit(&mut self, i: Instruction) {
        self.current_block.push(i);
    }

    fn seal_block(&mut self, term: Terminator) -> usize {
        self.current_block.terminator = term;
        let idx = self.blocks.len();
        let finished = std::mem::replace(
            &mut self.current_block,
            BasicBlock::new(format!("bb{}", idx + 1)),
        );
        self.blocks.push(finished);
        idx
    }

    fn current_block_idx(&self) -> usize {
        self.blocks.len()
    }

    fn start_new_block(&mut self, label: impl Into<String>) {
        self.current_block = BasicBlock::new(label.into());
    }

    // ── Expression lowering ──────────────────────────────────────────────────

    fn lower_expr(&mut self, expr: &Expr) -> Result<Value, DslError> {
        match expr {
            Expr::Float(f)   => Ok(Value::ConstF64(*f)),
            Expr::Integer(i) => Ok(Value::ConstI64(*i)),

            Expr::Identifier(name) => {
                if let Some(v) = self.env.get(name) {
                    Ok(v.clone())
                } else {
                    // Treat unknown names as loads from memory
                    let dest = self.fresh();
                    self.emit(Instruction::Load { dest: dest.clone(), name: name.clone() });
                    Ok(Value::Reg(dest))
                }
            }

            Expr::Neg(inner) => {
                let src  = self.lower_expr(inner)?;
                let dest = self.fresh();
                self.emit(Instruction::Neg { dest: dest.clone(), src });
                Ok(Value::Reg(dest))
            }

            Expr::BinOp { op, lhs, rhs } => {
                let lv   = self.lower_expr(lhs)?;
                let rv   = self.lower_expr(rhs)?;
                let dest = self.fresh();
                let instr = match op {
                    BinOpKind::Add => Instruction::Add { dest: dest.clone(), lhs: lv, rhs: rv },
                    BinOpKind::Sub => Instruction::Sub { dest: dest.clone(), lhs: lv, rhs: rv },
                    BinOpKind::Mul => Instruction::Mul { dest: dest.clone(), lhs: lv, rhs: rv },
                    BinOpKind::Div => Instruction::Div { dest: dest.clone(), lhs: lv, rhs: rv },
                    BinOpKind::Mod => Instruction::Mod { dest: dest.clone(), lhs: lv, rhs: rv },
                    BinOpKind::Lt  => Instruction::Lt  { dest: dest.clone(), lhs: lv, rhs: rv },
                    BinOpKind::Gt  => Instruction::Gt  { dest: dest.clone(), lhs: lv, rhs: rv },
                    BinOpKind::Le  => Instruction::Le  { dest: dest.clone(), lhs: lv, rhs: rv },
                    BinOpKind::Ge  => Instruction::Ge  { dest: dest.clone(), lhs: lv, rhs: rv },
                    BinOpKind::Eq  => Instruction::Eq  { dest: dest.clone(), lhs: lv, rhs: rv },
                    BinOpKind::Ne  => Instruction::Ne  { dest: dest.clone(), lhs: lv, rhs: rv },
                };
                self.emit(instr);
                Ok(Value::Reg(dest))
            }

            Expr::Pow { base, exp } => {
                let bv   = self.lower_expr(base)?;
                let ev   = self.lower_expr(exp)?;
                let dest = self.fresh();
                self.emit(Instruction::Pow { dest: dest.clone(), base: bv, exp: ev });
                Ok(Value::Reg(dest))
            }

            Expr::Call { func, args } => {
                let arg_vals = args.iter()
                    .map(|a| self.lower_expr(a))
                    .collect::<Result<Vec<_>, _>>()?;
                let dest = self.fresh();
                self.emit(Instruction::Call {
                    dest: dest.clone(),
                    func: func.clone(),
                    args: arg_vals,
                });
                Ok(Value::Reg(dest))
            }

            Expr::Index { array, idx } => {
                let base_v = self.lower_expr(array)?;
                let idx_v  = self.lower_expr(idx)?;
                let dest   = self.fresh();
                self.emit(Instruction::Index { dest: dest.clone(), base: base_v, idx: idx_v });
                Ok(Value::Reg(dest))
            }

            Expr::TensorLit(elems) => {
                let vals = elems.iter()
                    .map(|e| self.lower_expr(e))
                    .collect::<Result<Vec<_>, _>>()?;
                let dest = self.fresh();
                self.emit(Instruction::BuildTensor { dest: dest.clone(), elems: vals });
                Ok(Value::Reg(dest))
            }
        }
    }

    // ── Statement lowering ───────────────────────────────────────────────────

    fn lower_stmt(&mut self, stmt: &Stmt) -> Result<(), DslError> {
        match stmt {
            Stmt::Assign { name, value, .. } => {
                let v = self.lower_expr(value)?;
                self.env.insert(name.clone(), v.clone());
                // Materialise into a register for consistent SSA naming
                if let Value::Reg(_) = &v {
                    // Already in a register; record the alias
                } else {
                    // Inline constant — emit a load-immediate via store+load cycle
                    // (simpler: just record the constant in env)
                }
                self.emit(Instruction::Store { src: v, name: name.clone() });
                Ok(())
            }

            Stmt::Return(expr) => {
                let v = self.lower_expr(expr)?;
                self.seal_block(Terminator::Return(v));
                // Start a new (likely dead) block to absorb any trailing stmts
                self.start_new_block("ret_tail");
                Ok(())
            }

            Stmt::ExprStmt(expr) => {
                self.lower_expr(expr)?;
                Ok(())
            }

            Stmt::ForLoop { var, lo, hi, body } => {
                // ── Pattern:
                //   preheader: compute lo, hi; jump → header
                //   header:    phi(i); cmp i < hi; condjump → body | exit
                //   body:      ... ; i' = i+1; jump → header
                //   exit:      ...
                let lo_v = self.lower_expr(lo)?;
                let hi_v = self.lower_expr(hi)?;

                let preheader_idx = self.seal_block(Terminator::Jump(self.current_block_idx() + 1));
                let header_idx    = self.current_block_idx();
                self.start_new_block(format!("loop_header_{}", var));

                // Phi for induction variable
                let phi_reg = self.fresh();
                // We'll patch the phi incoming values after we know body block indices
                let phi_placeholder_pos = self.current_block.instructions.len();
                self.emit(Instruction::Phi {
                    dest: phi_reg.clone(),
                    incoming: vec![(lo_v.clone(), preheader_idx)], // body edge patched below
                });
                self.env.insert(var.clone(), Value::Reg(phi_reg.clone()));

                // Loop condition: i < hi
                let cond_reg = self.fresh();
                self.emit(Instruction::Lt {
                    dest: cond_reg.clone(),
                    lhs: Value::Reg(phi_reg.clone()),
                    rhs: hi_v,
                });
                let body_start = self.current_block_idx() + 1;
                let exit_placeholder = 9999; // patched after body
                self.seal_block(Terminator::CondJump {
                    cond: Value::Reg(cond_reg),
                    true_block: body_start,
                    false_block: exit_placeholder,
                });

                // Body
                self.start_new_block(format!("loop_body_{}", var));
                for s in body {
                    self.lower_stmt(s)?;
                }
                // Increment induction variable
                let inc_reg = self.fresh();
                self.emit(Instruction::Add {
                    dest: inc_reg.clone(),
                    lhs: Value::Reg(phi_reg.clone()),
                    rhs: Value::ConstI64(1),
                });
                self.env.insert(var.clone(), Value::Reg(inc_reg.clone()));
                let body_end_idx = self.seal_block(Terminator::Jump(header_idx));

                // Patch phi to include back-edge
                if let Some(Instruction::Phi { incoming, .. }) =
                    self.blocks.get_mut(header_idx)
                        .and_then(|b| b.instructions.get_mut(phi_placeholder_pos))
                {
                    incoming.push((Value::Reg(inc_reg), body_end_idx));
                }

                // Exit block
                let exit_idx = self.current_block_idx();
                self.start_new_block(format!("loop_exit_{}", var));

                // Patch condjump false_block in header
                if let Some(Terminator::CondJump { false_block, .. }) =
                    self.blocks.get_mut(header_idx).map(|b| &mut b.terminator)
                {
                    *false_block = exit_idx;
                }

                Ok(())
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public lowering API
// ─────────────────────────────────────────────────────────────────────────────

pub fn lower_program(program: &Program) -> Result<IrModule, DslError> {
    let mut module = IrModule::default();

    // Lower top-level function defs
    for fn_def in program.fns() {
        let ir_fn = lower_fn(fn_def)?;
        module.functions.push(ir_fn);
    }

    // Lower top-level statements into __main__
    let stmts: Vec<&Stmt> = program.stmts().collect();
    if !stmts.is_empty() {
        let mut ctx = LowerCtx::new("entry");
        for stmt in &stmts {
            ctx.lower_stmt(stmt)?;
        }
        // If no return emitted, seal with Return(0)
        if ctx.blocks.is_empty() || !matches!(
            ctx.blocks.last().unwrap().terminator, Terminator::Return(_))
        {
            ctx.seal_block(Terminator::Return(Value::ConstI64(0)));
        }
        module.main = ctx.blocks;
    }

    Ok(module)
}

fn lower_fn(fn_def: &FnDef) -> Result<IrFunction, DslError> {
    let mut ctx = LowerCtx::new("entry");

    // Bind params as loads
    for param in &fn_def.params {
        let dest = ctx.fresh();
        ctx.emit(Instruction::Load { dest: dest.clone(), name: param.name.clone() });
        ctx.env.insert(param.name.clone(), Value::Reg(dest));
    }

    for stmt in &fn_def.body {
        ctx.lower_stmt(stmt)?;
    }

    if ctx.blocks.is_empty() {
        // Implicit return
        ctx.seal_block(Terminator::Return(Value::ConstI64(0)));
    }

    Ok(IrFunction {
        name:   fn_def.name.clone(),
        params: fn_def.params.iter().map(|p| (p.name.clone(), p.ty.clone())).collect(),
        ret_ty: fn_def.ret_ty.clone(),
        blocks: ctx.blocks,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// IR pretty-printer
// ─────────────────────────────────────────────────────────────────────────────

pub fn print_module(module: &IrModule) -> String {
    let mut out = String::new();

    for func in &module.functions {
        let params: Vec<String> = func.params.iter()
            .map(|(n, t)| format!("{}: {}", n, t))
            .collect();
        out.push_str(&format!("fn {}({}) -> {} {{\n", func.name, params.join(", "), func.ret_ty));
        for block in &func.blocks {
            out.push_str(&print_block(block));
        }
        out.push_str("}\n\n");
    }

    if !module.main.is_empty() {
        out.push_str("fn __main__() {\n");
        for block in &module.main {
            out.push_str(&print_block(block));
        }
        out.push_str("}\n");
    }

    out
}

fn print_block(block: &BasicBlock) -> String {
    let mut out = format!("  {}:\n", block.label);
    for instr in &block.instructions {
        out.push_str(&format!("    {}\n", print_instr(instr)));
    }
    out.push_str(&format!("    {}\n", print_term(&block.terminator)));
    out
}

fn print_instr(i: &Instruction) -> String {
    match i {
        Instruction::Add  { dest, lhs, rhs } => format!("{} = add  {}, {}", dest, lhs, rhs),
        Instruction::Sub  { dest, lhs, rhs } => format!("{} = sub  {}, {}", dest, lhs, rhs),
        Instruction::Mul  { dest, lhs, rhs } => format!("{} = mul  {}, {}", dest, lhs, rhs),
        Instruction::Div  { dest, lhs, rhs } => format!("{} = div  {}, {}", dest, lhs, rhs),
        Instruction::Mod  { dest, lhs, rhs } => format!("{} = mod  {}, {}", dest, lhs, rhs),
        Instruction::Pow  { dest, base, exp } => format!("{} = pow  {}, {}", dest, base, exp),
        Instruction::Neg  { dest, src }       => format!("{} = neg  {}",      dest, src),
        Instruction::Lt   { dest, lhs, rhs } => format!("{} = lt   {}, {}", dest, lhs, rhs),
        Instruction::Gt   { dest, lhs, rhs } => format!("{} = gt   {}, {}", dest, lhs, rhs),
        Instruction::Le   { dest, lhs, rhs } => format!("{} = le   {}, {}", dest, lhs, rhs),
        Instruction::Ge   { dest, lhs, rhs } => format!("{} = ge   {}, {}", dest, lhs, rhs),
        Instruction::Eq   { dest, lhs, rhs } => format!("{} = eq   {}, {}", dest, lhs, rhs),
        Instruction::Ne   { dest, lhs, rhs } => format!("{} = ne   {}, {}", dest, lhs, rhs),
        Instruction::Load { dest, name }      => format!("{} = load @{}", dest, name),
        Instruction::Store{ src, name }       => format!("store {}, @{}", src, name),
        Instruction::Call { dest, func, args } => {
            let arg_s: Vec<_> = args.iter().map(|a| a.to_string()).collect();
            format!("{} = call {}({})", dest, func, arg_s.join(", "))
        }
        Instruction::BuildTensor { dest, elems } => {
            let e: Vec<_> = elems.iter().map(|v| v.to_string()).collect();
            format!("{} = tensor[{}]", dest, e.join(", "))
        }
        Instruction::Index { dest, base, idx } => format!("{} = index {}, {}", dest, base, idx),
        Instruction::Phi { dest, incoming } => {
            let parts: Vec<_> = incoming.iter()
                .map(|(v, b)| format!("[{}, bb{}]", v, b))
                .collect();
            format!("{} = phi {}", dest, parts.join(", "))
        }
        Instruction::Comment(s) => format!("; {}", s),
    }
}

fn print_term(t: &Terminator) -> String {
    match t {
        Terminator::Jump(b)                           => format!("jump bb{}", b),
        Terminator::Return(v)                         => format!("ret {}", v),
        Terminator::CondJump { cond, true_block, false_block } =>
            format!("condjump {}, bb{}, bb{}", cond, true_block, false_block),
        Terminator::Unreachable                       => "unreachable".to_string(),
    }
}
