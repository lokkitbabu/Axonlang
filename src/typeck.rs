// src/typeck.rs — Type Inference and Checking
//
// Infers the "runtime type" (f64 or i64) of every SSA register in the module.
// This is a forward dataflow analysis: we start with known types (constants,
// parameter loads) and propagate through arithmetic operations.
//
// Rules:
//   f64 op f64 → f64
//   i64 op i64 → i64
//   f64 op i64 → f64   (implicit widening)
//   pow(_, _)  → f64   (always)
//   lt/gt/etc  → i64   (boolean 0/1)
//   load       → f64   (conservative default — most scientific vars are f64)
//   call       → f64

use std::collections::HashMap;
use crate::ir::*;

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum RegTy {
    F64,
    I64,
    Unknown,
}

impl RegTy {
    fn join(a: RegTy, b: RegTy) -> RegTy {
        match (a, b) {
            (RegTy::F64, _) | (_, RegTy::F64)     => RegTy::F64,
            (RegTy::I64, RegTy::I64)               => RegTy::I64,
            _                                       => RegTy::Unknown,
        }
    }
}

impl std::fmt::Display for RegTy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { RegTy::F64 => write!(f, "f64"), RegTy::I64 => write!(f, "i64"), RegTy::Unknown => write!(f, "?") }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Type environment
// ─────────────────────────────────────────────────────────────────────────────

pub type TypeEnv = HashMap<usize, RegTy>;  // reg_id → type

pub fn infer_value(v: &Value, env: &TypeEnv) -> RegTy {
    match v {
        Value::ConstF64(_)       => RegTy::F64,
        Value::ConstI64(_)       => RegTy::I64,
        Value::Reg(Register(id)) => env.get(id).copied().unwrap_or(RegTy::Unknown),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Block-level inference
// ─────────────────────────────────────────────────────────────────────────────

pub fn infer_block(block: &BasicBlock, env: &mut TypeEnv) {
    for instr in &block.instructions {
        infer_instr(instr, env);
    }
}

fn infer_instr(instr: &Instruction, env: &mut TypeEnv) {
    match instr {
        Instruction::Add { dest, lhs, rhs } | Instruction::Sub { dest, lhs, rhs } |
        Instruction::Mul { dest, lhs, rhs } | Instruction::Mod { dest, lhs, rhs } => {
            let ty = RegTy::join(infer_value(lhs, env), infer_value(rhs, env));
            env.insert(dest.0, ty);
        }
        Instruction::Div { dest, lhs, rhs } => {
            // Division always produces f64 to avoid integer truncation confusion
            let _ = (lhs, rhs);
            env.insert(dest.0, RegTy::F64);
        }
        Instruction::Pow { dest, .. } => { env.insert(dest.0, RegTy::F64); }
        Instruction::Neg { dest, src } => {
            env.insert(dest.0, infer_value(src, env));
        }

        // Comparisons produce boolean-as-int (i64 0/1)
        Instruction::Lt { dest, .. } | Instruction::Gt { dest, .. } |
        Instruction::Le { dest, .. } | Instruction::Ge { dest, .. } |
        Instruction::Eq { dest, .. } | Instruction::Ne { dest, .. } => {
            env.insert(dest.0, RegTy::I64);
        }

        Instruction::Load { dest, .. } => { env.insert(dest.0, RegTy::F64); }
        Instruction::Call { dest, .. } => { env.insert(dest.0, RegTy::F64); }

        Instruction::Index { dest, .. } => { env.insert(dest.0, RegTy::F64); }
        Instruction::BuildTensor { dest, .. } => { env.insert(dest.0, RegTy::F64); }

        Instruction::Phi { dest, incoming } => {
            let ty = incoming.iter()
                .map(|(v, _)| infer_value(v, env))
                .fold(RegTy::Unknown, RegTy::join);
            env.insert(dest.0, ty);
        }

        Instruction::Store { .. } | Instruction::Comment(_) => {}
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Type checking — produce diagnostics
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct TypeError {
    pub message: String,
}

pub fn typecheck_block(block: &BasicBlock, env: &mut TypeEnv) -> Vec<TypeError> {
    let mut errors = Vec::new();

    for instr in &block.instructions {
        match instr {
            Instruction::Div { dest: _, lhs, rhs } => {
                // Warn: integer / integer is suspicious in scientific code
                if infer_value(lhs, env) == RegTy::I64
                    && infer_value(rhs, env) == RegTy::I64 {
                    errors.push(TypeError {
                        message: "integer division may truncate; consider using f64 literals".to_string(),
                    });
                }
            }
            Instruction::Pow { dest: _, base, exp } => {
                if infer_value(base, env) == RegTy::I64 {
                    errors.push(TypeError {
                        message: "pow of integer base — result promoted to f64, but base may need cast".to_string(),
                    });
                }
            }
            _ => {}
        }
        infer_instr(instr, env);
    }
    errors
}

// ─────────────────────────────────────────────────────────────────────────────
// Module-level — infer types for all functions and main
// ─────────────────────────────────────────────────────────────────────────────

pub struct ModuleTypes {
    /// function name → per-register type environment
    pub functions: HashMap<String, TypeEnv>,
    pub main: TypeEnv,
}

pub fn infer_module(module: &IrModule) -> ModuleTypes {
    let mut mt = ModuleTypes {
        functions: HashMap::new(),
        main:      HashMap::new(),
    };

    for func in &module.functions {
        let mut env = TypeEnv::new();
        for block in &func.blocks {
            infer_block(block, &mut env);
        }
        mt.functions.insert(func.name.clone(), env);
    }

    for block in &module.main {
        infer_block(block, &mut mt.main);
    }

    mt
}

/// Annotate the IR pretty-print with inferred types (returns annotated string).
pub fn annotate_ir(module: &IrModule) -> String {
    let mt     = infer_module(module);
    let ir_str = crate::ir::print_module(module);

    // Simple post-processing: append type annotation after each register def
    // Format:  %3 = mul %1, %2     →   %3:f64 = mul %1, %2
    let mut out = String::new();
    for line in ir_str.lines() {
        let trimmed = line.trim_start();
        // Detect lines like "  %N = ..."
        if trimmed.starts_with('%') {
            if let Some(eq_pos) = trimmed.find('=') {
                let reg_str = trimmed[..eq_pos].trim();
                if let Some(id_str) = reg_str.strip_prefix('%') {
                    if let Ok(id) = id_str.parse::<usize>() {
                        // Find in any function env or main
                        let ty = mt.functions.values()
                            .find_map(|env| env.get(&id))
                            .or_else(|| mt.main.get(&id))
                            .copied()
                            .unwrap_or(RegTy::Unknown);
                        let annotated = line.replacen(
                            &format!("%{} =", id),
                            &format!("%{}:{} =", id, ty),
                            1
                        );
                        out.push_str(&annotated);
                        out.push('\n');
                        continue;
                    }
                }
            }
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}
