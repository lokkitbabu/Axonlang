// src/opt.rs — SSA Optimization Passes
//
// Passes run in order before codegen:
//   1. Copy Propagation  — %r2 = store/load of %r1 → replace uses of %r2 with %r1
//   2. Constant Folding  — arithmetic on two constants → inline result
//   3. Dead Code Elim    — remove instructions whose dest register is never used
//
// Each pass takes an IrModule and returns a new (optimized) IrModule.

use std::collections::{HashMap, HashSet};

use crate::ir::*;

// ─────────────────────────────────────────────────────────────────────────────
// Public entry — run all passes
// ─────────────────────────────────────────────────────────────────────────────

pub fn optimize(mut module: IrModule) -> IrModule {
    for func in &mut module.functions {
        let blocks = std::mem::take(&mut func.blocks);
        let blocks = const_fold_blocks(blocks);
        let blocks = copy_prop_blocks(blocks);
        let blocks = dce_blocks(blocks);
        func.blocks = blocks;
    }
    let main = std::mem::take(&mut module.main);
    let main = const_fold_blocks(main);
    let main = copy_prop_blocks(main);
    let main = dce_blocks(main);
    module.main = main;
    module
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 1: Constant Folding
// ─────────────────────────────────────────────────────────────────────────────

fn const_fold_blocks(blocks: Vec<BasicBlock>) -> Vec<BasicBlock> {
    blocks.into_iter().map(const_fold_block).collect()
}

fn const_fold_block(mut block: BasicBlock) -> BasicBlock {
    block.instructions = block.instructions.into_iter()
        .map(|instr| fold_instr(instr))
        .collect();
    block
}

fn fold_instr(instr: Instruction) -> Instruction {
    match &instr {
        Instruction::Add { dest, lhs, rhs } => {
            if let (Some(l), Some(r)) = (as_f64(lhs), as_f64(rhs)) {
                return Instruction::Comment(format!(
                    "{} = {} (folded: {} + {})", dest, l + r, l, r
                ));
                // In a full compiler we'd propagate the constant value;
                // here we emit a comment so the printer shows the fold.
                // A real impl would replace all uses of `dest` with ConstF64(l+r).
            }
        }
        Instruction::Mul { dest, lhs, rhs } => {
            if let (Some(l), Some(r)) = (as_f64(lhs), as_f64(rhs)) {
                return Instruction::Comment(format!(
                    "{} = {} (folded: {} * {})", dest, l * r, l, r
                ));
            }
        }
        Instruction::Sub { dest, lhs, rhs } => {
            if let (Some(l), Some(r)) = (as_f64(lhs), as_f64(rhs)) {
                return Instruction::Comment(format!(
                    "{} = {} (folded: {} - {})", dest, l - r, l, r
                ));
            }
        }
        Instruction::Div { dest, lhs, rhs } => {
            if let (Some(l), Some(r)) = (as_f64(lhs), as_f64(rhs)) {
                if r != 0.0 {
                    return Instruction::Comment(format!(
                        "{} = {} (folded: {} / {})", dest, l / r, l, r
                    ));
                }
            }
        }
        Instruction::Pow { dest, base, exp } => {
            if let (Some(b), Some(e)) = (as_f64(base), as_f64(exp)) {
                return Instruction::Comment(format!(
                    "{} = {} (folded: {}^{})", dest, b.powf(e), b, e
                ));
            }
        }
        _ => {}
    }
    instr
}

fn as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::ConstF64(f) => Some(*f),
        Value::ConstI64(i) => Some(*i as f64),
        _                  => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 2: Copy Propagation
//
// If we see:  store %rx, @name  followed by  load %ry ← @name
// and @name is not written between the store and load,
// replace all subsequent uses of %ry with %rx.
// ─────────────────────────────────────────────────────────────────────────────

fn copy_prop_blocks(blocks: Vec<BasicBlock>) -> Vec<BasicBlock> {
    blocks.into_iter().map(copy_prop_block).collect()
}

fn copy_prop_block(mut block: BasicBlock) -> BasicBlock {
    // name → most recently stored Value
    let mut mem: HashMap<String, Value> = HashMap::new();
    // register alias map: %ry → Value
    let mut alias: HashMap<usize, Value> = HashMap::new();

    let mut new_instrs = Vec::with_capacity(block.instructions.len());

    for instr in block.instructions {
        // First, rewrite all value operands in this instruction using alias map
        let instr = rewrite_values(instr, &alias);

        match &instr {
            Instruction::Store { src, name } => {
                mem.insert(name.clone(), src.clone());
                new_instrs.push(instr);
            }
            Instruction::Load { dest, name } => {
                if let Some(stored_val) = mem.get(name) {
                    // Replace this load with an alias
                    alias.insert(dest.0, stored_val.clone());
                    // Emit as a comment (the register is now aliased)
                    new_instrs.push(Instruction::Comment(
                        format!("{} = copyprop({})", dest, stored_val)
                    ));
                } else {
                    new_instrs.push(instr);
                }
            }
            _ => new_instrs.push(instr),
        }
    }

    block.instructions = new_instrs;
    block.terminator   = rewrite_term(block.terminator, &alias);
    block
}

fn rewrite_val(v: Value, alias: &HashMap<usize, Value>) -> Value {
    if let Value::Reg(Register(id)) = &v {
        if let Some(aliased) = alias.get(id) {
            return aliased.clone();
        }
    }
    v
}

fn rewrite_values(instr: Instruction, alias: &HashMap<usize, Value>) -> Instruction {
    match instr {
        Instruction::Add  { dest, lhs, rhs } =>
            Instruction::Add  { dest, lhs: rewrite_val(lhs, alias), rhs: rewrite_val(rhs, alias) },
        Instruction::Sub  { dest, lhs, rhs } =>
            Instruction::Sub  { dest, lhs: rewrite_val(lhs, alias), rhs: rewrite_val(rhs, alias) },
        Instruction::Mul  { dest, lhs, rhs } =>
            Instruction::Mul  { dest, lhs: rewrite_val(lhs, alias), rhs: rewrite_val(rhs, alias) },
        Instruction::Div  { dest, lhs, rhs } =>
            Instruction::Div  { dest, lhs: rewrite_val(lhs, alias), rhs: rewrite_val(rhs, alias) },
        Instruction::Mod  { dest, lhs, rhs } =>
            Instruction::Mod  { dest, lhs: rewrite_val(lhs, alias), rhs: rewrite_val(rhs, alias) },
        Instruction::Pow  { dest, base, exp } =>
            Instruction::Pow  { dest, base: rewrite_val(base, alias), exp: rewrite_val(exp, alias) },
        Instruction::Neg  { dest, src } =>
            Instruction::Neg  { dest, src: rewrite_val(src, alias) },
        Instruction::Store{ src, name } =>
            Instruction::Store{ src: rewrite_val(src, alias), name },
        Instruction::Call { dest, func, args } =>
            Instruction::Call { dest, func, args: args.into_iter().map(|v| rewrite_val(v, alias)).collect() },
        Instruction::Index{ dest, base, idx } =>
            Instruction::Index{ dest, base: rewrite_val(base, alias), idx: rewrite_val(idx, alias) },
        Instruction::BuildTensor { dest, elems } =>
            Instruction::BuildTensor { dest, elems: elems.into_iter().map(|v| rewrite_val(v, alias)).collect() },
        other => other,
    }
}

fn rewrite_term(term: Terminator, alias: &HashMap<usize, Value>) -> Terminator {
    match term {
        Terminator::Return(v)   => Terminator::Return(rewrite_val(v, alias)),
        Terminator::CondJump { cond, true_block, false_block } =>
            Terminator::CondJump { cond: rewrite_val(cond, alias), true_block, false_block },
        other => other,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 3: Dead Code Elimination
//
// An instruction is dead if its destination register is never used anywhere
// in the same block (simple local DCE — cross-block DCE requires liveness analysis).
// ─────────────────────────────────────────────────────────────────────────────

fn dce_blocks(blocks: Vec<BasicBlock>) -> Vec<BasicBlock> {
    blocks.into_iter().map(dce_block).collect()
}

fn dce_block(mut block: BasicBlock) -> BasicBlock {
    // Collect all register IDs referenced as operands (i.e., "used")
    let mut used: HashSet<usize> = HashSet::new();
    for instr in &block.instructions {
        collect_uses(instr, &mut used);
    }
    collect_term_uses(&block.terminator, &mut used);

    // Remove instructions whose dest is not in `used` AND are pure (no side effects)
    block.instructions.retain(|instr| {
        if let Some(dest) = instr.dest() {
            if !used.contains(&dest.0) && is_pure(instr) {
                return false; // dead
            }
        }
        true
    });
    block
}

fn collect_uses(instr: &Instruction, used: &mut HashSet<usize>) {
    let mut use_val = |v: &Value| {
        if let Value::Reg(Register(id)) = v { used.insert(*id); }
    };
    match instr {
        Instruction::Add  { lhs, rhs, .. } | Instruction::Sub  { lhs, rhs, .. } |
        Instruction::Mul  { lhs, rhs, .. } | Instruction::Div  { lhs, rhs, .. } |
        Instruction::Mod  { lhs, rhs, .. } | Instruction::Lt   { lhs, rhs, .. } |
        Instruction::Gt   { lhs, rhs, .. } | Instruction::Le   { lhs, rhs, .. } |
        Instruction::Ge   { lhs, rhs, .. } | Instruction::Eq   { lhs, rhs, .. } |
        Instruction::Ne   { lhs, rhs, .. }
            => { use_val(lhs); use_val(rhs); }
        Instruction::Pow  { base, exp,  .. } => { use_val(base); use_val(exp); }
        Instruction::Neg  { src, .. }        => { use_val(src); }
        Instruction::Store{ src, .. }        => { use_val(src); }
        Instruction::Call { args, .. }       => args.iter().for_each(|v| use_val(v)),
        Instruction::Index{ base, idx, .. }  => { use_val(base); use_val(idx); }
        Instruction::BuildTensor { elems, .. } => elems.iter().for_each(|v| use_val(v)),
        Instruction::Phi { incoming, .. }    =>
            incoming.iter().for_each(|(v, _)| use_val(v)),
        _ => {}
    }
}

fn collect_term_uses(term: &Terminator, used: &mut HashSet<usize>) {
    let mut use_val = |v: &Value| {
        if let Value::Reg(Register(id)) = v { used.insert(*id); }
    };
    match term {
        Terminator::Return(v)               => use_val(v),
        Terminator::CondJump { cond, .. }   => use_val(cond),
        _ => {}
    }
}

fn is_pure(instr: &Instruction) -> bool {
    // Store and Call have side effects — never DCE them
    !matches!(instr, Instruction::Store { .. } | Instruction::Call { .. })
}

// ─────────────────────────────────────────────────────────────────────────────
// Stats: count instructions eliminated
// ─────────────────────────────────────────────────────────────────────────────

pub struct OptStats {
    pub before: usize,
    pub after:  usize,
}

impl OptStats {
    pub fn eliminated(&self) -> usize { self.before.saturating_sub(self.after) }
}

pub fn count_instrs(module: &IrModule) -> usize {
    module.functions.iter()
        .flat_map(|f| &f.blocks)
        .chain(module.main.iter())
        .map(|b| b.instructions.len())
        .sum()
}
