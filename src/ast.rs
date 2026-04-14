// src/ast.rs — Abstract Syntax Tree
// Typed representation produced by the PEG parser.

use serde::{Deserialize, Serialize};

// ── Type System ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypeAnnotation {
    F64,
    F32,
    Tensor,
    Vec(Box<TypeAnnotation>),
    Named(String),
}

impl std::fmt::Display for TypeAnnotation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TypeAnnotation::F64       => write!(f, "f64"),
            TypeAnnotation::F32       => write!(f, "f32"),
            TypeAnnotation::Tensor    => write!(f, "Tensor"),
            TypeAnnotation::Vec(inner) => write!(f, "Vec<{}>", inner),
            TypeAnnotation::Named(s)  => write!(f, "{}", s),
        }
    }
}

// ── Expressions ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    /// Floating-point literal
    Float(f64),
    /// Integer literal
    Integer(i64),
    /// Variable reference
    Identifier(String),
    /// Binary operation: lhs op rhs
    BinOp {
        op:  BinOpKind,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// Unary negation
    Neg(Box<Expr>),
    /// Exponentiation: base ^ exp
    Pow {
        base: Box<Expr>,
        exp:  Box<Expr>,
    },
    /// Function / intrinsic call: f(args...)
    Call {
        func: String,
        args: Vec<Expr>,
    },
    /// Index: array[idx]
    Index {
        array: Box<Expr>,
        idx:   Box<Expr>,
    },
    /// Tensor literal: [e0, e1, ...]
    TensorLit(Vec<Expr>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BinOpKind {
    Add, Sub, Mul, Div, Mod,
    Lt, Gt, Le, Ge, Eq, Ne,
}

impl std::fmt::Display for BinOpKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            BinOpKind::Add => "+", BinOpKind::Sub => "-",
            BinOpKind::Mul => "*", BinOpKind::Div => "/",
            BinOpKind::Mod => "%", BinOpKind::Lt  => "<",
            BinOpKind::Gt  => ">", BinOpKind::Le  => "<=",
            BinOpKind::Ge  => ">=", BinOpKind::Eq => "==",
            BinOpKind::Ne  => "!=",
        };
        write!(f, "{}", s)
    }
}

// ── Statements ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Stmt {
    /// let x: T = expr;
    Assign {
        name:     String,
        type_ann: Option<TypeAnnotation>,
        value:    Expr,
    },
    /// for i in lo..hi { body }
    ForLoop {
        var:  String,
        lo:   Expr,
        hi:   Expr,
        body: Vec<Stmt>,
    },
    /// return expr;
    Return(Expr),
    /// expr;  (expression as statement — side-effects / calls)
    ExprStmt(Expr),
}

// ── Top-level Items ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    pub ty:   TypeAnnotation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FnDef {
    pub name:   String,
    pub params: Vec<Param>,
    pub ret_ty: TypeAnnotation,
    pub body:   Vec<Stmt>,
}

/// A complete parsed program.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TopLevel {
    FnDef(FnDef),
    Stmt(Stmt),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Program {
    pub items: Vec<TopLevel>,
}

impl Program {
    pub fn fns(&self)   -> impl Iterator<Item = &FnDef> {
        self.items.iter().filter_map(|i| match i {
            TopLevel::FnDef(f) => Some(f),
            _ => None,
        })
    }
    pub fn stmts(&self) -> impl Iterator<Item = &Stmt> {
        self.items.iter().filter_map(|i| match i {
            TopLevel::Stmt(s) => Some(s),
            _ => None,
        })
    }
}
