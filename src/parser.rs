// src/parser.rs — PEG Parser (pest)

use pest::iterators::Pair;
use pest::Parser;
use pest_derive::Parser;

use crate::ast::*;
use crate::errors::DslError;

#[derive(Parser)]
#[grammar = "grammar/dsl.pest"]
pub struct DslParser;

// ─── Entry ───────────────────────────────────────────────────────────────────

pub fn parse(source: &str) -> Result<Program, DslError> {
    let pairs = DslParser::parse(Rule::program, source)
        .map_err(|e| DslError::ParseError(e.to_string()))?;

    let mut program = Program::default();
    for pair in pairs {
        if pair.as_rule() == Rule::program {
            for item in pair.into_inner() {
                match item.as_rule() {
                    Rule::item => {
                        let inner = item.into_inner().next().unwrap();
                        match inner.as_rule() {
                            Rule::fn_def => program.items.push(TopLevel::FnDef(parse_fn_def(inner)?)),
                            Rule::stmt   => program.items.push(TopLevel::Stmt(parse_stmt(inner)?)),
                            r => return Err(DslError::ParseError(format!("Unknown item: {:?}", r))),
                        }
                    }
                    Rule::EOI => {}
                    r => return Err(DslError::ParseError(format!("Unexpected program child: {:?}", r))),
                }
            }
        }
    }
    Ok(program)
}

// ─── Statements ──────────────────────────────────────────────────────────────

fn parse_stmt(pair: Pair<Rule>) -> Result<Stmt, DslError> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::assign      => parse_assign(inner),
        Rule::for_loop    => parse_for_loop(inner),
        Rule::return_stmt => parse_return(inner),
        Rule::expr_stmt   => parse_expr_stmt(inner),
        r => Err(DslError::ParseError(format!("Unknown stmt rule: {:?}", r))),
    }
}

fn parse_assign(pair: Pair<Rule>) -> Result<Stmt, DslError> {
    let mut inner    = pair.into_inner();
    let name         = inner.next().unwrap().as_str().to_string();
    let next         = inner.next().unwrap();
    let (type_ann, value_pair) = if next.as_rule() == Rule::type_ann {
        (Some(parse_type_ann(next)?), inner.next().unwrap())
    } else {
        (None, next)
    };
    Ok(Stmt::Assign { name, type_ann, value: parse_expr(value_pair)? })
}

fn parse_for_loop(pair: Pair<Rule>) -> Result<Stmt, DslError> {
    let mut inner = pair.into_inner();
    let var  = inner.next().unwrap().as_str().to_string();
    let lo   = parse_expr(inner.next().unwrap())?;
    let hi   = parse_expr(inner.next().unwrap())?;
    let body = parse_block(inner.next().unwrap())?;
    Ok(Stmt::ForLoop { var, lo, hi, body })
}

fn parse_return(pair: Pair<Rule>) -> Result<Stmt, DslError> {
    let e = parse_expr(pair.into_inner().next().unwrap())?;
    Ok(Stmt::Return(e))
}

fn parse_expr_stmt(pair: Pair<Rule>) -> Result<Stmt, DslError> {
    let e = parse_expr(pair.into_inner().next().unwrap())?;
    Ok(Stmt::ExprStmt(e))
}

fn parse_block(pair: Pair<Rule>) -> Result<Vec<Stmt>, DslError> {
    pair.into_inner().map(parse_stmt).collect()
}

// ─── Functions ───────────────────────────────────────────────────────────────

fn parse_fn_def(pair: Pair<Rule>) -> Result<FnDef, DslError> {
    let mut inner  = pair.into_inner();
    let name       = inner.next().unwrap().as_str().to_string();
    let mut params = vec![];
    let mut next   = inner.next().unwrap();
    if next.as_rule() == Rule::param_list {
        params = next.into_inner().map(parse_param).collect::<Result<_, _>>()?;
        next   = inner.next().unwrap();
    }
    let ret_ty = parse_type_ann(next)?;
    let body   = parse_block(inner.next().unwrap())?;
    Ok(FnDef { name, params, ret_ty, body })
}

fn parse_param(pair: Pair<Rule>) -> Result<Param, DslError> {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();
    let ty   = parse_type_ann(inner.next().unwrap())?;
    Ok(Param { name, ty })
}

fn parse_type_ann(pair: Pair<Rule>) -> Result<TypeAnnotation, DslError> {
    // type_ann = { vec_type | type_kw | identifier }
    let child = pair.into_inner().next().unwrap();
    match child.as_rule() {
        Rule::vec_type => {
            let inner_pair = child.into_inner().next().unwrap();
            let inner_ty   = parse_type_ann(inner_pair)?;
            Ok(TypeAnnotation::Vec(Box::new(inner_ty)))
        }
        Rule::type_kw => match child.as_str() {
            "f64"    => Ok(TypeAnnotation::F64),
            "f32"    => Ok(TypeAnnotation::F32),
            "Tensor" => Ok(TypeAnnotation::Tensor),
            s        => Ok(TypeAnnotation::Named(s.to_string())),
        },
        Rule::identifier => Ok(TypeAnnotation::Named(child.as_str().to_string())),
        r => Err(DslError::ParseError(format!("Unknown type_ann child: {:?}", r))),
    }
}

// ─── Expressions ─────────────────────────────────────────────────────────────

fn parse_expr(pair: Pair<Rule>) -> Result<Expr, DslError> {
    match pair.as_rule() {
        Rule::expr         => parse_expr(pair.into_inner().next().unwrap()),
        Rule::comparison   => parse_comparison(pair),
        Rule::addition     => parse_addition(pair),
        Rule::multiplication => parse_multiplication(pair),
        Rule::power        => parse_power(pair),
        Rule::unary        => parse_unary(pair),
        Rule::postfix      => parse_postfix(pair),
        Rule::primary      => parse_primary(pair),
        Rule::paren_expr   => parse_expr(pair.into_inner().next().unwrap()),
        Rule::float        => Ok(Expr::Float(pair.as_str().parse()
                                .map_err(|_| DslError::ParseError(format!("Bad float: {}", pair.as_str())))?)),
        Rule::integer      => Ok(Expr::Integer(pair.as_str().parse()
                                .map_err(|_| DslError::ParseError(format!("Bad int: {}", pair.as_str())))?)),
        Rule::identifier   => Ok(Expr::Identifier(pair.as_str().to_string())),
        r => Err(DslError::ParseError(format!("Unexpected expr rule: {:?}", r))),
    }
}

fn parse_comparison(pair: Pair<Rule>) -> Result<Expr, DslError> {
    let mut inner = pair.into_inner();
    let lhs = parse_expr(inner.next().unwrap())?;
    if let Some(op_pair) = inner.next() {
        let op  = parse_cmp_op(&op_pair);
        let rhs = parse_expr(inner.next().unwrap())?;
        Ok(Expr::BinOp { op, lhs: Box::new(lhs), rhs: Box::new(rhs) })
    } else {
        Ok(lhs)
    }
}

fn parse_addition(pair: Pair<Rule>) -> Result<Expr, DslError> {
    let mut inner = pair.into_inner();
    let mut lhs   = parse_expr(inner.next().unwrap())?;
    while let Some(op_pair) = inner.next() {
        let op  = parse_add_op(&op_pair);
        let rhs = parse_expr(inner.next().unwrap())?;
        lhs = Expr::BinOp { op, lhs: Box::new(lhs), rhs: Box::new(rhs) };
    }
    Ok(lhs)
}

fn parse_multiplication(pair: Pair<Rule>) -> Result<Expr, DslError> {
    let mut inner = pair.into_inner();
    let mut lhs   = parse_expr(inner.next().unwrap())?;
    while let Some(op_pair) = inner.next() {
        let op  = parse_mul_op(&op_pair);
        let rhs = parse_expr(inner.next().unwrap())?;
        lhs = Expr::BinOp { op, lhs: Box::new(lhs), rhs: Box::new(rhs) };
    }
    Ok(lhs)
}

fn parse_power(pair: Pair<Rule>) -> Result<Expr, DslError> {
    let mut inner = pair.into_inner();
    let base      = parse_expr(inner.next().unwrap())?;
    if let Some(_pow_op) = inner.next() {         // consume pow_op
        let exp = parse_expr(inner.next().unwrap())?;
        Ok(Expr::Pow { base: Box::new(base), exp: Box::new(exp) })
    } else {
        Ok(base)
    }
}

fn parse_unary(pair: Pair<Rule>) -> Result<Expr, DslError> {
    // unary = { (neg_op ~ unary) | postfix }
    // neg_op is a named rule, so it DOES appear in inner pairs.
    let mut inner = pair.into_inner();
    let first     = inner.next().unwrap();
    match first.as_rule() {
        Rule::neg_op  => {
            let operand = parse_unary(inner.next().unwrap())?;
            Ok(Expr::Neg(Box::new(operand)))
        }
        Rule::postfix => parse_postfix(first),
        r => Err(DslError::ParseError(format!("Unknown unary child: {:?}", r))),
    }
}

fn parse_postfix(pair: Pair<Rule>) -> Result<Expr, DslError> {
    // postfix = { primary ~ (call_args | index_expr)* }
    let mut inner = pair.into_inner();
    let mut base  = parse_primary(inner.next().unwrap())?;

    for suffix in inner {
        match suffix.as_rule() {
            Rule::call_args => {
                let func = match &base {
                    Expr::Identifier(n) => n.clone(),
                    _ => return Err(DslError::ParseError("Call target must be an identifier".into())),
                };
                let args = suffix.into_inner()
                    .map(parse_expr)
                    .collect::<Result<Vec<_>, _>>()?;
                base = Expr::Call { func, args };
            }
            Rule::index_expr => {
                let idx = parse_expr(suffix.into_inner().next().unwrap())?;
                base = Expr::Index { array: Box::new(base), idx: Box::new(idx) };
            }
            r => return Err(DslError::ParseError(format!("Unknown postfix suffix: {:?}", r))),
        }
    }
    Ok(base)
}

fn parse_primary(pair: Pair<Rule>) -> Result<Expr, DslError> {
    // primary = { tensor_literal | float | integer | identifier | paren_expr }
    let child = pair.into_inner().next().unwrap();
    match child.as_rule() {
        Rule::tensor_literal => {
            let elems = child.into_inner()
                .map(parse_expr)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Expr::TensorLit(elems))
        }
        Rule::float      => Ok(Expr::Float(child.as_str().parse()
                                .map_err(|_| DslError::ParseError(format!("Bad float: {}", child.as_str())))?)),
        Rule::integer    => Ok(Expr::Integer(child.as_str().parse()
                                .map_err(|_| DslError::ParseError(format!("Bad int: {}", child.as_str())))?)),
        Rule::identifier => Ok(Expr::Identifier(child.as_str().to_string())),
        Rule::paren_expr => parse_expr(child.into_inner().next().unwrap()),
        r => Err(DslError::ParseError(format!("Unknown primary child: {:?}", r))),
    }
}

// ─── Operator helpers ─────────────────────────────────────────────────────────

fn parse_add_op(p: &Pair<Rule>) -> BinOpKind {
    match p.as_str() { "+" => BinOpKind::Add, _ => BinOpKind::Sub }
}
fn parse_mul_op(p: &Pair<Rule>) -> BinOpKind {
    match p.as_str() { "*" => BinOpKind::Mul, "%" => BinOpKind::Mod, _ => BinOpKind::Div }
}
fn parse_cmp_op(p: &Pair<Rule>) -> BinOpKind {
    match p.as_str() {
        "<"  => BinOpKind::Lt, ">"  => BinOpKind::Gt,
        "<=" => BinOpKind::Le, ">=" => BinOpKind::Ge,
        "==" => BinOpKind::Eq, _    => BinOpKind::Ne,
    }
}
