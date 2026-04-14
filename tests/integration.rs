// tests/integration.rs — End-to-end parser + IR tests

use scientific_dsl::{parser, ir, codegen, opt, autodiff, typeck};
use scientific_dsl::codegen::Target;

// ── Parse tests ──────────────────────────────────────────────────────────────

#[test]
fn test_parse_simple_assign() {
    let src = "let x = 3.14;";
    let prog = parser::parse(src).expect("parse failed");
    assert_eq!(prog.items.len(), 1);
}

#[test]
fn test_parse_binop_precedence() {
    // a + b * c  should parse as  a + (b * c)
    let src = "let z = a + b * c;";
    parser::parse(src).expect("parse failed");
}

#[test]
fn test_parse_power() {
    let src = "let e = x^2.0;";
    parser::parse(src).expect("parse failed");
}

#[test]
fn test_parse_function_call() {
    let src = "let y = sin(x);";
    parser::parse(src).expect("parse failed");
}

#[test]
fn test_parse_tensor_literal() {
    let src = "let v = [1.0, 2.0, 3.0];";
    parser::parse(src).expect("parse failed");
}

#[test]
fn test_parse_fn_def() {
    let src = r#"
        fn kinetic_energy(m: f64, v: f64) -> f64 {
            let ke = 0.5 * m * v^2.0;
            return ke;
        }
    "#;
    let prog = parser::parse(src).expect("parse failed");
    assert_eq!(prog.fns().count(), 1);
}

#[test]
fn test_parse_for_loop() {
    let src = r#"
        fn sum(n: f64) -> f64 {
            let acc = 0.0;
            for i in 0..10 {
                let acc = acc + 1.0;
            }
            return acc;
        }
    "#;
    parser::parse(src).expect("parse failed");
}

#[test]
fn test_parse_type_annotations() {
    let src = "let grid: Tensor = [0.0, 0.0, 0.0];";
    parser::parse(src).expect("parse failed");
}

#[test]
fn test_parse_negative_literal() {
    let src = "let t = -9.81;";
    parser::parse(src).expect("parse failed");
}

#[test]
fn test_parse_nested_calls() {
    let src = "let r = sqrt(x^2.0 + y^2.0);";
    parser::parse(src).expect("parse failed");
}

// ── IR lowering tests ─────────────────────────────────────────────────────────

#[test]
fn test_ir_lower_assign() {
    let src = "let x = 1.0 + 2.0;";
    let prog = parser::parse(src).unwrap();
    let module = ir::lower_program(&prog).expect("lower failed");
    // main block should contain at least an Add and a Store
    let instrs: Vec<_> = module.main.iter().flat_map(|b| &b.instructions).collect();
    assert!(instrs.iter().any(|i| matches!(i, ir::Instruction::Add { .. })));
    assert!(instrs.iter().any(|i| matches!(i, ir::Instruction::Store { .. })));
}

#[test]
fn test_ir_lower_fn() {
    let src = r#"
        fn area(r: f64) -> f64 {
            let pi = 3.14159;
            return pi * r^2.0;
        }
    "#;
    let prog   = parser::parse(src).unwrap();
    let module = ir::lower_program(&prog).unwrap();
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].name, "area");
}

#[test]
fn test_ir_print_roundtrip() {
    let src = r#"
        fn f(x: f64) -> f64 {
            return x * x;
        }
    "#;
    let prog   = parser::parse(src).unwrap();
    let module = ir::lower_program(&prog).unwrap();
    let printed = ir::print_module(&module);
    assert!(printed.contains("fn f"));
    assert!(printed.contains("mul"));
}

// ── Codegen tests ─────────────────────────────────────────────────────────────

#[test]
fn test_codegen_llvm() {
    let src = "let y = 2.0 * x + 1.0;";
    let prog   = parser::parse(src).unwrap();
    let module = ir::lower_program(&prog).unwrap();
    let code   = codegen::generate(&module, &Target::LlvmIr).unwrap();
    assert!(code.contains("fmul") || code.contains("fadd"));
}

#[test]
fn test_codegen_cuda() {
    let src = r#"
        fn heat_step(u: Tensor, alpha: f64) -> f64 {
            return alpha * u[1];
        }
    "#;
    let prog   = parser::parse(src).unwrap();
    let module = ir::lower_program(&prog).unwrap();
    let code   = codegen::generate(&module, &Target::CudaC).unwrap();
    assert!(code.contains("__global__"));
    assert!(code.contains("blockIdx"));
}

#[test]
fn test_codegen_mlir() {
    let src = "let z = a + b;";
    let prog   = parser::parse(src).unwrap();
    let module = ir::lower_program(&prog).unwrap();
    let code   = codegen::generate(&module, &Target::MlirDialect).unwrap();
    assert!(code.contains("arith.addf"));
}

// ── Autodiff tests ────────────────────────────────────────────────────────────

use scientific_dsl::ir::{Value, Instruction, BasicBlock, Register};

#[test]
fn test_autodiff_linear() {
    // f(x) = 3.0 * x  →  f'(x) = 3.0
    let src    = "fn f(x: f64) -> f64 { return 3.0 * x; }";
    let prog   = parser::parse(src).unwrap();
    let module = ir::lower_program(&prog).unwrap();
    let func   = &module.functions[0];
    let grad   = autodiff::differentiate(func).expect("autodiff failed");

    assert_eq!(grad.name, "f__grad");
    // The grad function should have only one block
    assert_eq!(grad.blocks.len(), 1);
    // It should return a Tensor (primal + grad)
    assert!(matches!(grad.ret_ty, scientific_dsl::ast::TypeAnnotation::Tensor));
    // Return should be via BuildTensor
    let build_tensors: Vec<_> = grad.blocks[0].instructions.iter()
        .filter(|i| matches!(i, Instruction::BuildTensor { .. }))
        .collect();
    assert!(!build_tensors.is_empty(), "Expected a BuildTensor in grad function");
}

#[test]
fn test_autodiff_quadratic() {
    // f(x) = x^2.0  →  should produce adjoint involving mul and pow
    let src  = "fn sq(x: f64) -> f64 { return x^2.0; }";
    let prog = parser::parse(src).unwrap();
    let module = ir::lower_program(&prog).unwrap();
    let grad = autodiff::differentiate(&module.functions[0]).unwrap();

    // There must be a Pow in the backward sweep (for the e*x^(e-1) term)
    let has_pow = grad.blocks[0].instructions.iter()
        .any(|i| matches!(i, Instruction::Pow { .. }));
    assert!(has_pow, "x^2 backward should produce a Pow instruction");
}

#[test]
fn test_autodiff_add_sub() {
    // f(x, y) = x + y  →  df/dx = 1, df/dy = 1
    // f(x, y) = x - y  →  df/dx = 1, df/dy = -1
    for (src, name) in &[
        ("fn add(x: f64, y: f64) -> f64 { return x + y; }", "add"),
        ("fn sub(x: f64, y: f64) -> f64 { return x - y; }", "sub"),
    ] {
        let prog   = parser::parse(src).unwrap();
        let module = ir::lower_program(&prog).unwrap();
        let grad   = autodiff::differentiate(&module.functions[0]).unwrap();
        assert_eq!(grad.name, format!("{}__grad", name));
        assert!(!grad.blocks[0].instructions.is_empty());
    }
}

#[test]
fn test_autodiff_division() {
    // f(a, b) = a / b  — verify backward emits Div and Mul (for b² term)
    let src    = "fn div(a: f64, b: f64) -> f64 { return a / b; }";
    let prog   = parser::parse(src).unwrap();
    let module = ir::lower_program(&prog).unwrap();
    let grad   = autodiff::differentiate(&module.functions[0]).unwrap();

    let has_div = grad.blocks[0].instructions.iter()
        .any(|i| matches!(i, Instruction::Div { .. }));
    assert!(has_div, "div backward should emit a Div for adj_a=s/b");
}

#[test]
fn test_autodiff_negation() {
    let src    = "fn neg_fn(x: f64) -> f64 { return -x; }";
    let prog   = parser::parse(src).unwrap();
    let module = ir::lower_program(&prog).unwrap();
    let grad   = autodiff::differentiate(&module.functions[0]).unwrap();

    let neg_count = grad.blocks[0].instructions.iter()
        .filter(|i| matches!(i, Instruction::Neg { .. }))
        .count();
    // neg backward needs at least one Neg to negate the seed
    assert!(neg_count >= 1, "negation backward should emit a Neg");
}

#[test]
fn test_autodiff_module() {
    let src = r#"
        fn kinetic(m: f64, v: f64) -> f64 { return 0.5 * m * v^2.0; }
        fn potential(m: f64, g: f64, h: f64) -> f64 { return m * g * h; }
    "#;
    let prog   = parser::parse(src).unwrap();
    let mut module = ir::lower_program(&prog).unwrap();
    autodiff::differentiate_module(&mut module).expect("module autodiff failed");

    // Should have 4 functions: kinetic, potential, kinetic__grad, potential__grad
    assert_eq!(module.functions.len(), 4);
    let names: Vec<_> = module.functions.iter().map(|f| &f.name).collect();
    assert!(names.iter().any(|n| n.as_str() == "kinetic__grad"));
    assert!(names.iter().any(|n| n.as_str() == "potential__grad"));
}

// ── Type inference tests ──────────────────────────────────────────────────────


#[test]
fn test_typeck_constant_types() {
    // Use an expression that creates register-defining instructions
    let src  = "let z = 1.0 + 2.0;";
    let prog = parser::parse(src).unwrap();
    let module = ir::lower_program(&prog).unwrap();
    let mt = typeck::infer_module(&module);
    // 1.0 + 2.0 emits an Add instruction defining a register → env non-empty
    assert!(!mt.main.is_empty(), "Expected typed registers from Add instruction");
}

#[test]
fn test_typeck_arithmetic_propagation() {
    let src  = "fn f(x: f64) -> f64 { return x * x; }";
    let prog = parser::parse(src).unwrap();
    let module = ir::lower_program(&prog).unwrap();
    let mt   = typeck::infer_module(&module);
    let env  = mt.functions.get("f").unwrap();
    // All typed registers should be F64 (x is f64, x*x is f64)
    for ty in env.values() {
        assert_eq!(*ty, typeck::RegTy::F64, "Expected all regs to be F64");
    }
}

#[test]
fn test_typeck_comparison_result() {
    // Comparison produces i64 (boolean)
    let src  = "fn cmp(a: f64, b: f64) -> f64 { return a + b; }";
    let prog = parser::parse(src).unwrap();
    let module = ir::lower_program(&prog).unwrap();

    // Inject a synthetic Lt instruction to test typing
    use scientific_dsl::ir::{Register, Instruction, Value};
    let mut env = typeck::TypeEnv::new();
    env.insert(0, typeck::RegTy::F64);
    env.insert(1, typeck::RegTy::F64);
    let lt_instr = Instruction::Lt {
        dest: Register(2),
        lhs:  Value::Reg(Register(0)),
        rhs:  Value::Reg(Register(1)),
    };
    let mut block = scientific_dsl::ir::BasicBlock::new("test");
    block.push(lt_instr);
    typeck::infer_block(&block, &mut env);
    assert_eq!(env[&2], typeck::RegTy::I64, "Lt result should be I64");
}

#[test]
fn test_typeck_annotated_ir() {
    let src  = "fn area(r: f64) -> f64 { return 3.14 * r^2.0; }";
    let prog = parser::parse(src).unwrap();
    let module = ir::lower_program(&prog).unwrap();
    let annotated = typeck::annotate_ir(&module);
    // Annotated IR should contain ":f64" type annotations on registers
    assert!(annotated.contains(":f64"), "Expected :f64 type annotations");
}

// ── End-to-end pipeline: parse → lower → opt → autodiff → codegen ───────────

#[test]
fn test_full_pipeline_with_grad() {
    let src = r#"
        fn lorenz_dx(x: f64, y: f64, sigma: f64) -> f64 {
            return sigma * (y - x);
        }
    "#;
    let prog   = parser::parse(src).unwrap();
    let mut module = ir::lower_program(&prog).unwrap();
    let module = opt::optimize(module);
    let mut module = module;
    autodiff::differentiate_module(&mut module).unwrap();

    // Should have lorenz_dx AND lorenz_dx__grad
    assert!(module.functions.iter().any(|f| f.name == "lorenz_dx__grad"));

    // Codegen all three targets without panicking
    let llvm = codegen::generate(&module, &codegen::Target::LlvmIr).unwrap();
    let cuda = codegen::generate(&module, &codegen::Target::CudaC).unwrap();
    let mlir = codegen::generate(&module, &codegen::Target::MlirDialect).unwrap();

    assert!(llvm.contains("lorenz_dx__grad"));
    assert!(cuda.contains("lorenz_dx__grad"));
    assert!(mlir.contains("lorenz_dx__grad"));
}
