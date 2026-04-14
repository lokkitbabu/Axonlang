// tests/proptest_suite.rs — Property-Based Fuzzing (manual LCG, no proptest dep)
use scientific_dsl::{parser, ir, opt, codegen, tape, interp};
use scientific_dsl::codegen::Target;
use scientific_dsl::opt::{optimize, count_instrs};
use scientific_dsl::interp::Interpreter;

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self { Lcg(seed) }
    fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn f64_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (self.u64() as f64 / u64::MAX as f64) * (hi - lo)
    }
    fn pick<T: Copy>(&mut self, xs: &[T]) -> T { xs[self.u64() as usize % xs.len()] }
}

const OPS: &[&str] = &["+", "-", "*"];
const LITS: &[&str] = &["0.5", "1.0", "2.0", "3.0"];

fn gen_expr(rng: &mut Lcg, depth: u32) -> String {
    if depth == 0 || rng.u64() % 3 == 0 {
        return if rng.u64() % 2 == 0 { rng.pick(LITS).to_string() } else { "x".to_string() };
    }
    format!("({} {} {})", gen_expr(rng, depth-1), rng.pick(OPS), gen_expr(rng, depth-1))
}

fn programs(n: usize) -> Vec<String> {
    let mut rng = Lcg::new(0xDEADBEEFCAFE1234);
    (0..n).map(|_| format!("fn f(x: f64) -> f64 {{ return {}; }}", gen_expr(&mut rng, 3))).collect()
}

#[test] fn prop_parser_no_panic() {
    for s in programs(500) { let _ = parser::parse(&s); }
}

#[test] fn prop_parse_implies_lower() {
    let mut parsed = 0; let mut lowered = 0;
    for s in programs(300) {
        if let Ok(p) = parser::parse(&s) { parsed += 1;
            assert!(ir::lower_program(&p).is_ok()); lowered += 1; }
    }
    assert!(parsed >= 200); assert_eq!(parsed, lowered);
}

#[test] fn prop_codegen_all_targets() {
    for s in programs(200) {
        if let Ok(p) = parser::parse(&s) {
            if let Ok(m) = ir::lower_program(&p) {
                let m = optimize(m);
                for t in &[Target::LlvmIr, Target::CudaC, Target::MlirDialect] {
                    assert!(codegen::generate(&m, t).is_ok(), "codegen {:?} failed: {}", t, s);
                }
            }
        }
    }
}

#[test] fn prop_interpreter_deterministic() {
    let mut rng = Lcg::new(0x12345678ABCDEF00);
    for s in programs(200) {
        let x = rng.f64_range(0.5, 10.0);
        if let Ok(p) = parser::parse(&s) {
            if let Ok(m) = ir::lower_program(&p) {
                let m = optimize(m);
                if let Some(f) = m.functions.first() {
                    let r1 = Interpreter::new().call(f, &[x]);
                    let r2 = Interpreter::new().call(f, &[x]);
                    match (r1, r2) {
                        (Ok(a), Ok(b)) => assert_eq!(a.as_f64(), b.as_f64()),
                        (Err(_), Err(_)) => {}
                        _ => panic!("divergent Ok/Err for {}", s),
                    }
                }
            }
        }
    }
}

#[test] fn prop_opt_idempotent() {
    for s in programs(200) {
        if let Ok(p) = parser::parse(&s) {
            if let Ok(m) = ir::lower_program(&p) {
                let once  = count_instrs(&optimize(m.clone()));
                let twice = count_instrs(&optimize(optimize(m)));
                assert_eq!(once, twice, "opt not idempotent: {}", s);
            }
        }
    }
}

#[test] fn prop_opt_never_increases() {
    for s in programs(300) {
        if let Ok(p) = parser::parse(&s) {
            if let Ok(m) = ir::lower_program(&p) {
                let before = count_instrs(&m);
                let after  = count_instrs(&optimize(m));
                assert!(after <= before, "opt increased {} → {}: {}", before, after, s);
            }
        }
    }
}

#[test] fn prop_tape_primal_matches_interp() {
    let mut rng = Lcg::new(0xC0DEBABE11112222);
    for s in programs(200) {
        let x = rng.f64_range(0.5, 5.0);
        if let Ok(p) = parser::parse(&s) {
            if let Ok(m) = ir::lower_program(&p) {
                let m = optimize(m);
                if let Some(f) = m.functions.first() {
                    if let (Ok((primal, _)), Ok(iv)) = (tape::tape_grad(f, &[x]), Interpreter::new().call(f, &[x])) {
                        let iv = iv.as_f64();
                        if primal.is_finite() && iv.is_finite() {
                            assert!((primal - iv).abs() < 1e-9,
                                "tape primal {} ≠ interp {} for f({}): {}", primal, iv, x, s);
                        }
                    }
                }
            }
        }
    }
}

#[test] fn prop_tape_grad_vs_fd() {
    let mut rng = Lcg::new(0xFEEDFACE01020304);
    let mut ok = 0; let mut total = 0;
    for s in programs(300) {
        let x = rng.f64_range(0.5, 5.0);
        if let Ok(p) = parser::parse(&s) {
            if let Ok(m) = ir::lower_program(&p) {
                let m = optimize(m);
                if let Some(f) = m.functions.first() {
                    let h = 1e-5_f64;
                    if let (Ok((_, g)), Ok((fp, _)), Ok((fm, _))) = (
                        tape::tape_grad(f, &[x]),
                        tape::tape_grad(f, &[x + h]),
                        tape::tape_grad(f, &[x - h]),
                    ) {
                        let analytic = g[0];
                        let numeric  = (fp - fm) / (2.0 * h);
                        if analytic.is_finite() && numeric.is_finite() {
                            total += 1;
                            let err = (analytic - numeric).abs() / numeric.abs().max(1.0);
                            if err < 0.1 { ok += 1; }
                        }
                    }
                }
            }
        }
    }
    let fail_rate = 1.0 - ok as f64 / total.max(1) as f64;
    assert!(fail_rate < 0.05, "FD agreement {}/{} ({:.1}% fail)", ok, total, fail_rate*100.0);
}
