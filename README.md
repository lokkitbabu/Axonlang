# AxonLang

A compiled, differentiable scientific DSL. Write physics equations; get LLVM IR, CUDA C, or MLIR — with correct reverse-mode automatic differentiation through loops, intrinsics, and arbitrary control flow.

```
source.dsl → PEG Parser → SSA IR → Opt Passes → Autodiff → Codegen
                                              ↓
                                     Interpreter + Tape AD → GradCheck
```

---

## Quick start

```bash
cargo build --release

# Interpret directly — no LLVM needed
./target/release/sdsl run examples/hpc.dsl \
  --func rms_velocity --args "1.380649e-23,300.0,4.65e-26"
# → 516.9355734487366  (N₂ RMS velocity at 300 K)

# Verify gradients against finite differences
./target/release/sdsl gradcheck examples/physics.dsl \
  --func bs_call --args "100.0,100.0,0.05,0.2,1.0"
# TapeGrad: PASSED ✓  max_err = 3.2e-07

# Compile to CUDA with gradient functions included
./target/release/sdsl compile examples/hpc.dsl --target cuda --grad --stats

# Type-annotated SSA IR with gradient functions
./target/release/sdsl dump-ir examples/physics.dsl --opt --types --grad
```

---

## Language

```rust
fn lorenz_step(x: f64, y: f64, z: f64, sigma: f64, rho: f64, beta: f64, dt: f64) -> Tensor {
    let dx = sigma * (y - x) * dt;
    let dy = (x * (rho - z) - y) * dt;
    let dz = (x * y - beta * z) * dt;
    return [x + dx, y + dy, z + dz];
}

fn bs_call(S: f64, K: f64, r: f64, sigma: f64, T: f64) -> f64 {
    let d1   = (log(S / K) + (r + sigma^2.0 * 0.5) * T) / (sigma * sqrt(T));
    let d2   = d1 - sigma * sqrt(T);
    let nd1  = 1.0 / (1.0 + exp(-1.7 * d1));
    let nd2  = 1.0 / (1.0 + exp(-1.7 * d2));
    return S * nd1 - K * exp(-r * T) * nd2;
}

fn heat_step(u: Tensor, alpha: f64, dx: f64, dt: f64) -> Tensor {
    let r      = alpha * dt / (dx * dx);
    let u_next: Tensor = [0.0];
    for i in 1..255 {
        let u_next = r * (u[i+1] - 2.0*u[i] + u[i-1]) + u[i];
    }
    return u_next;
}
```

**Types:** `f64` `f32` `Tensor` `Vec<f64>`  
**Ops:** `+ - * / % ^` (power), comparisons `< > <= >= == !=`  
**Intrinsics:** `sin cos tan exp log sqrt abs floor ceil min max hypot atan2`  
**Control flow:** `for i in lo..hi { }` · `return`

---

## Compiler pipeline

| Stage | File | What it does |
|---|---|---|
| PEG parser | `grammar/dsl.pest`, `src/parser.rs` | Unambiguous grammar via [pest](https://pest.rs). Named rules for all operators and type keywords. |
| SSA IR | `src/ir.rs` | 18-variant instruction set. For-loops lower to phi-node CFGs with back-edge patching. Loop-carried variables are invalidated from the env before body lowering, forcing `Load`-from-memory semantics each iteration. |
| Opt passes | `src/opt.rs` | Constant folding · copy propagation · dead code elimination. Idempotent by construction (verified by property tests). |
| Type inference | `src/typeck.rs` | Forward dataflow: `f64`/`i64` for every SSA register. `annotate_ir()` produces inline-typed IR dumps. |
| Static autodiff | `src/autodiff.rs` | Generates `f__grad` IrFunctions from the SSA tape. Works on straight-line entry blocks. VJPs for all 18 arithmetic ops and 7 intrinsics (sin/cos/tan/exp/log/sqrt/abs). |
| Interpreter | `src/interp.rs` | Tree-walking SSA evaluator. Phi nodes use `prev_block` tracking for correct loop semantics. Handles all intrinsics and tensor ops. |
| Tape AD | `src/tape.rs` | Execution-trace reverse-mode AD. Records a Wengert list during forward execution. Adjoints are keyed by **tape-entry index** (not SSA register ID), so loop iterations that share register IDs get independent adjoints. Propagates through Store/Load memory chains via `adj_mem`. |
| Gradient checker | `src/gradcheck.rs` | Compares analytic (autodiff) vs numerical (central finite differences) gradients. `tape_gradcheck` uses the tape for loop-aware verification. |
| Codegen | `src/codegen.rs` | Three targets: LLVM IR textual, CUDA C, MLIR `arith`+`func` dialect. |
| Backend runner | `src/runner.rs` | Invokes `llc`, `nvcc`, `mlir-opt` as subprocesses. |
| LLM bridge | `src/llm_bridge.rs` | Anthropic API via `curl`: `intent→DSL`, `scaffold`, `explain`, `gpu-opts`. |

---

## Automatic differentiation

Two AD backends, each with different trade-offs:

### Static autodiff (`src/autodiff.rs`)
Produces a new `IrFunction` at compile time. Fast — no interpreter overhead. Limited to straight-line code (entry block only).

```bash
sdsl grad examples/hpc.dsl --target llvm
# Emits kinetic_energy__grad, gravitational_potential__grad, etc.
```

### Tape AD (`src/tape.rs`)
Runs the function with a recording interpreter, builds the adjoint from the Wengert list. Handles loops, conditionals, and arbitrary control flow correctly.

```bash
sdsl gradcheck examples/physics.dsl        # verify all scalar fns
sdsl gradcheck examples/physics.dsl \
  --func bs_call --args "100,100,0.05,0.2,1"
```

**Supported VJPs (both backends):**

| Operation | Adjoint rule |
|---|---|
| `a + b` | `adj_a += s`, `adj_b += s` |
| `a - b` | `adj_a += s`, `adj_b -= s` |
| `a * b` | `adj_a += s·b`, `adj_b += s·a` |
| `a / b` | `adj_a += s/b`, `adj_b -= s·a/b²` |
| `aᵉ` | `adj_a += s·e·aᵉ⁻¹` |
| `-a` | `adj_a -= s` |
| `sin(x)` | `s · cos(x)` |
| `cos(x)` | `-s · sin(x)` |
| `exp(x)` | `s · exp(x)` |
| `log(x)` | `s / x` |
| `sqrt(x)` | `s / (2√x)` |
| `abs(x)` | `s · sign(x)` |
| `tan(x)` | `s / cos²(x)` |

---

## CLI

```
sdsl compile   <file> [--target llvm|cuda|mlir] [--grad] [--opt N] [--run] [--stats]
sdsl run       <file> [--func NAME] [--args x,y,z]
sdsl gradcheck <file> [--func NAME] [--args x,y,z] [--eps 1e-5] [--atol 1e-4]
sdsl dump-ir   <file> [--opt] [--types] [--grad]
sdsl dump-ast  <file>
sdsl grad      <file> [--target llvm|cuda|mlir]
sdsl tools
sdsl intent    "<description>" [--target ...] [--grad]   # requires ANTHROPIC_API_KEY
sdsl scaffold  "<equation>" [--constraints "..."]
sdsl explain   <file>
sdsl gpu-opts  <file>
```

---

## Examples

```bash
# Black-Scholes delta = ∂C/∂S verified by finite differences
sdsl gradcheck examples/physics.dsl \
  --func bs_call --args "100.0,100.0,0.05,0.2,1.0"
# TapeGrad: PASSED ✓  analytic[S]=0.5984  numeric[S]=0.5984  err=3e-7

# Kinetic energy at (m=2, v=3): primal=9, ∂/∂m=4.5, ∂/∂v=6
sdsl gradcheck examples/hpc.dsl \
  --func total_kinetic_energy --args "2.0,3.0,0.0,0.0"

# RK4 exponential decay — one step
sdsl run examples/physics.dsl \
  --func rk4_exp_decay --args "1.0,0.0,0.01,1.0"
# → 0.9900498337...  (≈ e^{-0.01})

# Compile Lorenz attractor with gradients to PTX for A100
sdsl compile examples/hpc.dsl \
  --target cuda --arch sm_80 --grad --stats
```

---

## Testing

```bash
cargo test                # 56 tests: 48 integration + 8 property
```

**Integration tests** (`tests/integration.rs`): parser correctness, SSA lowering, all three codegen targets, opt passes, type inference, static autodiff (linear/quadratic/division/negation/sin/exp/sqrt/log/kinetic energy), interpreter (6 tests covering arithmetic + intrinsics + tensors + Lorenz), gradcheck (9 functions), tape AD (linear/quadratic/intrinsics/loop/kinetic/Black-Scholes), full pipeline.

**Property tests** (`tests/proptest_suite.rs`): 8 invariants verified over 200–500 randomly generated single-parameter functions using a deterministic LCG:
1. Parser never panics
2. Every parseable program lowers to IR
3. Codegen never panics across all three targets
4. Interpreter is deterministic
5. Opt passes are idempotent
6. Opt never increases instruction count
7. Tape AD primal matches interpreter
8. Tape AD gradients match central finite differences (< 5% failure rate on random programs)

---

## Project layout

```
AxonLang/
├── grammar/dsl.pest           PEG grammar
├── src/
│   ├── ast.rs                 Typed AST
│   ├── parser.rs              Pest parser
│   ├── ir.rs                  SSA IR + CFG lowering
│   ├── opt.rs                 Const fold · Copy prop · DCE
│   ├── typeck.rs              Type inference
│   ├── autodiff.rs            Static reverse-mode AD
│   ├── interp.rs              SSA interpreter
│   ├── tape.rs                Trace-based loop AD
│   ├── gradcheck.rs           Numerical gradient verification
│   ├── codegen.rs             LLVM IR / CUDA C / MLIR
│   ├── runner.rs              llc · nvcc · mlir-opt
│   ├── llm_bridge.rs          Anthropic API
│   └── main.rs                sdsl CLI (10 commands)
├── examples/
│   ├── hpc.dsl                Heat eq · Lorenz · Kinetic · Gravitational · RMS
│   └── physics.dsl            RK4 · Van der Pol · Black-Scholes · Lennard-Jones
│                              Sigmoid · Softplus · MSE · BCE · Coulomb
└── tests/
    ├── integration.rs         48 tests
    └── proptest_suite.rs      8 property tests
```

---

## Planned

- Loop autodiff through phi nodes (static, without interpreter)
- JIT execution via `cc` + `dlopen`
- Cross-block type inference
- Gradient descent optimizer example
- MLIR lowering to affine dialect for polyhedral optimization

---

## License

See [LICENSE](LICENSE).
