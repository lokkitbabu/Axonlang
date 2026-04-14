# AxonLang — Scientific DSL Compiler

A compiled, differentiable scientific DSL targeting HPC and GPU workloads.  
Write physics equations in a clean syntax; get LLVM IR, CUDA C, or MLIR out — with reverse-mode automatic differentiation included.

```
source.dsl  →  PEG Parser  →  SSA IR  →  Opt Passes  →  Autodiff  →  Codegen
                                                    ↓
                                             Interpreter  →  GradCheck
```

---

## Quick Start

```bash
cargo build --release

# Compile to LLVM IR
./target/release/sdsl compile examples/hpc.dsl --target llvm

# Interpret directly (no LLVM needed)
./target/release/sdsl run examples/hpc.dsl --func rms_velocity \
  --args "1.380649e-23,300.0,4.65e-26"
# → 516.9355734487366  (N₂ at 300 K)

# Generate + verify gradients
./target/release/sdsl gradcheck examples/hpc.dsl
# GradCheck: PASSED ✓  (analytic vs finite-difference, max_err < 1e-10)

# Emit gradient functions alongside primal
./target/release/sdsl compile examples/hpc.dsl --target cuda --grad

# Type-annotated SSA IR dump
./target/release/sdsl dump-ir examples/hpc.dsl --opt --types --grad
```

---

## Language

```rust
// Scalar function with type annotations
fn kinetic_energy(m: f64, v: f64) -> f64 {
    return 0.5 * m * v^2.0;
}

// Lorenz attractor step
fn lorenz_step(x: f64, y: f64, z: f64, sigma: f64, rho: f64, beta: f64, dt: f64) -> Tensor {
    let dx = sigma * (y - x) * dt;
    let dy = (x * (rho - z) - y) * dt;
    let dz = (x * y - beta * z) * dt;
    return [x + dx, y + dy, z + dz];
}

// 1D heat diffusion (explicit Euler)
fn heat_diffusion(u: Tensor, alpha: f64, dx: f64, dt: f64) -> Tensor {
    let r = alpha * dt / (dx * dx);
    let u_next: Tensor = [0.0];
    for i in 1..255 {
        let u_next = r * (u[i+1] - 2.0*u[i] + u[i-1]) + u[i];
    }
    return u_next;
}
```

### Supported operations

| Category | Operations |
|---|---|
| Arithmetic | `+` `-` `*` `/` `%` `^` (power) |
| Comparisons | `<` `>` `<=` `>=` `==` `!=` |
| Intrinsics | `sin` `cos` `tan` `exp` `log` `sqrt` `abs` `floor` `ceil` `min` `max` `hypot` |
| Types | `f64` `f32` `Tensor` `Vec<f64>` |
| Control flow | `for i in lo..hi { }` `return` |
| Tensor ops | `[a, b, c]` literals, `arr[i]` indexing |

---

## Compiler Pipeline

### 1 — PEG Parser (`src/parser.rs`, `grammar/dsl.pest`)
Unambiguous PEG grammar via [pest](https://pest.rs).  Named operator rules (`neg_op`, `pow_op`, `type_kw`) ensure all tokens appear as typed pairs in the parse tree.

### 2 — SSA IR (`src/ir.rs`)
18-variant instruction set in Static Single Assignment form.  For-loops lower to phi-node CFGs with proper back-edge semantics.  Pretty-prints as human-readable assembly.

```
fn kinetic_energy(m: f64, v: f64) -> f64 {
  entry:
    %0:f64 = load @m
    %1:f64 = load @v
    %2:f64 = mul  0.5, %0       ; 0.5 * m
    %3:f64 = pow  %1, 2         ; v^2
    %4:f64 = mul  %2, %3        ; 0.5 * m * v²
    ret %4
}
```

### 3 — Optimization Passes (`src/opt.rs`)
Three SSA passes applied before codegen:
- **Constant folding** — `3.0 * 2.0 → 6.0` at compile time
- **Copy propagation** — eliminate redundant load/store pairs
- **Dead code elimination** — remove instructions with unused destinations

### 4 — Type Inference (`src/typeck.rs`)
Forward dataflow analysis that infers `f64`/`i64` for every SSA register.  Produces annotated IR (`%3:f64 = mul ...`) and emits diagnostics for suspicious integer division or integer-base `pow`.

### 5 — Reverse-Mode Autodiff (`src/autodiff.rs`)
Generates gradient functions directly from the SSA tape.  For each primal function `f(x₁..xₙ) → f64`, emits `f__grad(x₁..xₙ) → Tensor` containing `[f(x), ∂f/∂x₁, ..., ∂f/∂xₙ]`.

**Supported VJPs:**

| Primal | Adjoint rule |
|---|---|
| `d = a + b` | `adj_a += s; adj_b += s` |
| `d = a * b` | `adj_a += s·b; adj_b += s·a` |
| `d = a / b` | `adj_a += s/b; adj_b -= s·a/b²` |
| `d = aᵉ` | `adj_a += s·e·aᵉ⁻¹` |
| `d = -a` | `adj_a -= s` |
| `sin(x)` | `adj_x += s·cos(x)` |
| `cos(x)` | `adj_x -= s·sin(x)` |
| `exp(x)` | `adj_x += s·exp(x)` |
| `log(x)` | `adj_x += s/x` |
| `sqrt(x)` | `adj_x += s/(2√x)` |
| `abs(x)` | `adj_x += s·sign(x)` |
| `tan(x)` | `adj_x += s/cos²(x)` |

### 6 — Interpreter (`src/interp.rs`)
Tree-walking interpreter over SSA IR.  Evaluates functions in pure Rust — no LLVM required.  Handles loops via phi-node predecessor tracking, tensor construction/indexing, and all 15+ builtin intrinsics.

### 7 — Gradient Checker (`src/gradcheck.rs`)
Compares analytical gradients (autodiff) against central finite differences `(f(x+h) − f(x−h)) / 2h`.  Used for testing and as a correctness guarantee before GPU dispatch.

### 8 — Code Generators (`src/codegen.rs`)
Three emission targets:

| Target | Output | Use |
|---|---|---|
| `--target llvm` | LLVM IR textual (`.ll`) | Feed to `llc`; compile to native or PTX |
| `--target cuda` | CUDA C (`.cu`) | Feed to `nvcc`; target NVIDIA GPUs |
| `--target mlir` | MLIR `arith`+`func` dialect | Feed to `mlir-opt`; MLIR pipeline |

### 9 — Backend Runner (`src/runner.rs`)
Invokes `llc`, `opt`, `nvcc`, `mlir-opt` as subprocesses.  Probes tool availability with `sdsl tools`.

### 10 — LLM Bridge (`src/llm_bridge.rs`)
Calls the Anthropic Messages API (via `curl`) to:
- Convert natural-language descriptions to DSL (`sdsl intent`)
- Scaffold named equations (`sdsl scaffold`)
- Explain existing programs (`sdsl explain`)
- Suggest GPU parallelism rewrites (`sdsl gpu-opts`)

Requires `ANTHROPIC_API_KEY` in environment.

---

## CLI Reference

```
sdsl compile  <file> [--target llvm|cuda|mlir] [--grad] [--opt N] [--run] [--stats]
sdsl run      <file> [--func NAME] [--args x,y,z]
sdsl gradcheck <file> [--func NAME] [--args x,y,z] [--eps 1e-5] [--atol 1e-4]
sdsl dump-ir  <file> [--opt] [--types] [--grad]
sdsl dump-ast <file>
sdsl grad     <file> [--target llvm|cuda|mlir]
sdsl tools
sdsl intent   "<description>" [--target ...] [--grad]
sdsl scaffold "<equation>" [--constraints "..."]
sdsl explain  <file>
sdsl gpu-opts <file>
```

---

## Examples

```bash
# Run the Lorenz attractor at a specific point
sdsl run examples/hpc.dsl --func lorenz_dx --args "1.0,2.0,10.0"
# → 10

# Verify ∂(0.5·m·v²)/∂m = 0.5v², ∂/∂v = mv
sdsl gradcheck examples/hpc.dsl --func kinetic_energy --args "2.0,3.0"
# GradCheck: PASSED ✓  analytic: [4.5, 6.0]  numeric: [4.5, 6.0]

# Compile heat diffusion to CUDA for A100
sdsl compile examples/hpc.dsl --target cuda --arch sm_80 --grad --run

# LLM: generate a Black-Scholes PDE implementation
sdsl scaffold "Black-Scholes European call option" \
  --constraints "risk-free rate r, volatility sigma, time to expiry T"
```

---

## Project Structure

```
scientific-dsl/
├── grammar/dsl.pest          PEG grammar (operators, types, control flow)
├── src/
│   ├── ast.rs                Typed AST (Expr, Stmt, FnDef, Program)
│   ├── parser.rs             Pest-based parser with rule-dispatch
│   ├── ir.rs                 SSA IR: 18 instruction variants, BasicBlock, CFG lowering
│   ├── opt.rs                Const fold · Copy prop · DCE
│   ├── typeck.rs             Type inference · Annotated IR printer
│   ├── autodiff.rs           Reverse-mode AD · VJPs for all ops + 7 intrinsics
│   ├── interp.rs             Tree-walking SSA interpreter · phi-node CFG execution
│   ├── gradcheck.rs          Numerical gradient verification (central differences)
│   ├── codegen.rs            LLVM IR · CUDA C · MLIR emission
│   ├── runner.rs             llc · nvcc · mlir-opt subprocess invocation
│   ├── llm_bridge.rs         Anthropic API bridge (curl-based, zero HTTP deps)
│   ├── errors.rs             Unified DslError
│   ├── lib.rs                Public surface
│   └── main.rs               sdsl CLI (10 commands)
├── examples/hpc.dsl          Heat eq · Lorenz · Kinetic energy · Gravitational potential
└── tests/integration.rs      42 passing tests
```

---

## Development Status

| Component | Status |
|---|---|
| PEG grammar + parser | ✅ Complete |
| SSA IR lowering | ✅ Complete |
| Opt passes (fold/prop/DCE) | ✅ Complete |
| Type inference | ✅ Complete |
| Reverse-mode autodiff (straight-line) | ✅ Complete |
| Intrinsic VJPs (sin/cos/exp/log/sqrt/abs/tan) | ✅ Complete |
| SSA interpreter + phi-node CFG | ✅ Complete |
| Numerical gradient checker | ✅ Complete |
| LLVM IR / CUDA C / MLIR codegen | ✅ Complete |
| Backend runner (llc/nvcc/mlir-opt) | ✅ Complete |
| LLM bridge (intent → DSL) | ✅ Complete |
| Loop autodiff (through phi nodes) | 🔲 Planned |
| JIT execution (memmap + fn pointer) | 🔲 Planned |
| Cross-block type inference | 🔲 Planned |
| proptest parser fuzzing | 🔲 Planned |

---

## License

See [LICENSE](LICENSE).
