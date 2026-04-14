mod ast;
mod parser;
mod ir;
mod opt;
mod autodiff;
mod typeck;
mod interp;
mod gradcheck;
mod codegen;
mod runner;
mod llm_bridge;
mod errors;

use std::fs;
use std::path::PathBuf;
use clap::{Parser, Subcommand};
use anyhow::Context;

use codegen::Target;
use opt::{count_instrs, optimize};
use runner::{LlvmOptions, CudaOptions};

#[derive(Parser, Debug)]
#[command(
    name  = "sdsl",
    about = "Scientific DSL  •  parse → SSA IR → opt → autodiff → interp → codegen → HPC",
    version = "0.4.0",
    after_help = "\
Examples:
  sdsl compile  lorenz.dsl --target cuda --grad --run
  sdsl run      lorenz.dsl --fn lorenz_dx --args 1.0,2.0,10.0
  sdsl gradcheck lorenz.dsl --args 1.0,2.0,10.0
  sdsl dump-ir  lorenz.dsl --opt --types --grad
  sdsl grad     lorenz.dsl --target llvm"
)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Parse → IR → opt → (autodiff) → codegen → (backend)
    Compile {
        file:   String,
        #[arg(long, default_value = "llvm")] target: String,
        #[arg(long, default_value = "2")]    opt:    u8,
        #[arg(long)] run:    bool,
        #[arg(long)] output: Option<String>,
        #[arg(long)] ptx:    bool,
        #[arg(long, default_value = "sm_80")] arch: String,
        #[arg(long)] stats:  bool,
        #[arg(long)] grad:   bool,
    },
    /// Interpret and run a DSL function directly (no LLVM needed)
    Run {
        file: String,
        /// Function to call (default: first function in file)
        #[arg(long)] func: Option<String>,
        /// Comma-separated f64 arguments e.g. "1.0,2.0,3.0"
        #[arg(long, default_value = "")] args: String,
    },
    /// Verify autodiff gradients against finite differences
    Gradcheck {
        file: String,
        /// Function to check (default: all scalar-returning functions)
        #[arg(long)] func: Option<String>,
        /// Comma-separated input values
        #[arg(long, default_value = "")] args: String,
        /// Finite difference step size
        #[arg(long, default_value = "1e-5")] eps: f64,
        /// Absolute tolerance
        #[arg(long, default_value = "1e-4")] atol: f64,
    },
    /// Show (optionally optimised, type-annotated, with grad) SSA IR
    DumpIr {
        file: String,
        #[arg(long)] opt:   bool,
        #[arg(long)] types: bool,
        #[arg(long)] grad:  bool,
    },
    /// Show AST as JSON
    DumpAst { file: String },
    /// Probe backend tools on PATH
    Tools,
    /// Emit gradient functions for all scalar fns as chosen target
    Grad {
        file: String,
        #[arg(long, default_value = "llvm")] target: String,
    },
    /// LLM: natural-language description → DSL → compile
    Intent {
        description: String,
        #[arg(long, default_value = "llvm")] target: String,
        #[arg(long)] grad: bool,
    },
    /// LLM: scaffold a named equation
    Scaffold {
        equation: String,
        #[arg(long, default_value = "")] constraints: String,
    },
    /// LLM: explain a .dsl file
    Explain { file: String },
    /// LLM: suggest GPU parallelism rewrites
    GpuOpts { file: String },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {

        Cmd::Compile { file, target, opt: opt_level, run, output, ptx, arch, stats, grad } => {
            let source  = fs::read_to_string(&file).with_context(|| format!("Cannot read '{}'", file))?;
            let program = parser::parse(&source)?;
            let raw     = ir::lower_program(&program)?;
            let before  = count_instrs(&raw);
            let mut module = optimize(raw);

            if grad {
                autodiff::differentiate_module(&mut module)?;
                let n = module.functions.iter().filter(|f| f.name.ends_with("__grad")).count();
                eprintln!("∂ {} gradient function(s) added", n);
            }
            if stats {
                eprintln!("Opt: {} → {} instrs ({} eliminated)",
                    before, count_instrs(&module), before.saturating_sub(count_instrs(&module)));
            }

            let tgt: Target = target.parse().map_err(|e: errors::DslError| anyhow::anyhow!("{}", e))?;
            let code = codegen::generate(&module, &tgt)?;
            let stem: PathBuf = output.as_ref()
                .map(|p| PathBuf::from(p).with_extension(""))
                .unwrap_or_else(|| PathBuf::from(&file).with_extension(""));
            let ext = match tgt { Target::CudaC => "cu", Target::MlirDialect => "mlir", _ => "ll" };
            fs::write(stem.with_extension(ext), &code)?;
            eprintln!("→ {}", stem.with_extension(ext).display());

            if run {
                match tgt {
                    Target::LlvmIr => {
                        match runner::run_llvm(&code, &stem, &LlvmOptions { opt_level, emit_ptx: ptx, emit_obj: false, target: None }) {
                            Ok(o) => eprintln!("✓ llc → {}", o.artifact.display()),
                            Err(e) => eprintln!("✗ {}", e),
                        }
                    }
                    Target::CudaC => {
                        match runner::run_cuda(&code, &stem, &CudaOptions { arch, emit_ptx: true, opt_level }) {
                            Ok(o) => eprintln!("✓ nvcc → {}", o.artifact.display()),
                            Err(e) => eprintln!("✗ {}", e),
                        }
                    }
                    Target::MlirDialect => {
                        match runner::run_mlir(&code, &stem) {
                            Ok(o) => eprintln!("mlir-opt: {}", if o.verified { "✓" } else { "✗" }),
                            Err(e) => eprintln!("✗ {}", e),
                        }
                    }
                }
            } else if output.is_none() {
                println!("{}", code);
            }
        }

        Cmd::Run { file, func, args } => {
            let source  = fs::read_to_string(&file)?;
            let program = parser::parse(&source)?;
            let module  = optimize(ir::lower_program(&program)?);

            let inputs: Vec<f64> = if args.is_empty() {
                vec![]
            } else {
                args.split(',').map(|s| s.trim().parse::<f64>()
                    .map_err(|_| anyhow::anyhow!("Bad arg: '{}'", s)))
                    .collect::<anyhow::Result<Vec<_>>>()?
            };

            // Pick target function
            let target_fn = match &func {
                Some(name) => module.functions.iter()
                    .find(|f| &f.name == name)
                    .ok_or_else(|| anyhow::anyhow!("Function '{}' not found", name))?,
                None => module.functions.first()
                    .ok_or_else(|| anyhow::anyhow!("No functions in file"))?,
            };

            let mut interp = interp::Interpreter::new();
            let result = interp.call(target_fn, &inputs)
                .map_err(|e| anyhow::anyhow!("{}", e))?;

            match result {
                interp::Val::Scalar(v) => println!("{}", v),
                interp::Val::Tensor(v) => {
                    let s: Vec<_> = v.iter().map(|x| format!("{:.8}", x)).collect();
                    println!("[{}]", s.join(", "));
                }
            }
        }

        Cmd::Gradcheck { file, func, args, eps, atol } => {
            let source  = fs::read_to_string(&file)?;
            let program = parser::parse(&source)?;
            let module  = optimize(ir::lower_program(&program)?);
            let opts    = gradcheck::GradCheckOpts { eps, atol, rtol: atol * 10.0 };

            let inputs: Vec<f64> = if args.is_empty() {
                vec![]
            } else {
                args.split(',').map(|s| s.trim().parse::<f64>()
                    .map_err(|_| anyhow::anyhow!("Bad arg: '{}'", s)))
                    .collect::<anyhow::Result<Vec<_>>>()?
            };

            use ast::TypeAnnotation;

            match &func {
                Some(name) => {
                    let f = module.functions.iter()
                        .find(|f| &f.name == name)
                        .ok_or_else(|| anyhow::anyhow!("Function '{}' not found", name))?;
                    let result = gradcheck::gradcheck(f, &inputs, &opts)
                        .map_err(|e| anyhow::anyhow!("{}", e))?;
                    println!("{}", result.report());
                    if !result.passed { std::process::exit(1); }
                }
                None => {
                    // Check all scalar-returning fns, use provided args or default to 1.0s
                    let results = gradcheck::gradcheck_module(
                        &module,
                        |f| {
                            let n = f.params.len();
                            if inputs.is_empty() {
                                vec![1.0; n]
                            } else if inputs.len() >= n {
                                inputs[..n].to_vec()
                            } else {
                                // pad with 1.0 if fewer args than params
                                let mut v = inputs.clone();
                                v.resize(n, 1.0);
                                v
                            }
                        },
                        &opts,
                    );
                    let mut any_failed = false;
                    for r in &results {
                        match &r.result {
                            Ok(gc) => {
                                println!("{}  {}", r.func_name, if gc.passed { "✓" } else { "✗" });
                                println!("{}", gc.report());
                                if !gc.passed { any_failed = true; }
                            }
                            Err(e) => {
                                eprintln!("{}: ERROR — {}", r.func_name, e);
                                any_failed = true;
                            }
                        }
                    }
                    if any_failed { std::process::exit(1); }
                }
            }
        }

        Cmd::DumpIr { file, opt: do_opt, types, grad } => {
            let source  = fs::read_to_string(&file)?;
            let program = parser::parse(&source)?;
            let raw     = ir::lower_program(&program)?;
            let mut module = if do_opt { optimize(raw) } else { raw };
            if grad { autodiff::differentiate_module(&mut module)?; }
            if types { print!("{}", typeck::annotate_ir(&module)); }
            else     { print!("{}", ir::print_module(&module)); }
        }

        Cmd::DumpAst { file } => {
            let source  = fs::read_to_string(&file)?;
            let program = parser::parse(&source)?;
            println!("{}", serde_json::to_string_pretty(&program)?);
        }

        Cmd::Tools => runner::probe_tools().print_summary(),

        Cmd::Grad { file, target } => {
            let source  = fs::read_to_string(&file)?;
            let program = parser::parse(&source)?;
            let mut module = optimize(ir::lower_program(&program)?);
            autodiff::differentiate_module(&mut module)?;
            let tgt: Target = target.parse().map_err(|e: errors::DslError| anyhow::anyhow!("{}", e))?;
            println!("{}", codegen::generate(&module, &tgt)?);
        }

        Cmd::Intent { description, target, grad } => {
            eprintln!("→ Calling Anthropic API…");
            let (dsl, program) = llm_bridge::intent_to_dsl(&description)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            eprintln!("✓ Generated DSL:\n{}\n", dsl);
            let raw = ir::lower_program(&program)?;
            let mut module = optimize(raw);
            if grad { autodiff::differentiate_module(&mut module)?; }
            let tgt: Target = target.parse().map_err(|e: errors::DslError| anyhow::anyhow!("{}", e))?;
            println!("{}", codegen::generate(&module, &tgt)?);
        }

        Cmd::Scaffold { equation, constraints } => {
            eprintln!("→ Scaffolding '{}'…", equation);
            println!("{}", llm_bridge::scaffold_equation(&equation, &constraints)
                .map_err(|e| anyhow::anyhow!("{}", e))?);
        }

        Cmd::Explain { file } =>
            println!("{}", llm_bridge::explain_dsl(&fs::read_to_string(&file)?)
                .map_err(|e| anyhow::anyhow!("{}", e))?),

        Cmd::GpuOpts { file } =>
            println!("{}", llm_bridge::suggest_gpu_opts(&fs::read_to_string(&file)?)
                .map_err(|e| anyhow::anyhow!("{}", e))?),
    }
    Ok(())
}
