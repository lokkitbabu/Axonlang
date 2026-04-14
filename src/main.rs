mod ast;
mod parser;
mod ir;
mod opt;
mod autodiff;
mod typeck;
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
    about = "Scientific DSL compiler  •  parse → SSA IR → opt → autodiff → codegen → HPC",
    version = "0.3.0",
    after_help = "\
Examples:
  sdsl compile heat.dsl --target cuda --grad --run
  sdsl dump-ir heat.dsl --opt --types
  sdsl intent \"3D Navier-Stokes momentum\" --target llvm --grad
  sdsl grad    heat.dsl --target llvm"
)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Parse → IR → opt → (autodiff) → codegen → (backend tool)
    Compile {
        file:   String,
        #[arg(long, default_value = "llvm")] target: String,
        #[arg(long, default_value = "2")]    opt:    u8,
        #[arg(long)] run:    bool,
        #[arg(long)] output: Option<String>,
        #[arg(long)] ptx:    bool,
        #[arg(long, default_value = "sm_80")] arch: String,
        #[arg(long)] stats:  bool,
        /// Also generate gradient functions for all scalar-returning fns
        #[arg(long)] grad:   bool,
    },
    /// Show (optionally optimised, optionally type-annotated) SSA IR
    DumpIr {
        file: String,
        #[arg(long)] opt:   bool,
        /// Annotate registers with inferred types
        #[arg(long)] types: bool,
        /// Append gradient functions to the IR dump
        #[arg(long)] grad:  bool,
    },
    /// Show AST as JSON
    DumpAst { file: String },
    /// Probe backend tools on PATH
    Tools,
    /// Compile gradient functions for all scalar fns and emit as chosen target
    Grad {
        file:   String,
        #[arg(long, default_value = "llvm")] target: String,
    },
    /// LLM: natural-language → DSL → compile
    Intent {
        description: String,
        #[arg(long, default_value = "llvm")] target: String,
        #[arg(long)] grad: bool,
    },
    /// LLM: scaffold DSL for a named equation
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
            let program = parser::parse(&source).with_context(|| "Parse error")?;
            let raw     = ir::lower_program(&program).with_context(|| "IR lowering")?;
            let before  = count_instrs(&raw);
            let mut module = optimize(raw);

            if grad {
                autodiff::differentiate_module(&mut module)
                    .with_context(|| "Autodiff pass failed")?;
                let added = module.functions.iter().filter(|f| f.name.ends_with("__grad")).count();
                eprintln!("∂ added {} gradient function(s)", added);
            }

            if stats {
                eprintln!("Opt: {} → {} instrs ({} eliminated)",
                    before, count_instrs(&module),
                    before.saturating_sub(count_instrs(&module)));
            }

            let tgt: Target = target.parse().map_err(|e: errors::DslError| anyhow::anyhow!("{}", e))?;
            let code = codegen::generate(&module, &tgt).with_context(|| "Codegen")?;

            let out_stem: PathBuf = output.as_ref()
                .map(|p| PathBuf::from(p).with_extension(""))
                .unwrap_or_else(|| PathBuf::from(&file).with_extension(""));

            let ext = match tgt { Target::CudaC => "cu", Target::MlirDialect => "mlir", _ => "ll" };
            let code_path = out_stem.with_extension(ext);
            fs::write(&code_path, &code)?;
            eprintln!("→ {}", code_path.display());

            if run {
                match tgt {
                    Target::LlvmIr => {
                        let opts = LlvmOptions { opt_level, emit_ptx: ptx, emit_obj: false, target: None };
                        match runner::run_llvm(&code, &out_stem, &opts) {
                            Ok(o)  => eprintln!("✓ llc → {}", o.artifact.display()),
                            Err(e) => eprintln!("✗ {}", e),
                        }
                    }
                    Target::CudaC => {
                        let opts = CudaOptions { arch, emit_ptx: true, opt_level };
                        match runner::run_cuda(&code, &out_stem, &opts) {
                            Ok(o)  => eprintln!("✓ nvcc → {}", o.artifact.display()),
                            Err(e) => eprintln!("✗ {}", e),
                        }
                    }
                    Target::MlirDialect => {
                        match runner::run_mlir(&code, &out_stem) {
                            Ok(o)  => eprintln!("mlir-opt: {}", if o.verified { "✓" } else { "✗" }),
                            Err(e) => eprintln!("✗ {}", e),
                        }
                    }
                }
            } else if output.is_none() {
                println!("{}", code);
            }
        }

        Cmd::DumpIr { file, opt: do_opt, types, grad } => {
            let source  = fs::read_to_string(&file)?;
            let program = parser::parse(&source)?;
            let raw     = ir::lower_program(&program)?;
            let mut module = if do_opt { optimize(raw) } else { raw };
            if grad {
                autodiff::differentiate_module(&mut module)?;
            }
            if types {
                print!("{}", typeck::annotate_ir(&module));
            } else {
                print!("{}", ir::print_module(&module));
            }
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
            let raw     = ir::lower_program(&program)?;
            let mut module = optimize(raw);
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

        Cmd::Explain { file } => {
            println!("{}", llm_bridge::explain_dsl(&fs::read_to_string(&file)?)
                .map_err(|e| anyhow::anyhow!("{}", e))?);
        }

        Cmd::GpuOpts { file } => {
            println!("{}", llm_bridge::suggest_gpu_opts(&fs::read_to_string(&file)?)
                .map_err(|e| anyhow::anyhow!("{}", e))?);
        }
    }
    Ok(())
}
