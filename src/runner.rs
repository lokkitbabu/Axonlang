// src/runner.rs — Backend Compiler Invocation
//
// After codegen produces textual IR, this module hands it to the real
// backend tools:
//
//   LLVM  : llc  → .s (native asm) or --filetype=obj → .o
//           opt  → run LLVM optimisation passes first
//   CUDA  : nvcc → .ptx or .cubin
//   MLIR  : mlir-opt → verify + run MLIR passes
//
// All tools are invoked as subprocess via std::process::Command.
// Missing tools are reported gracefully — callers decide whether to abort.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::errors::DslError;

// ─────────────────────────────────────────────────────────────────────────────
// Tool availability probes
// ─────────────────────────────────────────────────────────────────────────────

pub fn probe_tools() -> ToolReport {
    ToolReport {
        llc:      which("llc"),
        opt:      which("opt"),
        nvcc:     which("nvcc"),
        mlir_opt: which("mlir-opt"),
        clang:    which("clang"),
    }
}

fn which(tool: &str) -> Option<PathBuf> {
    let out = Command::new("which").arg(tool).output().ok()?;
    if out.status.success() {
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !s.is_empty() { return Some(PathBuf::from(s)); }
    }
    None
}

#[derive(Debug)]
pub struct ToolReport {
    pub llc:      Option<PathBuf>,
    pub opt:      Option<PathBuf>,
    pub nvcc:     Option<PathBuf>,
    pub mlir_opt: Option<PathBuf>,
    pub clang:    Option<PathBuf>,
}

impl ToolReport {
    pub fn print_summary(&self) {
        let check = |p: &Option<PathBuf>, name: &str| {
            match p {
                Some(path) => eprintln!("  ✓ {}  ({})", name, path.display()),
                None       => eprintln!("  ✗ {}  (not found in PATH)", name),
            }
        };
        eprintln!("Backend tools:");
        check(&self.llc,      "llc     ");
        check(&self.opt,      "opt     ");
        check(&self.clang,    "clang   ");
        check(&self.nvcc,     "nvcc    ");
        check(&self.mlir_opt, "mlir-opt");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LLVM backend
// ─────────────────────────────────────────────────────────────────────────────

/// Options for the LLVM backend pipeline.
#[derive(Debug, Clone)]
pub struct LlvmOptions {
    /// Optimisation level: 0–3
    pub opt_level: u8,
    /// Target triple (None = host)
    pub target:    Option<String>,
    /// Emit PTX for NVIDIA GPU (requires llc with NVPTX target)
    pub emit_ptx:  bool,
    /// Emit native object file instead of assembly
    pub emit_obj:  bool,
}

impl Default for LlvmOptions {
    fn default() -> Self {
        LlvmOptions { opt_level: 2, target: None, emit_ptx: false, emit_obj: false }
    }
}

pub struct LlvmOutput {
    /// Path to the generated .s / .o / .ptx file
    pub artifact: PathBuf,
    /// Stdout+stderr from llc
    pub log: String,
}

/// Write `llvm_ir` to a temp file, run `opt` then `llc`, return output path.
pub fn run_llvm(
    llvm_ir:  &str,
    out_stem: &Path,
    opts:     &LlvmOptions,
) -> Result<LlvmOutput, DslError> {
    let tools = probe_tools();

    // Write .ll source
    let ll_path = out_stem.with_extension("ll");
    fs::write(&ll_path, llvm_ir)
        .map_err(|e| DslError::Io(e))?;

    let mut log = String::new();
    let mut current_ll = ll_path.clone();

    // ── opt pass ─────────────────────────────────────────────────────────────
    if let Some(opt_bin) = &tools.opt {
        let opt_out = out_stem.with_extension("opt.ll");
        let status = Command::new(opt_bin)
            .args([
                &format!("-O{}", opts.opt_level),
                "-S",
                current_ll.to_str().unwrap(),
                "-o", opt_out.to_str().unwrap(),
            ])
            .output()
            .map_err(|e| DslError::CodegenError(format!("opt failed: {}", e)))?;

        log.push_str(&String::from_utf8_lossy(&status.stderr));
        if status.status.success() {
            current_ll = opt_out;
        }
        // Non-success is non-fatal; we proceed with unoptimised IR
    }

    // ── llc pass ─────────────────────────────────────────────────────────────
    let llc_bin = tools.llc.as_ref()
        .ok_or_else(|| DslError::CodegenError(
            "`llc` not found in PATH. Install LLVM (apt install llvm).".to_string()))?;

    let ext        = if opts.emit_obj { "o" } else if opts.emit_ptx { "ptx" } else { "s" };
    let filetype   = if opts.emit_obj { "obj" } else { "asm" };
    let artifact   = out_stem.with_extension(ext);

    let mut args: Vec<String> = vec![
        format!("-O{}", opts.opt_level),
        format!("--filetype={}", filetype),
        current_ll.to_str().unwrap().to_string(),
        "-o".to_string(), artifact.to_str().unwrap().to_string(),
    ];
    if opts.emit_ptx {
        args.push("--march=nvptx64".to_string());
        args.push("--mcpu=sm_80".to_string());
    } else if let Some(triple) = &opts.target {
        args.push(format!("--mtriple={}", triple));
    }

    let output = Command::new(llc_bin)
        .args(&args)
        .output()
        .map_err(|e| DslError::CodegenError(format!("llc failed to launch: {}", e)))?;

    log.push_str(&String::from_utf8_lossy(&output.stderr));

    if !output.status.success() {
        return Err(DslError::CodegenError(format!(
            "llc exited with {:?}\n{}",
            output.status.code(),
            log
        )));
    }

    Ok(LlvmOutput { artifact, log })
}

// ─────────────────────────────────────────────────────────────────────────────
// CUDA backend
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CudaOptions {
    /// SM architecture: e.g. "sm_80" for A100, "sm_90" for H100
    pub arch:      String,
    /// Emit PTX text instead of cubin
    pub emit_ptx:  bool,
    /// Optimisation level
    pub opt_level: u8,
}

impl Default for CudaOptions {
    fn default() -> Self {
        CudaOptions { arch: "sm_80".to_string(), emit_ptx: true, opt_level: 2 }
    }
}

pub struct CudaOutput {
    pub artifact: PathBuf,
    pub log:      String,
}

/// Write CUDA C to a .cu file and invoke nvcc.
pub fn run_cuda(
    cuda_src: &str,
    out_stem: &Path,
    opts:     &CudaOptions,
) -> Result<CudaOutput, DslError> {
    let tools = probe_tools();
    let nvcc   = tools.nvcc.as_ref()
        .ok_or_else(|| DslError::CodegenError(
            "`nvcc` not found. Install CUDA toolkit.".to_string()))?;

    let cu_path = out_stem.with_extension("cu");
    fs::write(&cu_path, cuda_src).map_err(|e| DslError::Io(e))?;

    let ext = if opts.emit_ptx { "ptx" } else { "cubin" };
    let artifact = out_stem.with_extension(ext);

    let mut args = vec![
        format!("-arch={}", opts.arch),
        format!("-O{}", opts.opt_level),
        cu_path.to_str().unwrap().to_string(),
        "-o".to_string(), artifact.to_str().unwrap().to_string(),
    ];
    if opts.emit_ptx { args.push("--ptx".to_string()); }
    else             { args.push("--cubin".to_string()); }

    let output = Command::new(nvcc)
        .args(&args)
        .output()
        .map_err(|e| DslError::CodegenError(format!("nvcc failed: {}", e)))?;

    let log = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    if !output.status.success() {
        return Err(DslError::CodegenError(format!(
            "nvcc exited {:?}\n{}", output.status.code(), log)));
    }
    Ok(CudaOutput { artifact, log })
}

// ─────────────────────────────────────────────────────────────────────────────
// MLIR backend
// ─────────────────────────────────────────────────────────────────────────────

pub struct MlirOutput {
    pub verified: bool,
    pub log:      String,
}

/// Validate + run canonicalisation passes via mlir-opt.
pub fn run_mlir(mlir_src: &str, out_stem: &Path) -> Result<MlirOutput, DslError> {
    let tools    = probe_tools();
    let mlir_opt = tools.mlir_opt.as_ref()
        .ok_or_else(|| DslError::CodegenError(
            "`mlir-opt` not found. Install LLVM with MLIR.".to_string()))?;

    let mlir_path = out_stem.with_extension("mlir");
    fs::write(&mlir_path, mlir_src).map_err(|e| DslError::Io(e))?;

    let output = Command::new(mlir_opt)
        .args([
            "--canonicalize",
            "--cse",
            mlir_path.to_str().unwrap(),
        ])
        .output()
        .map_err(|e| DslError::CodegenError(format!("mlir-opt failed: {}", e)))?;

    let log = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    Ok(MlirOutput { verified: output.status.success(), log })
}
