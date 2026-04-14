// src/llm_bridge.rs — LLM-Guided Generation Layer
//
// Uses curl(1) via std::process::Command to call the Anthropic Messages API.
// This avoids any Rust HTTP library dependency, keeping the build minimal.
// Set ANTHROPIC_API_KEY in the environment before using LLM features.

use std::env;
use std::process::Command;

use serde_json::{json, Value};

use crate::ast::Program;
use crate::errors::DslError;
use crate::parser;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const MODEL:   &str = "claude-sonnet-4-20250514";

const DSL_GRAMMAR: &str = r#"
The DSL syntax is:
  let <name> [: <type>] = <expr> ;
  for <var> in <lo>..<hi> { <stmts> }
  fn <name>(<param>: <type>, ...) -> <type> { <stmts> }
  return <expr> ;

Types: f64, f32, Tensor, Vec<f64>
Ops:   + - * / % ^ (power)
Calls: f(a, b, ...)   Index: arr[i]   Tensor literal: [e0, e1, ...]
Scientific intrinsics: sin cos tan exp log sqrt grad jacobian dot matmul norm

Output ONLY valid DSL source. No markdown fences, no prose.
"#;

const SYS_GEN: &str  = "You are a scientific computing compiler. Convert natural language \
                         to a scientific DSL. Output only valid DSL. No markdown.";
const SYS_EXPLAIN: &str = "You are a scientific computing expert. Explain DSL programs clearly.";

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Convert natural-language intent → DSL source → validated Program.
pub fn intent_to_dsl(intent: &str) -> Result<(String, Program), DslError> {
    let prompt = format!("{}\n\nGenerate DSL for:\n{}", DSL_GRAMMAR, intent);
    let raw    = call_api(SYS_GEN, &prompt)?;
    let src    = strip_fences(&raw);
    let prog   = parser::parse(&src).map_err(|e| {
        DslError::LlmValidationError(format!(
            "LLM output failed to parse.\nSource:\n{}\nError: {}", src, e))
    })?;
    Ok((src, prog))
}

/// Explain what a DSL source file computes.
pub fn explain_dsl(source: &str) -> Result<String, DslError> {
    let prompt = format!("Explain what this scientific DSL program computes:\n\n{}", source);
    call_api(SYS_EXPLAIN, &prompt)
}

/// Scaffold DSL for a named scientific equation with constraints.
pub fn scaffold_equation(equation: &str, constraints: &str) -> Result<String, DslError> {
    let prompt = format!(
        "{}\n\nGenerate DSL for: {}\nConstraints: {}\nOutput only DSL.",
        DSL_GRAMMAR, equation, constraints
    );
    let raw = call_api(SYS_GEN, &prompt)?;
    Ok(strip_fences(&raw))
}

/// Ask the LLM to suggest GPU parallelism rewrites for a DSL snippet.
pub fn suggest_gpu_opts(source: &str) -> Result<String, DslError> {
    let prompt = format!(
        "Given this scientific DSL, identify which loops can be GPU-parallelised \
         and suggest rewrites:\n\n{}", source
    );
    call_api(SYS_EXPLAIN, &prompt)
}

// ─────────────────────────────────────────────────────────────────────────────
// HTTP via curl subprocess
// ─────────────────────────────────────────────────────────────────────────────

fn call_api(system: &str, user_msg: &str) -> Result<String, DslError> {
    let api_key = env::var("ANTHROPIC_API_KEY")
        .map_err(|_| DslError::LlmError("ANTHROPIC_API_KEY not set".to_string()))?;

    let body = json!({
        "model":      MODEL,
        "max_tokens": 2048,
        "system":     system,
        "messages":   [{ "role": "user", "content": user_msg }]
    });

    let body_str = serde_json::to_string(&body)
        .map_err(|e| DslError::LlmError(e.to_string()))?;

    let output = Command::new("curl")
        .args([
            "--silent", "--fail-with-body",
            "-X", "POST",
            API_URL,
            "-H", "content-type: application/json",
            "-H", &format!("x-api-key: {}", api_key),
            "-H", "anthropic-version: 2023-06-01",
            "-d", &body_str,
        ])
        .output()
        .map_err(|e| DslError::LlmError(format!("Failed to run curl: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(DslError::LlmError(format!(
            "curl exited {:?}\nstdout: {}\nstderr: {}",
            output.status.code(), stdout, stderr
        )));
    }

    let resp_str = String::from_utf8_lossy(&output.stdout);
    let resp: Value = serde_json::from_str(&resp_str)
        .map_err(|e| DslError::LlmError(format!("JSON parse error: {}\nRaw: {}", e, resp_str)))?;

    // Check for API-level errors
    if let Some(err) = resp.get("error") {
        return Err(DslError::LlmError(format!("API error: {}", err)));
    }

    // Extract first text block
    resp["content"]
        .as_array()
        .and_then(|arr| arr.iter().find(|b| b["type"] == "text"))
        .and_then(|b| b["text"].as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| DslError::LlmError("No text block in API response".to_string()))
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn strip_fences(s: &str) -> String {
    let s = s.trim();
    if s.starts_with("```") {
        let after = s.find('\n').map(|i| &s[i + 1..]).unwrap_or(s);
        let before = after.rfind("```").map(|i| &after[..i]).unwrap_or(after);
        before.trim().to_string()
    } else {
        s.to_string()
    }
}
