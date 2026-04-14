// src/errors.rs — Unified error type

use thiserror::Error;

#[derive(Debug, Error)]
pub enum DslError {
    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("IR lowering error: {0}")]
    IrError(String),

    #[error("Codegen error: {0}")]
    CodegenError(String),

    #[error("LLM API error: {0}")]
    LlmError(String),

    #[error("LLM output validation error: {0}")]
    LlmValidationError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
