// ═══════════════════════════════════════════════════════════════════════════════
// Pipeline Module - لغة المرجع
// ═══════════════════════════════════════════════════════════════════════════════

pub mod engine;

pub use engine::{
    generate_code, parse_intent, run_example, run_pipeline, Intent, PipelineEngine, PipelineResult,
};
