// ═══════════════════════════════════════════════════════════════════════════════
// محرك الذكاء الاصطناعي - لغة المرجع
// ═══════════════════════════════════════════════════════════════════════════════

pub mod arabic_nlp;
pub mod gguf_inference;
pub mod inference;
pub mod local_engine;
pub mod pipeline;
pub mod real_inference;
pub mod smart_model_manager;
pub mod unified_vibe;
pub mod vibe_advanced;

pub use pipeline::engine::{
    generate_code, parse_intent, run_example, run_pipeline, Intent, PipelineEngine, PipelineResult,
};

pub use inference::{
    create_engine, text_to_code, text_to_intent_json, AIEngine, CacheStats, InferenceCache,
    InferenceResult, ModelConfig, ModelType,
};

pub use gguf_inference::{GGUFConfig, GGUFEngine, GGUFResult};

pub use arabic_nlp::{
    ArabicNlp, Entity, EntityType, ParseResult, SentenceStructure, SentenceType, Token,
};

pub use real_inference::{text_to_code_real, RealAIEngine, RealInferenceResult, RealModelConfig};

// ═══════════════════════════════════════════════════════════════════════════════
// تصديرات Vibe Coding المتقدم
// ═══════════════════════════════════════════════════════════════════════════════

pub use vibe_advanced::{
    vibe_detect_intents,
    vibe_explain_code,
    // الدوال السهلة
    vibe_process,
    vibe_smart_completion,
    vibe_suggest_fix,
    CodeTemplate,
    ComplexityLevel,
    ConfidenceLevel,
    ContextType,
    ConversationItem,
    DetectedIntent,
    EngineStats,
    // ملاحظة: SentenceStructure و SentenceType موجودة في arabic_nlp
    // استخدام vibe_advanced::SentenceStructure as VibeSentenceStructure
    // السياق والتعلم
    ExecutionContext,
    // النوايا
    IntentType,
    LearningType,
    RelationType,
    // التحليل الدلالي
    SemanticAnalysis,
    SemanticEntity,
    SemanticEntityType,
    SemanticRelation,
    UserPattern,
    // المحرك الرئيسي
    VibeCodingEngine,
    VibeResult,
};

// ═══════════════════════════════════════════════════════════════════════════════
// تصديرات المحرك الموحد (Vibe + GGUF)
// ═══════════════════════════════════════════════════════════════════════════════

pub use unified_vibe::{
    unified_process_batch,
    unified_text_to_code,
    // الدوال السهلة
    unified_vibe_process,
    EngineState,
    EngineType,
    UnifiedStats,
    UnifiedVibeConfig,
    // المحرك الموحد
    UnifiedVibeEngine,
    UnifiedVibeResult,
};

// ═══════════════════════════════════════════════════════════════════════════════
// تصديرات مدير النماذج الذكي
// ═══════════════════════════════════════════════════════════════════════════════

pub use smart_model_manager::{
    // نتيجة الاستدلال - مع تسمية مختلفة لتجنب التعارض
    InferenceResult as SmartInferenceResult,
    // معلومات النموذج
    ModelInfo,
    ModelManagerConfig,
    ModelState,
    // المدير الرئيسي
    SmartModelManager,
};
