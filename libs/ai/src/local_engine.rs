// ═══════════════════════════════════════════════════════════════════════════════
// محرك AI محلي - لا يتطلب llama.cpp خارجي
// Local AI Engine - No external llama.cpp required
// ═══════════════════════════════════════════════════════════════════════════════
// يدعم:
// - تحويل النص العربي إلى كود بلغة المرجع (محلياً بدون شبكة)
// - ذاكرة تخزين مؤقت للنتائج (Caching)
// - تحليل نحوي وسياقي للنص العربي
// ═══════════════════════════════════════════════════════════════════════════════

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// نتيجة الاستدلال المحلي
#[derive(Debug, Clone)]
pub struct LocalInferenceResult {
    /// النص المُنتج
    pub text: String,
    /// عدد التوكنات المقدرة
    pub tokens: usize,
    /// الوقت المستغرق (ms)
    pub duration_ms: u64,
    /// الثقة في النتيجة
    pub confidence: f32,
}

/// سياق النص العربي
#[derive(Debug, Clone)]
pub struct ArabicContext {
    /// الكلمات المفتاحية المكتشفة
    pub keywords: Vec<String>,
    /// النية المكتشفة
    pub intent: IntentType,
    /// المتغيرات المقترحة
    pub variables: HashMap<String, String>,
    /// مستوى الثقة
    pub confidence: f32,
}

/// أنواع النية
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IntentType {
    /// إنشاء متغير
    VariableDeclaration,
    /// طباعة
    Print,
    /// شرط
    Condition,
    /// دالة
    Function,
    /// حلقة
    Loop,
    /// تصدير
    Export,
    /// استيراد
    Import,
    /// فئة/كائن
    Class,
    /// معالجة قائمة
    List,
    /// معالجة قاموس
    Dictionary,
    /// عملية حسابية
    Arithmetic,
    /// مقارنة
    Comparison,
    /// غير معروف
    Unknown,
}

/// محرك AI المحلي
pub struct LocalAIEngine {
    /// ذاكرة التخزين المؤقت
    cache: HashMap<String, (LocalInferenceResult, Instant)>,
    /// مدة صلاحية الـ cache
    cache_ttl: Duration,
    /// سجل التعلم
    learning_history: Vec<(String, String)>,
    /// إحصائيات
    stats: EngineStats,
}

/// إحصائيات المحرك
#[derive(Debug, Clone, Default)]
pub struct EngineStats {
    pub total_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub average_duration_ms: u64,
}

impl LocalAIEngine {
    /// إنشاء محرك جديد
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            cache_ttl: Duration::from_secs(1800), // 30 دقيقة
            learning_history: Vec::new(),
            stats: EngineStats::default(),
        }
    }

    /// تحويل نص عربي إلى كود
    pub fn text_to_code(&mut self, text: &str) -> LocalInferenceResult {
        self.infer(text)
    }

    /// تشغيل الاستدلال
    pub fn infer(&mut self, text: &str) -> LocalInferenceResult {
        self.stats.total_requests += 1;

        // التحقق من الـ cache
        if let Some((result, timestamp)) = self.cache.get(text) {
            if timestamp.elapsed() < self.cache_ttl {
                self.stats.cache_hits += 1;
                return result.clone();
            }
        }

        self.stats.cache_misses += 1;
        let start = Instant::now();

        // تحليل النص
        let context = self.analyze_arabic_text(text);

        // توليد الكود
        let code = self.generate_code(&context, text);

        let result = LocalInferenceResult {
            text: code,
            tokens: text.split_whitespace().count(),
            duration_ms: start.elapsed().as_millis() as u64,
            confidence: context.confidence,
        };

        // تخزين في الـ cache
        self.cache.insert(text.to_string(), (result.clone(), Instant::now()));

        // تحديث الإحصائيات
        self.stats.average_duration_ms = 
            (self.stats.average_duration_ms * (self.stats.total_requests - 1) + result.duration_ms) 
            / self.stats.total_requests;

        // تخزين في سجل التعلم
        self.learning_history.push((text.to_string(), result.text.clone()));

        result
    }

    /// تحليل النص العربي
    fn analyze_arabic_text(&self, text: &str) -> ArabicContext {
        let text_lower = text.to_lowercase();
        let mut keywords = Vec::new();
        let mut variables = HashMap::new();
        let mut confidence = 0.8;

        // استخراج الكلمات المفتاحية
        let keyword_patterns = [
            ("متغير", "variable"),
            ("أنشئ", "create"),
            ("اكتب", "write"),
            ("اطبع", "print"),
            ("اعرض", "display"),
            ("إذا", "if"),
            ("شرط", "condition"),
            ("دالة", "function"),
            ("وظيفة", "function"),
            ("كرر", "repeat"),
            ("حلقة", "loop"),
            ("طالما", "while"),
            ("صدر", "export"),
            ("استورد", "import"),
            ("فئة", "class"),
            ("كائن", "object"),
            ("قائمة", "list"),
            ("مصفوفة", "array"),
            ("قاموس", "dictionary"),
            ("أضف", "add"),
            ("احسب", "calculate"),
            ("اجمع", "sum"),
            ("اطرح", "subtract"),
            ("اضرب", "multiply"),
            ("اقسم", "divide"),
        ];

        for (arabic, english) in keyword_patterns {
            if text_lower.contains(arabic) {
                keywords.push(english.to_string());
            }
        }

        // تحديد النية
        let intent = self.detect_intent(&text_lower);

        // استخراج المتغيرات المقترحة
        variables = self.extract_variables(text, &intent);

        // حساب الثقة
        if keywords.is_empty() {
            confidence = 0.3;
        } else if keywords.len() == 1 {
            confidence = 0.6;
        }

        ArabicContext {
            keywords,
            intent,
            variables,
            confidence,
        }
    }

    /// تحديد النية
    fn detect_intent(&self, text: &str) -> IntentType {
        // التصدير له أولوية قصوى
        if text.contains("صدر") && (text.contains("برنامج") || text.contains("البرنامج")) {
            return IntentType::Export;
        }
        
        // ترتيب الأولويات
        if text.contains("متغير") || text.contains("أنشئ متغير") || text.contains("عرف متغير") {
            return IntentType::VariableDeclaration;
        }
        if text.contains("اطبع") || text.contains("اعرض") || text.contains("اكتب") {
            return IntentType::Print;
        }
        if text.contains("إذا") || text.contains("شرط") {
            return IntentType::Condition;
        }
        if text.contains("دالة") || text.contains("وظيفة") {
            return IntentType::Function;
        }
        if text.contains("كرر") || text.contains("حلقة") || text.contains("طالما") {
            return IntentType::Loop;
        }
        if text.contains("فئة") || text.contains("كائن") {
            return IntentType::Class;
        }
        if text.contains("قائمة") || text.contains("مصفوفة") {
            return IntentType::List;
        }
        if text.contains("قاموس") || text.contains("خريطة") {
            return IntentType::Dictionary;
        }
        if text.contains("اجمع") || text.contains("أضف") {
            return IntentType::Arithmetic;
        }
        if text.contains("أكبر") || text.contains("أصغر") || text.contains("يساوي") {
            return IntentType::Comparison;
        }
        if text.contains("استورد") {
            return IntentType::Import;
        }

        IntentType::Unknown
    }

    /// استخراج المتغيرات
    fn extract_variables(&self, text: &str, intent: &IntentType) -> HashMap<String, String> {
        let mut vars = HashMap::new();
        let words: Vec<&str> = text.split_whitespace().collect();

        match intent {
            IntentType::VariableDeclaration => {
                for (i, word) in words.iter().enumerate() {
                    if *word == "متغير" || *word == "أنشئ" {
                        if let Some(name) = words.get(i + 1) {
                            if !["يساوي", "بقيمة", "القيمة", "الذي", "التي"].contains(name) {
                                vars.insert("name".to_string(), name.to_string());
                            }
                        }
                    }
                    if *word == "يساوي" || *word == "بقيمة" {
                        if let Some(value) = words.get(i + 1) {
                            vars.insert("value".to_string(), value.to_string());
                        }
                    }
                }
            }
            IntentType::Loop => {
                // استخراج عدد التكرارات
                let nums: String = text.chars().filter(|c| c.is_ascii_digit()).collect();
                if !nums.is_empty() {
                    vars.insert("count".to_string(), nums);
                }
            }
            IntentType::Condition => {
                // استخراج رقم المقارنة
                let nums: String = text.chars().filter(|c| c.is_ascii_digit()).collect();
                if !nums.is_empty() {
                    vars.insert("threshold".to_string(), nums);
                }
            }
            _ => {}
        }

        vars
    }

    /// توليد الكود
    fn generate_code(&self, context: &ArabicContext, original_text: &str) -> String {
        match context.intent {
            IntentType::VariableDeclaration => self.gen_variable(context, original_text),
            IntentType::Print => self.gen_print(original_text),
            IntentType::Condition => self.gen_condition(context, original_text),
            IntentType::Function => self.gen_function(original_text),
            IntentType::Loop => self.gen_loop(context, original_text),
            IntentType::Export => self.gen_export(original_text),
            IntentType::List => self.gen_list(original_text),
            IntentType::Dictionary => self.gen_dictionary(original_text),
            IntentType::Class => self.gen_class(original_text),
            IntentType::Import => self.gen_import(original_text),
            IntentType::Arithmetic => self.gen_arithmetic(original_text),
            IntentType::Comparison => self.gen_comparison(original_text),
            IntentType::Unknown => self.gen_unknown(original_text),
        }
    }

    /// توليد إعلان متغير
    fn gen_variable(&self, context: &ArabicContext, _text: &str) -> String {
        let name = context.variables.get("name").map(|s| s.as_str()).unwrap_or("س");
        let value = context.variables.get("value").map(|s| s.as_str()).unwrap_or("0");
        format!("متغير {} = {}؛", name, value)
    }

    /// توليد أمر طباعة
    fn gen_print(&self, text: &str) -> String {
        let content = text
            .replace("اطبع", "")
            .replace("اعرض", "")
            .replace("اكتب", "")
            .replace("رسالة", "")
            .replace("نص", "")
            .trim()
            .to_string();

        if content.is_empty() {
            "اطبع(\"مرحبا بالعالم\")؛".to_string()
        } else {
            format!("اطبع(\"{}\")؛", content)
        }
    }

    /// توليد شرط
    fn gen_condition(&self, context: &ArabicContext, text: &str) -> String {
        let text_lower = text.to_lowercase();
        
        // استخراج المتغير للمقارنة
        let words: Vec<&str> = text.split_whitespace().collect();
        let var_name = words
            .iter()
            .skip_while(|w| **w != "كان")
            .nth(1)
            .unwrap_or(&"س");

        // تحديد العملية
        let (op, _threshold) = if text_lower.contains("أكبر") {
            (">", context.variables.get("threshold").map(|s| s.as_str()).unwrap_or("10"))
        } else if text_lower.contains("أصغر") {
            ("<", context.variables.get("threshold").map(|s| s.as_str()).unwrap_or("10"))
        } else if text_lower.contains("يساوي") {
            ("==", context.variables.get("threshold").map(|s| s.as_str()).unwrap_or("10"))
        } else {
            ("==", "صحيح")
        };

        // استخراج الإجراء
        let body = if text_lower.contains("اطبع") {
            let msg = if text_lower.contains("مرحبا") {
                "مرحبا"
            } else if text_lower.contains("كبير") {
                "كبير"
            } else if text_lower.contains("صغير") {
                "صغير"
            } else {
                "تم"
            };
            format!("اطبع(\"{}\")؛", msg)
        } else {
            "اطعب(\"الشرط محقق\")؛".to_string()
        };

        format!("إذا {} {} {{\n    {}\n}}", var_name, op, body)
    }

    /// توليد دالة
    fn gen_function(&self, text: &str) -> String {
        let text_lower = text.to_lowercase();

        let (name, params, body) = if text_lower.contains("جمع") || text_lower.contains("تضيف") {
            ("اجمع", "أ، ب", "أعطِ أ + ب؛")
        } else if text_lower.contains("ضرب") || text_lower.contains("تضرب") {
            ("اضرب", "أ، ب", "أعطِ أ * ب؛")
        } else if text_lower.contains("طرح") || text_lower.contains("تطرح") {
            ("اطرح", "أ، ب", "أعطِ أ - ب؛")
        } else if text_lower.contains("قسم") {
            ("اقسم", "أ، ب", "أعطِ أ / ب؛")
        } else {
            // استخراج الاسم
            let name = text
                .split_whitespace()
                .find(|w| w.len() > 2 && !["أنشئ", "دالة", "وظيفة", "التي", "تقوم", "بـ"].contains(w))
                .unwrap_or("دالة_جديدة");
            (name, "", "أعطِ لا_شيء؛")
        };

        format!("دالة {}({}) {{\n    {}\n}}", name, params, body)
    }

    /// توليد حلقة
    fn gen_loop(&self, context: &ArabicContext, text: &str) -> String {
        let text_lower = text.to_lowercase();

        // تحديد عدد التكرارات
        let count = if text_lower.contains("ثلاث") || text.contains("3") || text.contains("٣") {
            3
        } else if text_lower.contains("خمس") || text.contains("5") || text.contains("٥") {
            5
        } else if text_lower.contains("عشر") || text.contains("10") || text.contains("١٠") {
            10
        } else if text_lower.contains("مرتين") || text.contains("2") || text.contains("٢") {
            2
        } else {
            context
                .variables
                .get("count")
                .and_then(|s| s.parse().ok())
                .unwrap_or(1)
        };

        // تحديد الإجراء
        let body = if text_lower.contains("اطبع") {
            let msg = if text_lower.contains("مرحبا") {
                "مرحبا"
            } else {
                "تكرار"
            };
            format!("اطبع(\"{}\")؛", msg)
        } else {
            "اطبع(\"تكرار\")؛".to_string()
        };

        format!(
            "متغير ع = 0؛\nطالما ع < {} {{\n    {}\n    ع = ع + 1؛\n}}",
            count, body
        )
    }

    /// توليد تصدير
    fn gen_export(&self, text: &str) -> String {
        let text_lower = text.to_lowercase();

        // استخراج اسم البرنامج
        let program_name = if text.contains("البرنامج") {
            let after = text.split("البرنامج").nth(1).unwrap_or("").trim();
            if after.contains("على") {
                after.split("على").next().unwrap_or("myapp").trim().replace(" ", "_")
            } else {
                after.split_whitespace().next().unwrap_or("myapp").to_string()
            }
        } else {
            "myapp".to_string()
        };

        // تحديد المنصة
        let platform = if text_lower.contains("ويندوز") || text_lower.contains("windows") {
            "windows"
        } else if text_lower.contains("لينكس") || text_lower.contains("linux") {
            "linux"
        } else if text_lower.contains("ماك") || text_lower.contains("mac") {
            "macos"
        } else if text_lower.contains("ويب") || text_lower.contains("web") {
            "web"
        } else {
            "windows"
        };

        format!(
            "// 📦 تصدير البرنامج\n// البرنامج: {}\n// المنصة: {}\n\nصدر البرنامج \"{}\" على {}؛",
            program_name, platform, program_name, platform
        )
    }

    /// توليد قائمة
    fn gen_list(&self, _text: &str) -> String {
        "قائمة أرقام = [1، 2، 3، 4، 5]؛\n\n// للوصول: أرقام[0] → 1\n// للإضافة: أرقام.أضف(6)؛".to_string()
    }

    /// توليد قاموس
    fn gen_dictionary(&self, _text: &str) -> String {
        "قاموس بيانات = {\"اسم\": \"أحمد\"، \"عمر\": 25}؛\n\n// للوصول: بيانات[\"اسم\"] → \"أحمد\"".to_string()
    }

    /// توليد فئة
    fn gen_class(&self, text: &str) -> String {
        let name = text
            .split_whitespace()
            .find(|w| w.len() > 2 && !["فئة", "كائن", "جديد", "أنشئ"].contains(w))
            .unwrap_or("كائن_جديد");

        format!(
            "فئة {} {{\n    // خصائص\n    متغير اسم = \"\"؛\n\n    // دالة البناء\n    دالة أنشئ(الاسم) {{\n        هذا.اسم = الاسم؛\n    }}\n\n    // دالة عرض\n    دالة عرض() {{\n        اطعب(هذا.اسم)؛\n    }}\n}}",
            name
        )
    }

    /// توليد استيراد
    fn gen_import(&self, text: &str) -> String {
        let module = text
            .split_whitespace()
            .find(|w| w.len() > 2 && !["استورد", "من", "استيراد"].contains(w))
            .unwrap_or("وحدة");

        format!("استورد {}؛", module)
    }

    /// توليد عملية حسابية
    fn gen_arithmetic(&self, text: &str) -> String {
        let text_lower = text.to_lowercase();

        if text_lower.contains("جمع") || text_lower.contains("أضف") {
            "متغير نتيجة = أ + ب؛"
        } else if text_lower.contains("اطرح") || text_lower.contains("طرح") {
            "متغير نتيجة = أ - ب؛"
        } else if text_lower.contains("اضرب") || text_lower.contains("ضرب") {
            "متغير نتيجة = أ * ب؛"
        } else if text_lower.contains("اقسم") || text_lower.contains("قسم") {
            "متغير نتيجة = أ / ب؛"
        } else {
            "متغير نتيجة = أ + ب؛"
        }.to_string()
    }

    /// توليد مقارنة
    fn gen_comparison(&self, text: &str) -> String {
        let text_lower = text.to_lowercase();

        let op = if text_lower.contains("أكبر") {
            ">"
        } else if text_lower.contains("أصغر") {
            "<"
        } else if text_lower.contains("يساوي") {
            "=="
        } else if text_lower.contains("لا يساوي") {
            "!="
        } else {
            "=="
        };

        format!("متغير نتيجة = أ {} ب؛", op)
    }

    /// توليد لني غير معروف
    fn gen_unknown(&self, text: &str) -> String {
        format!("// ⚠️ لم أفهم المطلوب: {}\n// حاول صياغته بشكل مختلف\n// مثال: \"اطبع مرحبا\" أو \"أنشئ متغير س يساوي 10\"", text)
    }

    /// الحصول على الإحصائيات
    pub fn get_stats(&self) -> &EngineStats {
        &self.stats
    }

    /// مسح الـ cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// الحصول على حجم الـ cache
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// إضافة مثال للتعلم
    pub fn learn(&mut self, input: &str, output: &str) {
        self.learning_history.push((input.to_string(), output.to_string()));
    }

    /// الحصول على عدد الأمثلة المتعلمة
    pub fn learning_count(&self) -> usize {
        self.learning_history.len()
    }
}

impl Default for LocalAIEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// دالة سهلة للتحويل
pub fn local_text_to_code(text: &str) -> String {
    let mut engine = LocalAIEngine::new();
    engine.text_to_code(text).text
}

// ═══════════════════════════════════════════════════════════════════════════════
// اختبارات
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = LocalAIEngine::new();
        assert_eq!(engine.cache_size(), 0);
    }

    #[test]
    fn test_variable() {
        let mut engine = LocalAIEngine::new();
        let result = engine.text_to_code("أنشئ متغير س يساوي 10");
        assert!(result.text.contains("متغير"));
        assert!(result.text.contains("س"));
    }

    #[test]
    fn test_print() {
        let mut engine = LocalAIEngine::new();
        let result = engine.text_to_code("اطبع مرحبا");
        assert!(result.text.contains("اطبع"));
    }

    #[test]
    fn test_condition() {
        let mut engine = LocalAIEngine::new();
        let result = engine.text_to_code("إذا كان س أكبر من 10 اطبع كبير");
        assert!(result.text.contains("إذا"));
    }

    #[test]
    fn test_function() {
        let mut engine = LocalAIEngine::new();
        let result = engine.text_to_code("أنشئ دالة تجمع رقمين");
        assert!(result.text.contains("دالة"));
    }

    #[test]
    fn test_loop() {
        let mut engine = LocalAIEngine::new();
        let result = engine.text_to_code("كرر طباعة مرحبا 5 مرات");
        assert!(result.text.contains("طالما"));
    }

    #[test]
    fn test_export() {
        let mut engine = LocalAIEngine::new();
        let result = engine.text_to_code("صدر البرنامج myapp على ويندوز");
        assert!(result.text.contains("صدر"));
    }

    #[test]
    fn test_cache() {
        let mut engine = LocalAIEngine::new();
        
        // أول طلب
        let _ = engine.text_to_code("اطبع اختبار");
        assert_eq!(engine.cache_size(), 1);
        
        // طلب متطابق - من الـ cache
        let _ = engine.text_to_code("اطبع اختبار");
        let stats = engine.get_stats();
        assert_eq!(stats.cache_hits, 1);
    }

    #[test]
    fn test_learning() {
        let mut engine = LocalAIEngine::new();
        engine.learn("مثال", "نتيجة");
        assert_eq!(engine.learning_count(), 1);
    }

    #[test]
    fn test_intent_detection() {
        let engine = LocalAIEngine::new();
        
        assert_eq!(engine.detect_intent("اطبع مرحبا"), IntentType::Print);
        assert_eq!(engine.detect_intent("أنشئ متغير س"), IntentType::VariableDeclaration);
        assert_eq!(engine.detect_intent("إذا كان س أكبر من 5"), IntentType::Condition);
        assert_eq!(engine.detect_intent("كرر 5 مرات"), IntentType::Loop);
        assert_eq!(engine.detect_intent("صدر البرنامج"), IntentType::Export);
    }
}
