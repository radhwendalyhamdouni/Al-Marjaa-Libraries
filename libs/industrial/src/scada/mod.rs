// ═══════════════════════════════════════════════════════════════════════════════
// SCADA Historian - قاعدة بيانات تاريخية للأنظمة الصناعية
// ═══════════════════════════════════════════════════════════════════════════════

//! # SCADA Historian
//! 
//! قاعدة بيانات تاريخية لتخزين واسترجاع البيانات الصناعية.
//! 
//! ## الميزات
//! - تخزين بيانات Tags مع timestamps
//! - استرجاع بيانات حسب الفترة الزمنية
//! - حساب إحصائيات (متوسط، حد أقصى، حد أدنى)
//! - ضغط البيانات التلقائي
//! - تصدير البيانات (CSV, JSON)
//! 
//! ## مثال
//! 
//! ```rust
//! use almarjaa::industrial::scada::{ScadaHistorian, TagValue};
//! use chrono::{Utc, Duration};
//! 
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut historian = ScadaHistorian::new("scada.db")?;
//! 
//! // تخزين قيمة
//! historian.store("Temperature", 25.5)?;
//! 
//! // استرجاع آخر قيمة
//! let last = historian.get_last_value("Temperature")?;
//! 
//! // استرجاع بيانات لآخر ساعة
//! let now = Utc::now();
//! let data = historian.get_range("Temperature", now - Duration::hours(1), now)?;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::path::Path;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
// الأنواع الأساسية
// ═══════════════════════════════════════════════════════════════════════════════

/// جودة البيانات
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataQuality {
    /// جودة جيدة
    Good,
    /// جودة مشكوك فيها
    Uncertain,
    /// جودة سيئة
    Bad,
    /// غير متصل
    Disconnected,
}

impl Default for DataQuality {
    fn default() -> Self {
        Self::Good
    }
}

/// قيمة Tag مسجلة
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagRecord {
    /// اسم Tag
    pub tag_name: String,
    /// القيمة
    pub value: f64,
    /// الجودة
    pub quality: DataQuality,
    /// وقت التسجيل
    pub timestamp: DateTime<Utc>,
}

/// إحصائيات Tag
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagStatistics {
    /// اسم Tag
    pub tag_name: String,
    /// عدد القراءات
    pub count: u64,
    /// المتوسط
    pub average: f64,
    /// الحد الأدنى
    pub min: f64,
    /// الحد الأقصى
    pub max: f64,
    /// الانحراف المعياري
    pub std_dev: f64,
    /// وقت البدء
    pub start_time: DateTime<Utc>,
    /// وقت الانتهاء
    pub end_time: DateTime<Utc>,
}

/// تكوين Tag
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagConfig {
    /// اسم Tag
    pub name: String,
    /// الوصف
    pub description: String,
    /// الوحدة
    pub unit: String,
    /// نطاق القيم (min, max)
    pub range: (f64, f64),
    /// حد التحذير الأدنى
    pub low_warning: Option<f64>,
    /// حد التحذير الأعلى
    pub high_warning: Option<f64>,
    /// حد الإنذار الأدنى
    pub low_alarm: Option<f64>,
    /// حد الإنذار الأعلى
    pub high_alarm: Option<f64>,
    /// فترة التخزين (ثواني)
    pub archive_period: u32,
    /// تفعيل الضغط
    pub compression_enabled: bool,
    /// حد الضغط
    pub compression_deviation: f64,
}

impl TagConfig {
    /// إنشاء تكوين جديد
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            unit: String::new(),
            range: (0.0, 100.0),
            low_warning: None,
            high_warning: None,
            low_alarm: None,
            high_alarm: None,
            archive_period: 1,
            compression_enabled: true,
            compression_deviation: 0.1,
        }
    }
    
    /// تعيين الوصف
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }
    
    /// تعيين الوحدة
    pub fn with_unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = unit.into();
        self
    }
    
    /// تعيين النطاق
    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.range = (min, max);
        self
    }
    
    /// تعيين حدود التحذير والإنذار
    pub fn with_limits(mut self, low_warn: f64, high_warn: f64, low_alarm: f64, high_alarm: f64) -> Self {
        self.low_warning = Some(low_warn);
        self.high_warning = Some(high_warn);
        self.low_alarm = Some(low_alarm);
        self.high_alarm = Some(high_alarm);
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// قاعدة البيانات في الذاكرة
// ═══════════════════════════════════════════════════════════════════════════════

/// SCADA Historian (في الذاكرة للتبسيط)
#[derive(Debug)]
pub struct ScadaHistorian {
    /// اسم قاعدة البيانات
    db_name: String,
    /// تكوينات Tags
    tag_configs: HashMap<String, TagConfig>,
    /// البيانات المخزنة
    data: HashMap<String, Vec<TagRecord>>,
    /// آخر قيم
    last_values: HashMap<String, TagRecord>,
    /// إحصائيات يومية
    daily_stats: HashMap<String, TagStatistics>,
    /// الحد الأقصى للسجلات لكل Tag
    max_records_per_tag: usize,
}

impl ScadaHistorian {
    /// إنشاء historian جديد
    pub fn new(db_name: impl Into<String>) -> Self {
        Self {
            db_name: db_name.into(),
            tag_configs: HashMap::new(),
            data: HashMap::new(),
            last_values: HashMap::new(),
            daily_stats: HashMap::new(),
            max_records_per_tag: 100000,
        }
    }
    
    /// فتح قاعدة بيانات من ملف
    #[cfg(feature = "scada-db")]
    pub async fn open(path: &Path) -> Result<Self, String> {
        // سيتم التنفيذ مع sqlx
        Ok(Self::new(path.to_string_lossy()))
    }
    
    /// تسجيل Tag جديد
    pub fn register_tag(&mut self, config: TagConfig) {
        let name = config.name.clone();
        self.tag_configs.insert(name.clone(), config);
        self.data.insert(name, Vec::new());
    }
    
    /// تخزين قيمة
    pub fn store(&mut self, tag_name: &str, value: f64) -> Result<(), String> {
        self.store_with_quality(tag_name, value, DataQuality::Good)
    }
    
    /// تخزين قيمة مع جودة
    pub fn store_with_quality(&mut self, tag_name: &str, value: f64, quality: DataQuality) -> Result<(), String> {
        // التحقق من وجود Tag
        if !self.tag_configs.contains_key(tag_name) {
            // تسجيل تلقائي
            self.register_tag(TagConfig::new(tag_name));
        }
        
        // التحقق من النطاق
        if let Some(config) = self.tag_configs.get(tag_name) {
            if value < config.range.0 || value > config.range.1 {
                tracing::warn!(
                    "Tag '{}' قيمة {} خارج النطاق [{}, {}]",
                    tag_name, value, config.range.0, config.range.1
                );
            }
        }
        
        let record = TagRecord {
            tag_name: tag_name.to_string(),
            value,
            quality,
            timestamp: Utc::now(),
        };
        
        // تحديث آخر قيمة
        self.last_values.insert(tag_name.to_string(), record.clone());
        
        // إضافة للبيانات
        if let Some(data) = self.data.get_mut(tag_name) {
            data.push(record);
            
            // ضغط البيانات إذا تجاوزت الحد
            if data.len() > self.max_records_per_tag {
                self.compress_data(tag_name);
            }
        }
        
        Ok(())
    }
    
    /// الحصول على آخر قيمة
    pub fn get_last_value(&self, tag_name: &str) -> Option<&TagRecord> {
        self.last_values.get(tag_name)
    }
    
    /// الحصول على بيانات لفترة زمنية
    pub fn get_range(&self, tag_name: &str, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<&TagRecord> {
        if let Some(data) = self.data.get(tag_name) {
            data.iter()
                .filter(|r| r.timestamp >= start && r.timestamp <= end)
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// الحصول على كل البيانات لـ Tag
    pub fn get_all(&self, tag_name: &str) -> Vec<&TagRecord> {
        self.data.get(tag_name)
            .map(|d| d.iter().collect())
            .unwrap_or_default()
    }
    
    /// حساب الإحصائيات
    pub fn calculate_statistics(&self, tag_name: &str, start: DateTime<Utc>, end: DateTime<Utc>) -> Option<TagStatistics> {
        let records = self.get_range(tag_name, start, end);
        
        if records.is_empty() {
            return None;
        }
        
        let values: Vec<f64> = records.iter().map(|r| r.value).collect();
        let count = values.len() as u64;
        let sum: f64 = values.iter().sum();
        let average = sum / count as f64;
        
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        // حساب الانحراف المعياري
        let variance = values.iter()
            .map(|v| (v - average).powi(2))
            .sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();
        
        Some(TagStatistics {
            tag_name: tag_name.to_string(),
            count,
            average,
            min,
            max,
            std_dev,
            start_time: start,
            end_time: end,
        })
    }
    
    /// ضغط البيانات
    fn compress_data(&mut self, tag_name: &str) {
        if let Some(data) = self.data.get_mut(tag_name) {
            let config = self.tag_configs.get(tag_name);
            let deviation = config
                .map(|c| c.compression_deviation)
                .unwrap_or(0.1);
            
            // خوارزمية ضغط بسيطة: إزالة القيم المتشابهة
            let mut compressed = Vec::new();
            let mut last_kept: Option<&TagRecord> = None;
            
            for record in data.iter() {
                let should_keep = match last_kept {
                    None => true,
                    Some(last) => (record.value - last.value).abs() >= deviation,
                };
                
                if should_keep {
                    compressed.push(record.clone());
                    last_kept = Some(compressed.last().unwrap());
                }
            }
            
            tracing::info!(
                "ضغط بيانات '{}': {} -> {} سجل",
                tag_name, data.len(), compressed.len()
            );
            
            *data = compressed;
        }
    }
    
    /// مسح بيانات Tag
    pub fn clear(&mut self, tag_name: &str) {
        if let Some(data) = self.data.get_mut(tag_name) {
            data.clear();
        }
        self.last_values.remove(tag_name);
    }
    
    /// مسح كل البيانات
    pub fn clear_all(&mut self) {
        for data in self.data.values_mut() {
            data.clear();
        }
        self.last_values.clear();
    }
    
    /// تصدير إلى CSV
    pub fn export_csv(&self, tag_name: &str) -> Result<String, String> {
        let data = self.data.get(tag_name)
            .ok_or_else(|| format!("Tag '{}' غير موجود", tag_name))?;
        
        let mut csv = String::from("timestamp,value,quality\n");
        
        for record in data {
            let quality = match record.quality {
                DataQuality::Good => "Good",
                DataQuality::Uncertain => "Uncertain",
                DataQuality::Bad => "Bad",
                DataQuality::Disconnected => "Disconnected",
            };
            csv.push_str(&format!(
                "{},{},{}\n",
                record.timestamp.to_rfc3339(),
                record.value,
                quality
            ));
        }
        
        Ok(csv)
    }
    
    /// تصدير إلى JSON
    pub fn export_json(&self, tag_name: &str) -> Result<String, String> {
        let data = self.data.get(tag_name)
            .ok_or_else(|| format!("Tag '{}' غير موجود", tag_name))?;
        
        serde_json::to_string_pretty(data)
            .map_err(|e| format!("خطأ في تصدير JSON: {}", e))
    }
    
    /// عدد السجلات الإجمالي
    pub fn total_records(&self) -> usize {
        self.data.values().map(|d| d.len()).sum()
    }
    
    /// عدد Tags
    pub fn tag_count(&self) -> usize {
        self.tag_configs.len()
    }
    
    /// قائمة أسماء Tags
    pub fn tag_names(&self) -> Vec<&String> {
        self.tag_configs.keys().collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// سجل الأحداث
// ═══════════════════════════════════════════════════════════════════════════════

/// نوع الحدث
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventType {
    /// إنذار نشط
    AlarmActive,
    /// إنذار غير نشط
    AlarmInactive,
    /// تحذير
    Warning,
    /// تأكيد إنذار
    AlarmAck,
    /// أمر تحكم
    Command,
    /// تغيير قيمة
    ValueChange,
    /// خطأ اتصال
    ConnectionError,
    /// استعادة اتصال
    ConnectionRestored,
}

/// سجل حدث
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventRecord {
    /// معرف الحدث
    pub id: u64,
    /// نوع الحدث
    pub event_type: EventType,
    /// المصدر
    pub source: String,
    /// الرسالة
    pub message: String,
    /// القيمة
    pub value: Option<f64>,
    /// المستخدم
    pub user: Option<String>,
    /// وقت الحدث
    pub timestamp: DateTime<Utc>,
    /// مؤكد
    pub acknowledged: bool,
    /// وقت التأكيد
    pub ack_time: Option<DateTime<Utc>>,
    /// المستخدم الذي أكد
    pub ack_user: Option<String>,
}

/// سجل الأحداث
#[derive(Debug)]
pub struct EventLog {
    /// الأحداث
    events: Vec<EventRecord>,
    /// عداد الأحداث
    counter: u64,
    /// الحد الأقصى
    max_events: usize,
}

impl EventLog {
    /// إنشاء سجل جديد
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            counter: 0,
            max_events: 10000,
        }
    }
    
    /// تسجيل حدث
    pub fn log(&mut self, event_type: EventType, source: impl Into<String>, message: impl Into<String>) -> u64 {
        self.counter += 1;
        
        let event = EventRecord {
            id: self.counter,
            event_type,
            source: source.into(),
            message: message.into(),
            value: None,
            user: None,
            timestamp: Utc::now(),
            acknowledged: false,
            ack_time: None,
            ack_user: None,
        };
        
        self.events.push(event);
        
        // إزالة الأحداث القديمة
        if self.events.len() > self.max_events {
            self.events.remove(0);
        }
        
        self.counter
    }
    
    /// تسجيل حدث مع قيمة
    pub fn log_with_value(&mut self, event_type: EventType, source: impl Into<String>, message: impl Into<String>, value: f64) -> u64 {
        self.counter += 1;
        
        let event = EventRecord {
            id: self.counter,
            event_type,
            source: source.into(),
            message: message.into(),
            value: Some(value),
            user: None,
            timestamp: Utc::now(),
            acknowledged: false,
            ack_time: None,
            ack_user: None,
        };
        
        self.events.push(event);
        
        if self.events.len() > self.max_events {
            self.events.remove(0);
        }
        
        self.counter
    }
    
    /// تأكيد حدث
    pub fn acknowledge(&mut self, event_id: u64, user: impl Into<String>) -> bool {
        if let Some(event) = self.events.iter_mut().find(|e| e.id == event_id) {
            event.acknowledged = true;
            event.ack_time = Some(Utc::now());
            event.ack_user = Some(user.into());
            true
        } else {
            false
        }
    }
    
    /// الحصول على الأحداث النشطة (غير المؤكدة)
    pub fn get_active(&self) -> Vec<&EventRecord> {
        self.events.iter()
            .filter(|e| !e.acknowledged)
            .collect()
    }
    
    /// الحصول على الأحداث حسب النوع
    pub fn get_by_type(&self, event_type: EventType) -> Vec<&EventRecord> {
        self.events.iter()
            .filter(|e| e.event_type == event_type)
            .collect()
    }
    
    /// الحصول على الأحداث حسب الفترة
    pub fn get_range(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<&EventRecord> {
        self.events.iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .collect()
    }
    
    /// عدد الأحداث
    pub fn count(&self) -> usize {
        self.events.len()
    }
    
    /// عدد الأحداث النشطة
    pub fn active_count(&self) -> usize {
        self.events.iter().filter(|e| !e.acknowledged).count()
    }
}

impl Default for EventLog {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_historian_creation() {
        let historian = ScadaHistorian::new("test.db");
        assert_eq!(historian.tag_count(), 0);
    }
    
    #[test]
    fn test_tag_registration() {
        let mut historian = ScadaHistorian::new("test.db");
        
        let config = TagConfig::new("Temperature")
            .with_unit("°C")
            .with_range(0.0, 100.0);
        
        historian.register_tag(config);
        
        assert_eq!(historian.tag_count(), 1);
        assert!(historian.tag_names().contains(&"Temperature".to_string()));
    }
    
    #[test]
    fn test_store_and_retrieve() {
        let mut historian = ScadaHistorian::new("test.db");
        
        historian.store("Temperature", 25.5).unwrap();
        historian.store("Temperature", 26.0).unwrap();
        
        let last = historian.get_last_value("Temperature");
        assert!(last.is_some());
        assert_eq!(last.unwrap().value, 26.0);
    }
    
    #[test]
    fn test_statistics() {
        let mut historian = ScadaHistorian::new("test.db");
        
        historian.store("Speed", 100.0).unwrap();
        historian.store("Speed", 150.0).unwrap();
        historian.store("Speed", 200.0).unwrap();
        
        let now = Utc::now();
        let stats = historian.calculate_statistics("Speed", now - Duration::hours(1), now);
        
        assert!(stats.is_some());
        let s = stats.unwrap();
        assert_eq!(s.count, 3);
        assert_eq!(s.min, 100.0);
        assert_eq!(s.max, 200.0);
        assert!((s.average - 150.0).abs() < 0.01);
    }
    
    #[test]
    fn test_export_csv() {
        let mut historian = ScadaHistorian::new("test.db");
        
        historian.store("Temperature", 25.5).unwrap();
        
        let csv = historian.export_csv("Temperature");
        assert!(csv.is_ok());
        assert!(csv.unwrap().contains("timestamp,value,quality"));
    }
    
    #[test]
    fn test_export_json() {
        let mut historian = ScadaHistorian::new("test.db");
        
        historian.store("Temperature", 25.5).unwrap();
        
        let json = historian.export_json("Temperature");
        assert!(json.is_ok());
    }
    
    #[test]
    fn test_event_log() {
        let mut log = EventLog::new();
        
        let id = log.log(EventType::AlarmActive, "Temperature", "درجة حرارة مرتفعة");
        
        assert_eq!(log.count(), 1);
        assert_eq!(log.active_count(), 1);
        
        // تأكيد الحدث
        let ack_result = log.acknowledge(id, "operator");
        assert!(ack_result);
        assert_eq!(log.active_count(), 0);
    }
    
    #[test]
    fn test_event_log_with_value() {
        let mut log = EventLog::new();
        
        log.log_with_value(EventType::AlarmActive, "Pressure", "ضغط مرتفع", 15.5);
        
        let events = log.get_active();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].value, Some(15.5));
    }
    
    #[test]
    fn test_tag_config() {
        let config = TagConfig::new("Temperature")
            .with_description("درجة حرارة الخزان")
            .with_unit("°C")
            .with_range(0.0, 100.0)
            .with_limits(10.0, 90.0, 5.0, 95.0);
        
        assert_eq!(config.name, "Temperature");
        assert_eq!(config.unit, "°C");
        assert_eq!(config.range, (0.0, 100.0));
        assert_eq!(config.low_warning, Some(10.0));
        assert_eq!(config.high_alarm, Some(95.0));
    }
}
