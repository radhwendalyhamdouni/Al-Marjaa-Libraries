// ═══════════════════════════════════════════════════════════════════════════════
// الأنظمة الصناعية - Industrial Systems Module
// ═══════════════════════════════════════════════════════════════════════════════
// نظام SCADA/HMI متكامل للإنتاج الفعلي
// الإصدار: 1.0.0
// ═══════════════════════════════════════════════════════════════════════════════

//! # الأنظمة الصناعية - Industrial Systems
//! 
//! نظام متكامل للتحكم الصناعي يشمل:
//! - **Modbus TCP/RTU**: اتصال حقيقي بأجهزة PLC وأجهزة صناعية
//! - **MQTT**: للمراقبة عن بعد و IIoT
//! - **SCADA Historian**: قاعدة بيانات تاريخية
//! - **HMI رسومية**: واجهة إنسانية-آلية رسومية
//! - **نظام الإنذارات**: تنبيهات متعددة (SMS, Email, Push)
//! - **تعريفات الأجهزة**: درايفرات لأجهزة متنوعة
//! 
//! ## مثال سريع
//! 
//! ```almarjaa
//! # إنشاء اتصال Modbus
//! متغير plc = modbus_tcp("192.168.1.100", 502)؛
//! 
//! # قراءة سجل
//! متغير قيمة = plc.اقرأ(1, 40001)؛
//! 
//! # كتابة قيمة
//! plc.اكتب(1, 40001, 100)؛
//! 
//! # إرسال بيانات للمراقبة عن بعد
//! mqtt_أرسل("factory/line1/speed", قيمة)؛
//! ```

pub mod modbus;
pub mod mqtt;
pub mod hmi;
pub mod scada;
pub mod alarms;
pub mod drivers;

// ═══════════════════════════════════════════════════════════════════════════════
// التصديرات الرئيسية
// ═══════════════════════════════════════════════════════════════════════════════

// Modbus
pub use modbus::{
    ModbusClient, ModbusConfig, ModbusType, RegisterType, RtuConfig, Parity,
};

// MQTT
pub use mqtt::{
    MqttClient, MqttConfig, MqttMessage,
};

// HMI
pub use hmi::{
    HmiApp, HmiWidget, WidgetType, Gauge, GaugeType, Chart, ChartType,
    Button, ButtonType, Indicator, HmiColor, HmiTheme,
};

// SCADA
pub use scada::{
    ScadaHistorian, TagRecord, TagStatistics, TagConfig, EventLog, EventRecord, EventType,
};

// Alarms
pub use alarms::{
    AlarmManager, AlarmDefinition, AlarmInstance, AlarmSeverity, AlarmState, AlarmType,
};

// Drivers
pub use drivers::{
    DeviceDefinition, DeviceType, DeviceStatus, DeviceManager,
    TagDefinition, siemens_s7_1200, allen_bradley_micrologix, pt100_sensor,
    siemens_sinamics_vfd, energy_meter, arduino_uno_modbus,
};

// ═══════════════════════════════════════════════════════════════════════════════
// الأنواع المشتركة
// ═══════════════════════════════════════════════════════════════════════════════

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

/// نتيجة عملية صناعية
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndustrialResult<T> {
    /// هل نجحت العملية
    pub success: bool,
    /// القيمة المرجعة
    pub value: Option<T>,
    /// رسالة خطأ إن وجدت
    pub error: Option<String>,
    /// وقت التنفيذ
    pub timestamp: DateTime<Utc>,
}

impl<T> IndustrialResult<T> {
    pub fn ok(value: T) -> Self {
        Self {
            success: true,
            value: Some(value),
            error: None,
            timestamp: Utc::now(),
        }
    }
    
    pub fn err(message: impl Into<String>) -> Self {
        Self {
            success: false,
            value: None,
            error: Some(message.into()),
            timestamp: Utc::now(),
        }
    }
}

/// حالة الاتصال
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionStatus {
    /// غير متصل
    Disconnected,
    /// جاري الاتصال
    Connecting,
    /// متصل
    Connected,
    /// خطأ في الاتصال
    Error,
    /// إعادة المحاولة
    Reconnecting,
}

/// تكوين الاتصال العام
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionConfig {
    /// عنوان المضيف
    pub host: String,
    /// المنفذ
    pub port: u16,
    /// مهلة الاتصال بالميلي ثانية
    pub timeout_ms: u64,
    /// عدد محاولات إعادة الاتصال
    pub max_retries: u32,
    /// الفاصل بين المحاولات بالميلي ثانية
    pub retry_interval_ms: u64,
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 502,
            timeout_ms: 5000,
            max_retries: 3,
            retry_interval_ms: 1000,
        }
    }
}

/// علامة صناعية (Tag)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tag {
    /// اسم العلامة
    pub name: String,
    /// وصف العلامة
    pub description: String,
    /// نوع البيانات
    pub data_type: TagDataType,
    /// القيمة الحالية
    pub value: TagValue,
    /// الوحدة
    pub unit: String,
    /// الحد الأدنى
    pub min_value: Option<f64>,
    /// الحد الأعلى
    pub max_value: Option<f64>,
    /// عتبة الإنذار المنخفض
    pub low_alarm: Option<f64>,
    /// عتبة الإنذار العالي
    pub high_alarm: Option<f64>,
    /// وقت آخر تحديث
    pub last_update: DateTime<Utc>,
    /// جودة البيانات
    pub quality: DataQuality,
}

/// أنواع بيانات العلامات
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TagDataType {
    Boolean,
    Integer,
    Float,
    String,
}

/// قيمة العلامة
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TagValue {
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
}

impl Default for TagValue {
    fn default() -> Self {
        TagValue::Float(0.0)
    }
}

/// جودة البيانات
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataQuality {
    Good,
    Bad,
    Uncertain,
    NotConnected,
}

/// مدير العلامات
pub struct TagManager {
    tags: Arc<RwLock<HashMap<String, Tag>>>,
}

impl TagManager {
    pub fn new() -> Self {
        Self {
            tags: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// إضافة علامة جديدة
    pub fn add_tag(&self, tag: Tag) {
        let mut tags = self.tags.write().unwrap();
        tags.insert(tag.name.clone(), tag);
    }
    
    /// الحصول على علامة
    pub fn get_tag(&self, name: &str) -> Option<Tag> {
        let tags = self.tags.read().unwrap();
        tags.get(name).cloned()
    }
    
    /// تحديث قيمة علامة
    pub fn update_tag(&self, name: &str, value: TagValue) -> Result<(), String> {
        let mut tags = self.tags.write().unwrap();
        if let Some(tag) = tags.get_mut(name) {
            tag.value = value;
            tag.last_update = Utc::now();
            tag.quality = DataQuality::Good;
            Ok(())
        } else {
            Err(format!("العلامة '{}' غير موجودة", name))
        }
    }
    
    /// الحصول على جميع العلامات
    pub fn get_all_tags(&self) -> Vec<Tag> {
        let tags = self.tags.read().unwrap();
        tags.values().cloned().collect()
    }
    
    /// حذف علامة
    pub fn remove_tag(&self, name: &str) -> bool {
        let mut tags = self.tags.write().unwrap();
        tags.remove(name).is_some()
    }
}

impl Default for TagManager {
    fn default() -> Self {
        Self::new()
    }
}

/// معلومات النظام الصناعي
pub const MODULE_VERSION: &str = "1.0.0";
pub const MODULE_NAME: &str = "الأنظمة الصناعية";

/// الحصول على معلومات الوحدة
pub fn module_info() -> String {
    format!(
        r#"
╔═══════════════════════════════════════════════════════════════╗
║              الأنظمة الصناعية - Industrial Systems            ║
╠═══════════════════════════════════════════════════════════════╣
║  الإصدار: {}                                              ║
║  الميزات:                                                     ║
║    ✅ Modbus TCP/RTU حقيقي                                    ║
║    ✅ MQTT للمراقبة عن بعد                                    ║
║    ✅ SCADA Historian                                         ║
║    ✅ HMI رسومية                                              ║
║    ✅ نظام إنذارات متكامل                                     ║
║    ✅ درايفرات أجهزة متعددة                                   ║
╚═══════════════════════════════════════════════════════════════╝
"#,
        MODULE_VERSION
    )
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tag_manager() {
        let manager = TagManager::new();
        
        let tag = Tag {
            name: "Temperature".to_string(),
            description: "درجة حرارة الخزان".to_string(),
            data_type: TagDataType::Float,
            value: TagValue::Float(25.5),
            unit: "°C".to_string(),
            min_value: Some(0.0),
            max_value: Some(100.0),
            low_alarm: Some(10.0),
            high_alarm: Some(80.0),
            last_update: Utc::now(),
            quality: DataQuality::Good,
        };
        
        manager.add_tag(tag);
        
        let retrieved = manager.get_tag("Temperature");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "Temperature");
        
        manager.update_tag("Temperature", TagValue::Float(30.0)).unwrap();
        let updated = manager.get_tag("Temperature").unwrap();
        assert_eq!(updated.value, TagValue::Float(30.0));
    }
    
    #[test]
    fn test_industrial_result() {
        let result: IndustrialResult<i32> = IndustrialResult::ok(42);
        assert!(result.success);
        assert_eq!(result.value, Some(42));
        
        let err_result: IndustrialResult<i32> = IndustrialResult::err("خطأ في الاتصال");
        assert!(!err_result.success);
        assert!(err_result.error.is_some());
    }
}
