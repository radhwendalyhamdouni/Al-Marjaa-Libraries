// ═══════════════════════════════════════════════════════════════════════════════
// نظام الإنذارات - Alarms System
// ═══════════════════════════════════════════════════════════════════════════════

//! # نظام الإنذارات
//! 
//! نظام متكامل للإنذارات والتنبيهات الصناعية.
//! 
//! ## الميزات
//! - إنذارات عالية/منخفضة
//! - مستويات شدة متعددة
//! - تأخير التنشيط
//! - تنبيهات متعددة (SMS, Email, Push)
//! - تأكيد الإنذارات
//! - سجل الإنذارات

use std::collections::HashMap;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
// الأنواع الأساسية
// ═══════════════════════════════════════════════════════════════════════════════

/// شدة الإنذار
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlarmSeverity {
    /// معلومات
    Info = 0,
    /// تحذير
    Warning = 1,
    /// إنذار
    Alarm = 2,
    /// حرج
    Critical = 3,
    /// طوارئ
    Emergency = 4,
}

/// نوع الإنذار
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlarmType {
    /// إنذار أعلى
    High,
    /// إنذار أدنى
    Low,
    /// إنذار أعلى-أعلى
    HighHigh,
    /// إنذار أدنى-أدنى
    LowLow,
    /// تغيير معدل
    RateOfChange,
    /// انحراف
    Deviation,
    /// فشل اتصال
    CommunicationFailure,
    /// مخصص
    Custom,
}

/// حالة الإنذار
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlarmState {
    /// غير نشط
    Inactive,
    /// نشط (جديد)
    Active,
    /// مؤكد
    Acknowledged,
    /// ملغى
    Shelved,
    /// مكتمل
    Cleared,
}

// ═══════════════════════════════════════════════════════════════════════════════
// تعريف الإنذار
// ═══════════════════════════════════════════════════════════════════════════════

/// تعريف إنذار
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlarmDefinition {
    /// معرف الإنذار
    pub id: String,
    /// اسم الإنذار
    pub name: String,
    /// الوصف
    pub description: String,
    /// مصدر القيمة (Tag name)
    pub source: String,
    /// نوع الإنذار
    pub alarm_type: AlarmType,
    /// شدة الإنذار
    pub severity: AlarmSeverity,
    /// القيمة الحدية
    pub setpoint: f64,
    /// التأخير بالثواني
    pub delay_seconds: u32,
    /// منطقة التخلف (hysteresis)
    pub deadband: f64,
    /// الرسالة عند التنشيط
    pub on_message: String,
    /// الرسالة عند الإلغاء
    pub off_message: String,
    /// مفعّل
    pub enabled: bool,
    /// التنبيهات المفعلة
    pub notifications: NotificationConfig,
}

impl AlarmDefinition {
    /// إنشاء إنذار أعلى
    pub fn high(name: impl Into<String>, source: impl Into<String>, setpoint: f64) -> Self {
        Self {
            id: format!("alarm_{}", uuid::Uuid::new_v4()),
            name: name.into(),
            description: String::new(),
            source: source.into(),
            alarm_type: AlarmType::High,
            severity: AlarmSeverity::Alarm,
            setpoint,
            delay_seconds: 0,
            deadband: 0.0,
            on_message: "إنذار: القيمة أعلى من الحد".to_string(),
            off_message: "تم إلغاء الإنذار".to_string(),
            enabled: true,
            notifications: NotificationConfig::default(),
        }
    }
    
    /// إنشاء إنذار أدنى
    pub fn low(name: impl Into<String>, source: impl Into<String>, setpoint: f64) -> Self {
        Self {
            id: format!("alarm_{}", uuid::Uuid::new_v4()),
            name: name.into(),
            description: String::new(),
            source: source.into(),
            alarm_type: AlarmType::Low,
            severity: AlarmSeverity::Alarm,
            setpoint,
            delay_seconds: 0,
            deadband: 0.0,
            on_message: "إنذار: القيمة أدنى من الحد".to_string(),
            off_message: "تم إلغاء الإنذار".to_string(),
            enabled: true,
            notifications: NotificationConfig::default(),
        }
    }
    
    /// تعيين الشدة
    pub fn with_severity(mut self, severity: AlarmSeverity) -> Self {
        self.severity = severity;
        self
    }
    
    /// تعيين التأخير
    pub fn with_delay(mut self, seconds: u32) -> Self {
        self.delay_seconds = seconds;
        self
    }
    
    /// تعيين منطقة التخلف
    pub fn with_deadband(mut self, value: f64) -> Self {
        self.deadband = value;
        self
    }
    
    /// تعيين الرسائل
    pub fn with_messages(mut self, on_msg: impl Into<String>, off_msg: impl Into<String>) -> Self {
        self.on_message = on_msg.into();
        self.off_message = off_msg.into();
        self
    }
}

/// تكوين التنبيهات
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    /// SMS
    pub sms_enabled: bool,
    /// SMS المستلمين
    pub sms_recipients: Vec<String>,
    /// Email
    pub email_enabled: bool,
    /// Email المستلمين
    pub email_recipients: Vec<String>,
    /// Push notification
    pub push_enabled: bool,
    /// صوتي
    pub sound_enabled: bool,
}

impl Default for NotificationConfig {
    fn default() -> Self {
        Self {
            sms_enabled: false,
            sms_recipients: Vec::new(),
            email_enabled: false,
            email_recipients: Vec::new(),
            push_enabled: true,
            sound_enabled: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// حالة الإنذار النشطة
// ═══════════════════════════════════════════════════════════════════════════════

/// حالة إنذار نشطة
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlarmInstance {
    /// معرف الحالة
    pub instance_id: String,
    /// معرف التعريف
    pub definition_id: String,
    /// اسم الإنذار
    pub name: String,
    /// الحالة
    pub state: AlarmState,
    /// الشدة
    pub severity: AlarmSeverity,
    /// القيمة عند التنشيط
    pub value: f64,
    /// وقت التنشيط
    pub activated_at: DateTime<Utc>,
    /// وقت التأكيد
    pub acknowledged_at: Option<DateTime<Utc>>,
    /// المستخدم الذي أكد
    pub acknowledged_by: Option<String>,
    /// وقت الإلغاء
    pub cleared_at: Option<DateTime<Utc>>,
    /// الرسالة
    pub message: String,
    /// عدد مرات التنشيط
    pub activation_count: u32,
}

impl AlarmInstance {
    /// مدة الإنذار
    pub fn duration(&self) -> Duration {
        let end = self.cleared_at.unwrap_or_else(Utc::now);
        end - self.activated_at
    }
    
    /// هل الإنذار نشط
    pub fn is_active(&self) -> bool {
        matches!(self.state, AlarmState::Active)
    }
    
    /// هل يتطلب تأكيد
    pub fn needs_acknowledgment(&self) -> bool {
        matches!(self.state, AlarmState::Active)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// مدير الإنذارات
// ═══════════════════════════════════════════════════════════════════════════════

/// مدير الإنذارات
#[derive(Debug)]
pub struct AlarmManager {
    /// تعريفات الإنذارات
    definitions: HashMap<String, AlarmDefinition>,
    /// الإنذارات النشطة
    active_alarms: HashMap<String, AlarmInstance>,
    /// سجل الإنذارات
    history: Vec<AlarmInstance>,
    /// عداد الإنذارات
    counter: u64,
    /// آخر قيم للـ Tags
    last_values: HashMap<String, f64>,
    /// وقت دخول منطقة الإنذار
    alarm_entry_time: HashMap<String, DateTime<Utc>>,
}

impl AlarmManager {
    /// إنشاء مدير جديد
    pub fn new() -> Self {
        Self {
            definitions: HashMap::new(),
            active_alarms: HashMap::new(),
            history: Vec::new(),
            counter: 0,
            last_values: HashMap::new(),
            alarm_entry_time: HashMap::new(),
        }
    }
    
    /// إضافة تعريف إنذار
    pub fn add_alarm(&mut self, definition: AlarmDefinition) -> String {
        let id = definition.id.clone();
        self.definitions.insert(id.clone(), definition);
        id
    }
    
    /// تحديث قيمة Tag والتحقق من الإنذارات
    pub fn update_value(&mut self, tag_name: &str, value: f64) -> Vec<AlarmInstance> {
        let previous_value = self.last_values.get(tag_name).copied();
        self.last_values.insert(tag_name.to_string(), value);
        
        let mut triggered = Vec::new();
        
        // جمع الإنذارات المرتبطة بهذا Tag
        let alarms_to_check: Vec<(String, AlarmDefinition)> = self.definitions
            .iter()
            .filter(|(_, def)| def.source == tag_name && def.enabled)
            .map(|(id, def)| (id.clone(), def.clone()))
            .collect();
        
        // فحص الإنذارات
        for (def_id, definition) in alarms_to_check {
            let should_alarm = self.check_alarm_condition(&definition, value, previous_value);
            
            // معالجة حالة الإنذار
            if should_alarm {
                if let Some(instance) = self.handle_alarm_trigger(&def_id, &definition, value) {
                    triggered.push(instance);
                }
            } else {
                self.handle_alarm_clear(&def_id, &definition, value);
            }
        }
        
        triggered
    }
    
    /// فحص شرط الإنذار
    fn check_alarm_condition(&self, definition: &AlarmDefinition, value: f64, previous: Option<f64>) -> bool {
        match definition.alarm_type {
            AlarmType::High | AlarmType::HighHigh => {
                let threshold = definition.setpoint + definition.deadband;
                value >= threshold
            }
            AlarmType::Low | AlarmType::LowLow => {
                let threshold = definition.setpoint - definition.deadband;
                value <= threshold
            }
            AlarmType::RateOfChange => {
                if let Some(prev) = previous {
                    let rate = (value - prev).abs();
                    rate >= definition.setpoint
                } else {
                    false
                }
            }
            AlarmType::Deviation => {
                // يحتاج قيمة مرجعية
                false
            }
            AlarmType::CommunicationFailure => false,
            AlarmType::Custom => false,
        }
    }
    
    /// معالجة تنشيط إنذار
    fn handle_alarm_trigger(&mut self, def_id: &str, definition: &AlarmDefinition, value: f64) -> Option<AlarmInstance> {
        // التحقق من وجود إنذار نشط بالفعل
        if self.active_alarms.contains_key(def_id) {
            if let Some(existing) = self.active_alarms.get_mut(def_id) {
                existing.value = value;
                existing.activation_count += 1;
            }
            return None;
        }
        
        // التحقق من التأخير
        if definition.delay_seconds > 0 {
            let entry_time = self.alarm_entry_time.entry(def_id.to_string()).or_insert_with(Utc::now);
            let elapsed = (Utc::now() - *entry_time).num_seconds();
            
            if elapsed < definition.delay_seconds as i64 {
                return None;
            }
        }
        
        self.counter += 1;
        
        let instance = AlarmInstance {
            instance_id: format!("inst_{}", self.counter),
            definition_id: def_id.to_string(),
            name: definition.name.clone(),
            state: AlarmState::Active,
            severity: definition.severity,
            value,
            activated_at: Utc::now(),
            acknowledged_at: None,
            acknowledged_by: None,
            cleared_at: None,
            message: definition.on_message.clone(),
            activation_count: 1,
        };
        
        // إرسال التنبيهات
        self.send_notifications(&definition.notifications, &instance);
        
        self.active_alarms.insert(def_id.to_string(), instance.clone());
        
        Some(instance)
    }
    
    /// معالجة إلغاء إنذار
    fn handle_alarm_clear(&mut self, def_id: &str, definition: &AlarmDefinition, value: f64) {
        // إزالة وقت الدخول
        self.alarm_entry_time.remove(def_id);
        
        if let Some(mut instance) = self.active_alarms.remove(def_id) {
            instance.state = AlarmState::Cleared;
            instance.cleared_at = Some(Utc::now());
            instance.value = value;
            instance.message = definition.off_message.clone();
            
            self.history.push(instance);
        }
    }
    
    /// تأكيد إنذار
    pub fn acknowledge(&mut self, def_id: &str, user: &str) -> bool {
        if let Some(instance) = self.active_alarms.get_mut(def_id) {
            instance.state = AlarmState::Acknowledged;
            instance.acknowledged_at = Some(Utc::now());
            instance.acknowledged_by = Some(user.to_string());
            true
        } else {
            false
        }
    }
    
    /// تأكيد جميع الإنذارات
    pub fn acknowledge_all(&mut self, user: &str) -> usize {
        let count = self.active_alarms.len();
        for instance in self.active_alarms.values_mut() {
            instance.state = AlarmState::Acknowledged;
            instance.acknowledged_at = Some(Utc::now());
            instance.acknowledged_by = Some(user.to_string());
        }
        count
    }
    
    /// تعليق إنذار (shelve)
    pub fn shelve(&mut self, def_id: &str) -> bool {
        if let Some(instance) = self.active_alarms.get_mut(def_id) {
            instance.state = AlarmState::Shelved;
            true
        } else {
            false
        }
    }
    
    /// إرسال التنبيهات
    fn send_notifications(&self, config: &NotificationConfig, instance: &AlarmInstance) {
        if config.sound_enabled {
            tracing::warn!("🔔 إنذار: {} - {}", instance.name, instance.message);
        }
        
        if config.push_enabled {
            // سيتم إرسال Push notification
            tracing::info!("Push notification: {} - {}", instance.name, instance.message);
        }
        
        if config.sms_enabled && !config.sms_recipients.is_empty() {
            // سيتم إرسال SMS
            tracing::info!("SMS to {:?}: {}", config.sms_recipients, instance.message);
        }
        
        if config.email_enabled && !config.email_recipients.is_empty() {
            // سيتم إرسال Email
            tracing::info!("Email to {:?}: {}", config.email_recipients, instance.message);
        }
    }
    
    /// الحصول على الإنذارات النشطة
    pub fn get_active(&self) -> Vec<&AlarmInstance> {
        self.active_alarms.values().collect()
    }
    
    /// الحصول على الإنذارات غير المؤكدة
    pub fn get_unacknowledged(&self) -> Vec<&AlarmInstance> {
        self.active_alarms.values()
            .filter(|a| a.state == AlarmState::Active)
            .collect()
    }
    
    /// الحصول على سجل الإنذارات
    pub fn get_history(&self) -> &[AlarmInstance] {
        &self.history
    }
    
    /// عدد الإنذارات النشطة
    pub fn active_count(&self) -> usize {
        self.active_alarms.len()
    }
    
    /// عدد الإنذارات غير المؤكدة
    pub fn unacknowledged_count(&self) -> usize {
        self.active_alarms.values()
            .filter(|a| a.state == AlarmState::Active)
            .count()
    }
    
    /// الحصول على أعلى شدة
    pub fn highest_severity(&self) -> Option<AlarmSeverity> {
        self.active_alarms.values()
            .map(|a| a.severity)
            .max()
    }
    
    /// مسح السجل
    pub fn clear_history(&mut self) {
        self.history.clear();
    }
}

impl Default for AlarmManager {
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
    fn test_alarm_definition_high() {
        let alarm = AlarmDefinition::high("درجة_الحرارة", "Temperature", 80.0)
            .with_severity(AlarmSeverity::Critical)
            .with_deadband(2.0);
        
        assert_eq!(alarm.alarm_type, AlarmType::High);
        assert_eq!(alarm.setpoint, 80.0);
        assert_eq!(alarm.severity, AlarmSeverity::Critical);
        assert_eq!(alarm.deadband, 2.0);
    }
    
    #[test]
    fn test_alarm_definition_low() {
        let alarm = AlarmDefinition::low("الضغط", "Pressure", 5.0);
        
        assert_eq!(alarm.alarm_type, AlarmType::Low);
        assert_eq!(alarm.setpoint, 5.0);
    }
    
    #[test]
    fn test_alarm_manager() {
        let mut manager = AlarmManager::new();
        
        let alarm = AlarmDefinition::high("درجة_الحرارة", "Temperature", 80.0);
        manager.add_alarm(alarm);
        
        // قيمة طبيعية
        let triggered = manager.update_value("Temperature", 50.0);
        assert!(triggered.is_empty());
        assert_eq!(manager.active_count(), 0);
        
        // قيمة فوق الحد
        let triggered = manager.update_value("Temperature", 85.0);
        assert_eq!(triggered.len(), 1);
        assert_eq!(manager.active_count(), 1);
        
        // تأكيد الإنذار
        let def_id = triggered[0].definition_id.clone();
        manager.acknowledge(&def_id, "operator");
        
        let unacked = manager.get_unacknowledged();
        assert_eq!(unacked.len(), 0);
    }
    
    #[test]
    fn test_alarm_clear() {
        let mut manager = AlarmManager::new();
        
        let alarm = AlarmDefinition::high("درجة_الحرارة", "Temperature", 80.0);
        manager.add_alarm(alarm);
        
        // تنشيط
        manager.update_value("Temperature", 90.0);
        assert_eq!(manager.active_count(), 1);
        
        // إلغاء
        manager.update_value("Temperature", 70.0);
        assert_eq!(manager.active_count(), 0);
        assert_eq!(manager.history.len(), 1);
    }
    
    #[test]
    fn test_alarm_instance_duration() {
        let instance = AlarmInstance {
            instance_id: "inst_1".to_string(),
            definition_id: "alarm_1".to_string(),
            name: "Test".to_string(),
            state: AlarmState::Active,
            severity: AlarmSeverity::Alarm,
            value: 85.0,
            activated_at: Utc::now() - Duration::seconds(30),
            acknowledged_at: None,
            acknowledged_by: None,
            cleared_at: None,
            message: "Test".to_string(),
            activation_count: 1,
        };
        
        let duration = instance.duration();
        assert!(duration.num_seconds() >= 30);
    }
    
    #[test]
    fn test_acknowledge_all() {
        let mut manager = AlarmManager::new();
        
        manager.add_alarm(AlarmDefinition::high("T1", "Tag1", 50.0));
        manager.add_alarm(AlarmDefinition::high("T2", "Tag2", 50.0));
        
        manager.update_value("Tag1", 60.0);
        manager.update_value("Tag2", 70.0);
        
        assert_eq!(manager.active_count(), 2);
        
        let count = manager.acknowledge_all("admin");
        assert_eq!(count, 2);
        assert_eq!(manager.unacknowledged_count(), 0);
    }
    
    #[test]
    fn test_highest_severity() {
        let mut manager = AlarmManager::new();
        
        manager.add_alarm(
            AlarmDefinition::high("T1", "Tag1", 50.0)
                .with_severity(AlarmSeverity::Warning)
        );
        manager.add_alarm(
            AlarmDefinition::high("T2", "Tag2", 50.0)
                .with_severity(AlarmSeverity::Critical)
        );
        
        manager.update_value("Tag1", 60.0);
        manager.update_value("Tag2", 60.0);
        
        let highest = manager.highest_severity();
        assert_eq!(highest, Some(AlarmSeverity::Critical));
    }
}
