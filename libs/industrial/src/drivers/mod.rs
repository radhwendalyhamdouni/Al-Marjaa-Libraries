// ═══════════════════════════════════════════════════════════════════════════════
// Drivers - تعريفات الأجهزة الصناعية
// ═══════════════════════════════════════════════════════════════════════════════

//! # Drivers - تعريفات الأجهزة
//! 
//! تعريفات لأجهزة PLC وأجهزة القياس وأجهزة التحكم.
//! 
//! ## الميزات
//! - تعريفات جاهزة لأجهزة شائعة
//! - اكتشاف تلقائي للأجهزة
//! - تحويل الوحدات
//! - معايرة القيم

use std::collections::HashMap;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
// الأنواع الأساسية
// ═══════════════════════════════════════════════════════════════════════════════

/// نوع الجهاز
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceType {
    /// PLC - Programmable Logic Controller
    PLC,
    /// VFD - Variable Frequency Drive
    VFD,
    /// Sensor - حساس
    Sensor,
    /// Actuator - مشغل
    Actuator,
    /// HMI - Human Machine Interface
    HMI,
    /// Gateway - بوابة اتصال
    Gateway,
    /// RFID Reader
    RFID,
    /// Barcode Scanner
    BarcodeScanner,
    /// Vision System
    VisionSystem,
    /// Robot
    Robot,
    /// CNC Machine
    CNC,
    /// Energy Meter
    EnergyMeter,
    /// Custom
    Custom(String),
}

/// حالة الجهاز
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceStatus {
    /// غير متصل
    Offline,
    /// متصل
    Online,
    /// خطأ
    Error,
    /// تحذير
    Warning,
    /// صيانة
    Maintenance,
    /// غير معروف
    Unknown,
}

/// بروتوكول الاتصال
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Protocol {
    ModbusTCP,
    ModbusRTU,
    EthernetIP,
    PROFINET,
    OPCUA,
    MQTT,
    HTTP,
    Serial,
    CANopen,
    Profibus,
    DeviceNet,
    CCLink,
    BACnet,
    Custom(String),
}

// ═══════════════════════════════════════════════════════════════════════════════
// تعريف الجهاز
// ═══════════════════════════════════════════════════════════════════════════════

/// تعريف جهاز
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceDefinition {
    /// معرف الجهاز
    pub id: String,
    /// اسم الجهاز
    pub name: String,
    /// نوع الجهاز
    pub device_type: DeviceType,
    /// المصنع
    pub manufacturer: String,
    /// الموديل
    pub model: String,
    /// البروتوكول
    pub protocol: Protocol,
    /// عنوان IP (إن وجد)
    pub ip_address: Option<String>,
    /// المنفذ
    pub port: Option<u16>,
    /// Slave ID
    pub slave_id: Option<u8>,
    /// المعايير المعرّفة
    pub tags: HashMap<String, TagDefinition>,
    /// تكوين إضافي
    pub config: HashMap<String, String>,
}

impl DeviceDefinition {
    /// إنشاء جهاز جديد
    pub fn new(name: impl Into<String>, device_type: DeviceType) -> Self {
        Self {
            id: format!("device_{}", uuid::Uuid::new_v4()),
            name: name.into(),
            device_type,
            manufacturer: String::new(),
            model: String::new(),
            protocol: Protocol::ModbusTCP,
            ip_address: None,
            port: None,
            slave_id: None,
            tags: HashMap::new(),
            config: HashMap::new(),
        }
    }
    
    /// تعيين المصنع والموديل
    pub fn with_manufacturer(mut self, manufacturer: impl Into<String>, model: impl Into<String>) -> Self {
        self.manufacturer = manufacturer.into();
        self.model = model.into();
        self
    }
    
    /// تعيين البروتوكول
    pub fn with_protocol(mut self, protocol: Protocol) -> Self {
        self.protocol = protocol;
        self
    }
    
    /// تعيين العنوان
    pub fn with_address(mut self, ip: impl Into<String>, port: u16) -> Self {
        self.ip_address = Some(ip.into());
        self.port = Some(port);
        self
    }
    
    /// تعيين Slave ID
    pub fn with_slave_id(mut self, id: u8) -> Self {
        self.slave_id = Some(id);
        self
    }
    
    /// إضافة Tag
    pub fn add_tag(&mut self, tag: TagDefinition) {
        self.tags.insert(tag.name.clone(), tag);
    }
}

/// تعريف Tag
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagDefinition {
    /// اسم Tag
    pub name: String,
    /// الوصف
    pub description: String,
    /// نوع البيانات
    pub data_type: TagDataType,
    /// العنوان
    pub address: String,
    /// الوحدة
    pub unit: String,
    /// معامل التحويل (scale)
    pub scale: f64,
    /// الإزاحة (offset)
    pub offset: f64,
    /// للقراءة فقط
    pub read_only: bool,
    /// فترة التحديث (ms)
    pub scan_rate: u32,
}

/// نوع بيانات Tag
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TagDataType {
    Bool,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
    Float32,
    Float64,
    String,
    ByteArray,
}

impl TagDefinition {
    /// إنشاء Tag جديد
    pub fn new(name: impl Into<String>, address: impl Into<String>, data_type: TagDataType) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            data_type,
            address: address.into(),
            unit: String::new(),
            scale: 1.0,
            offset: 0.0,
            read_only: false,
            scan_rate: 1000,
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
    
    /// تعيين التحويل
    pub fn with_scaling(mut self, scale: f64, offset: f64) -> Self {
        self.scale = scale;
        self.offset = offset;
        self
    }
    
    /// تحويل القيمة الخام
    pub fn scale_value(&self, raw: f64) -> f64 {
        raw * self.scale + self.offset
    }
    
    /// تحويل القيمة للكتابة
    pub fn unscale_value(&self, value: f64) -> f64 {
        (value - self.offset) / self.scale
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// الأجهزة المعرفة مسبقاً
// ═══════════════════════════════════════════════════════════════════════════════

/// Siemens S7-1200 PLC
pub fn siemens_s7_1200(name: impl Into<String>, ip: impl Into<String>) -> DeviceDefinition {
    let mut device = DeviceDefinition::new(name, DeviceType::PLC)
        .with_manufacturer("Siemens", "S7-1200")
        .with_protocol(Protocol::ModbusTCP)
        .with_address(ip, 502);
    
    // إضافة Tags شائعة
    device.add_tag(
        TagDefinition::new("Motor_Status", "M0.0", TagDataType::Bool)
            .with_description("حالة المحرك")
            .with_unit("On/Off")
    );
    
    device.add_tag(
        TagDefinition::new("Speed", "MW2", TagDataType::UInt16)
            .with_description("سرعة المحرك")
            .with_unit("RPM")
            .with_scaling(1.0, 0.0)
    );
    
    device.add_tag(
        TagDefinition::new("Temperature", "MW4", TagDataType::Int16)
            .with_description("درجة الحرارة")
            .with_unit("°C")
            .with_scaling(0.1, 0.0)
    );
    
    device.add_tag(
        TagDefinition::new("Pressure", "MD6", TagDataType::Float32)
            .with_description("الضغط")
            .with_unit("bar")
    );
    
    device
}

/// Allen-Bradley MicroLogix PLC
pub fn allen_bradley_micrologix(name: impl Into<String>, ip: impl Into<String>) -> DeviceDefinition {
    let mut device = DeviceDefinition::new(name, DeviceType::PLC)
        .with_manufacturer("Allen-Bradley", "MicroLogix 1400")
        .with_protocol(Protocol::ModbusTCP)
        .with_address(ip, 502);
    
    device.add_tag(
        TagDefinition::new("Run_Light", "O:0/0", TagDataType::Bool)
            .with_description("ضوء التشغيل")
    );
    
    device.add_tag(
        TagDefinition::new("Counter", "N7:0", TagDataType::Int16)
            .with_description("العداد")
    );
    
    device
}

/// حساس درجة الحرارة PT100
pub fn pt100_sensor(name: impl Into<String>, slave_id: u8) -> DeviceDefinition {
    let mut device = DeviceDefinition::new(name, DeviceType::Sensor)
        .with_manufacturer("Generic", "PT100 RTD")
        .with_protocol(Protocol::ModbusRTU)
        .with_slave_id(slave_id);
    
    device.add_tag(
        TagDefinition::new("Temperature", "40001", TagDataType::Int16)
            .with_description("درجة الحرارة")
            .with_unit("°C")
            .with_scaling(0.1, 0.0)
    );
    
    device
}

/// محول تردد Siemens Sinamics
pub fn siemens_sinamics_vfd(name: impl Into<String>, ip: impl Into<String>) -> DeviceDefinition {
    let mut device = DeviceDefinition::new(name, DeviceType::VFD)
        .with_manufacturer("Siemens", "Sinamics G120")
        .with_protocol(Protocol::ModbusTCP)
        .with_address(ip, 502);
    
    // Control word
    device.add_tag(
        TagDefinition::new("Control_Word", "40100", TagDataType::UInt16)
            .with_description("كلمة التحكم")
            .with_unit("")
    );
    
    // Status word
    device.add_tag(
        TagDefinition::new("Status_Word", "40110", TagDataType::UInt16)
            .with_description("كلمة الحالة")
            .with_unit("")
    );
    
    // Speed setpoint
    device.add_tag(
        TagDefinition::new("Speed_Setpoint", "40101", TagDataType::UInt16)
            .with_description("السرعة المطلوبة")
            .with_unit("%")
    );
    
    // Actual speed
    device.add_tag(
        TagDefinition::new("Actual_Speed", "40111", TagDataType::UInt16)
            .with_description("السرعة الفعلية")
            .with_unit("RPM")
            .with_scaling(1.0, 0.0)
    );
    
    // Current
    device.add_tag(
        TagDefinition::new("Motor_Current", "40112", TagDataType::UInt16)
            .with_description("تيار المحرك")
            .with_unit("A")
            .with_scaling(0.1, 0.0)
    );
    
    device
}

/// عداد الطاقة
pub fn energy_meter(name: impl Into<String>, ip: impl Into<String>) -> DeviceDefinition {
    let mut device = DeviceDefinition::new(name, DeviceType::EnergyMeter)
        .with_manufacturer("Generic", "Energy Meter")
        .with_protocol(Protocol::ModbusTCP)
        .with_address(ip, 502);
    
    device.add_tag(
        TagDefinition::new("Voltage_L1", "40001", TagDataType::Float32)
            .with_description("جهد الطور L1")
            .with_unit("V")
    );
    
    device.add_tag(
        TagDefinition::new("Voltage_L2", "40003", TagDataType::Float32)
            .with_description("جهد الطور L2")
            .with_unit("V")
    );
    
    device.add_tag(
        TagDefinition::new("Voltage_L3", "40005", TagDataType::Float32)
            .with_description("جهد الطور L3")
            .with_unit("V")
    );
    
    device.add_tag(
        TagDefinition::new("Current_L1", "40007", TagDataType::Float32)
            .with_description("تيار الطور L1")
            .with_unit("A")
    );
    
    device.add_tag(
        TagDefinition::new("Current_L2", "40009", TagDataType::Float32)
            .with_description("تيار الطور L2")
            .with_unit("A")
    );
    
    device.add_tag(
        TagDefinition::new("Current_L3", "40011", TagDataType::Float32)
            .with_description("تيار الطور L3")
            .with_unit("A")
    );
    
    device.add_tag(
        TagDefinition::new("Power_Total", "40013", TagDataType::Float32)
            .with_description("القدرة الكلية")
            .with_unit("kW")
    );
    
    device.add_tag(
        TagDefinition::new("Energy_Total", "40015", TagDataType::Float64)
            .with_description("الطاقة الكلية")
            .with_unit("kWh")
    );
    
    device
}

/// Arduino UNO كـ Modbus Slave
pub fn arduino_uno_modbus(name: impl Into<String>, port: impl Into<String>) -> DeviceDefinition {
    let mut device = DeviceDefinition::new(name, DeviceType::PLC)
        .with_manufacturer("Arduino", "UNO")
        .with_protocol(Protocol::ModbusRTU);
    
    device.config.insert("serial_port".to_string(), port.into());
    device.config.insert("baud_rate".to_string(), "9600".to_string());
    
    // Digital inputs
    for i in 0..6 {
        device.add_tag(
            TagDefinition::new(
                format!("Digital_Input_{}", i),
                format!("1000{}", i),
                TagDataType::Bool
            )
            .with_description(format!("الدخل الرقمي {}", i))
            .with_unit("On/Off")
        );
    }
    
    // Digital outputs
    for i in 0..6 {
        device.add_tag(
            TagDefinition::new(
                format!("Digital_Output_{}", i),
                format!("1001{}", i),
                TagDataType::Bool
            )
            .with_description(format!("الخرج الرقمي {}", i))
            .with_unit("On/Off")
        );
    }
    
    // Analog inputs
    for i in 0..6 {
        device.add_tag(
            TagDefinition::new(
                format!("Analog_Input_{}", i),
                format!("3000{}", i),
                TagDataType::UInt16
            )
            .with_description(format!("الدخل التناظري {}", i))
            .with_unit("0-1023")
        );
    }
    
    device
}

// ═══════════════════════════════════════════════════════════════════════════════
// مدير الأجهزة
// ═══════════════════════════════════════════════════════════════════════════════

/// مدير الأجهزة
#[derive(Debug)]
pub struct DeviceManager {
    /// الأجهزة المسجلة
    devices: HashMap<String, DeviceInstance>,
}

/// حالة جهاز
#[derive(Debug)]
pub struct DeviceInstance {
    /// التعريف
    pub definition: DeviceDefinition,
    /// الحالة
    pub status: DeviceStatus,
    /// آخر اتصال
    pub last_communication: Option<DateTime<Utc>>,
    /// عدد أخطاء الاتصال
    pub error_count: u32,
}

impl DeviceManager {
    /// إنشاء مدير جديد
    pub fn new() -> Self {
        Self {
            devices: HashMap::new(),
        }
    }
    
    /// تسجيل جهاز
    pub fn register(&mut self, device: DeviceDefinition) -> String {
        let id = device.id.clone();
        let instance = DeviceInstance {
            definition: device,
            status: DeviceStatus::Unknown,
            last_communication: None,
            error_count: 0,
        };
        self.devices.insert(id.clone(), instance);
        id
    }
    
    /// إزالة جهاز
    pub fn unregister(&mut self, device_id: &str) -> bool {
        self.devices.remove(device_id).is_some()
    }
    
    /// الحصول على جهاز
    pub fn get(&self, device_id: &str) -> Option<&DeviceInstance> {
        self.devices.get(device_id)
    }
    
    /// الحصول على جهاز قابل للتعديل
    pub fn get_mut(&mut self, device_id: &str) -> Option<&mut DeviceInstance> {
        self.devices.get_mut(device_id)
    }
    
    /// قائمة الأجهزة
    pub fn list(&self) -> Vec<&DeviceInstance> {
        self.devices.values().collect()
    }
    
    /// قائمة الأجهزة حسب النوع
    pub fn list_by_type(&self, device_type: &DeviceType) -> Vec<&DeviceInstance> {
        self.devices.values()
            .filter(|d| &d.definition.device_type == device_type)
            .collect()
    }
    
    /// قائمة الأجهزة حسب الحالة
    pub fn list_by_status(&self, status: DeviceStatus) -> Vec<&DeviceInstance> {
        self.devices.values()
            .filter(|d| d.status == status)
            .collect()
    }
    
    /// عدد الأجهزة
    pub fn count(&self) -> usize {
        self.devices.len()
    }
    
    /// عدد الأجهزة المتصلة
    pub fn online_count(&self) -> usize {
        self.devices.values()
            .filter(|d| d.status == DeviceStatus::Online)
            .count()
    }
}

impl Default for DeviceManager {
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
    fn test_device_definition() {
        let device = DeviceDefinition::new("PLC1", DeviceType::PLC)
            .with_manufacturer("Siemens", "S7-1200")
            .with_protocol(Protocol::ModbusTCP)
            .with_address("192.168.1.100", 502);
        
        assert_eq!(device.name, "PLC1");
        assert_eq!(device.manufacturer, "Siemens");
        assert_eq!(device.ip_address, Some("192.168.1.100".to_string()));
        assert_eq!(device.port, Some(502));
    }
    
    #[test]
    fn test_tag_definition() {
        let tag = TagDefinition::new("Temperature", "40001", TagDataType::Int16)
            .with_description("درجة الحرارة")
            .with_unit("°C")
            .with_scaling(0.1, 0.0);
        
        assert_eq!(tag.name, "Temperature");
        assert_eq!(tag.scale, 0.1);
        
        // اختبار التحويل
        let raw = 250;
        let scaled = tag.scale_value(raw as f64);
        assert_eq!(scaled, 25.0);
        
        let unscaled = tag.unscale_value(25.0);
        assert_eq!(unscaled, 250.0);
    }
    
    #[test]
    fn test_siemens_s7_1200() {
        let device = siemens_s7_1200("PLC_Main", "192.168.1.10");
        
        assert_eq!(device.device_type, DeviceType::PLC);
        assert_eq!(device.manufacturer, "Siemens");
        assert!(device.tags.contains_key("Motor_Status"));
        assert!(device.tags.contains_key("Temperature"));
    }
    
    #[test]
    fn test_energy_meter() {
        let device = energy_meter("Meter1", "192.168.1.20");
        
        assert_eq!(device.device_type, DeviceType::EnergyMeter);
        assert!(device.tags.contains_key("Voltage_L1"));
        assert!(device.tags.contains_key("Power_Total"));
        assert!(device.tags.contains_key("Energy_Total"));
    }
    
    #[test]
    fn test_device_manager() {
        let mut manager = DeviceManager::new();
        
        let device = DeviceDefinition::new("TestDevice", DeviceType::PLC);
        let id = manager.register(device);
        
        assert_eq!(manager.count(), 1);
        assert!(manager.get(&id).is_some());
        
        manager.unregister(&id);
        assert_eq!(manager.count(), 0);
    }
    
    #[test]
    fn test_arduino_uno_modbus() {
        let device = arduino_uno_modbus("Arduino1", "/dev/ttyUSB0");
        
        assert_eq!(device.device_type, DeviceType::PLC);
        assert_eq!(device.protocol, Protocol::ModbusRTU);
        assert!(device.tags.contains_key("Digital_Input_0"));
        assert!(device.tags.contains_key("Analog_Input_0"));
    }
}
