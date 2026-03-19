// ═══════════════════════════════════════════════════════════════════════════════
// HMI - واجهة إنسانية-آلية رسومية
// ═══════════════════════════════════════════════════════════════════════════════

//! # HMI - Human Machine Interface
//! 
//! واجهة رسومية للتحكم الصناعي تدعم:
//! - العدادات (Gauges)
//! - الرسوم البيانية (Charts/Trends)
//! - الأزرار والمفاتيح (Buttons/Switches)
//! - المؤشرات (Indicators/LEDs)
//! - شاشات التحكم (Control Panels)
//! 
//! ## مثال
//! 
//! ```rust
//! use almarjaa::industrial::hmi::{HmiApp, Gauge, Chart, Button};
//! 
//! let mut app = HmiApp::new("لوحة التحكم");
//! 
//! // إضافة عداد سرعة
//! app.add_gauge(Gauge::new("السرعة", 0.0, 300.0));
//! 
//! // إضافة رسم بياني
//! app.add_chart(Chart::new("درجة الحرارة"));
//! ```

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
// الأنواع الأساسية
// ═══════════════════════════════════════════════════════════════════════════════

/// لون HMI
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HmiColor {
    Green,
    Yellow,
    Orange,
    Red,
    Blue,
    Cyan,
    Purple,
    Gray,
    White,
    Black,
}

impl HmiColor {
    /// الحصول على كود RGB
    pub fn rgb(&self) -> (u8, u8, u8) {
        match self {
            HmiColor::Green => (0, 200, 0),
            HmiColor::Yellow => (255, 200, 0),
            HmiColor::Orange => (255, 165, 0),
            HmiColor::Red => (220, 0, 0),
            HmiColor::Blue => (0, 100, 200),
            HmiColor::Cyan => (0, 200, 200),
            HmiColor::Purple => (150, 0, 200),
            HmiColor::Gray => (128, 128, 128),
            HmiColor::White => (255, 255, 255),
            HmiColor::Black => (0, 0, 0),
        }
    }
    
    /// الحصول على كود Hex
    pub fn hex(&self) -> String {
        let (r, g, b) = self.rgb();
        format!("#{:02X}{:02X}{:02X}", r, g, b)
    }
}

/// حالة العنصر
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WidgetState {
    Normal,
    Alarm,
    Warning,
    Fault,
    Disabled,
}

// ═══════════════════════════════════════════════════════════════════════════════
// عنصر HMI الأساسي
// ═══════════════════════════════════════════════════════════════════════════════

/// عنصر HMI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HmiWidget {
    /// معرف العنصر
    pub id: String,
    /// اسم العنصر
    pub name: String,
    /// نوع العنصر
    pub widget_type: WidgetType,
    /// الموقع X
    pub x: f32,
    /// الموقع Y
    pub y: f32,
    /// العرض
    pub width: f32,
    /// الارتفاع
    pub height: f32,
    /// الحالة
    pub state: WidgetState,
    /// مرئي
    pub visible: bool,
    /// مفعل
    pub enabled: bool,
    /// بيانات إضافية
    pub data: HashMap<String, String>,
}

/// نوع العنصر
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WidgetType {
    /// عداد (Gauge)
    Gauge,
    /// رسم بياني (Chart/Trend)
    Chart,
    /// زر (Button)
    Button,
    /// مفتاح (Switch)
    Switch,
    /// مؤشر LED
    Indicator,
    /// شريط تقدم
    ProgressBar,
    /// نص
    Text,
    /// حقل إدخال
    Input,
    /// صورة
    Image,
    /// حاوية
    Container,
    /// جدول بيانات
    DataGrid,
    /// شاشة عرض رقمية
    DigitalDisplay,
}

impl HmiWidget {
    /// إنشاء عنصر جديد
    pub fn new(id: impl Into<String>, name: impl Into<String>, widget_type: WidgetType) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            widget_type,
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
            state: WidgetState::Normal,
            visible: true,
            enabled: true,
            data: HashMap::new(),
        }
    }
    
    /// تعيين الموقع
    pub fn with_position(mut self, x: f32, y: f32) -> Self {
        self.x = x;
        self.y = y;
        self
    }
    
    /// تعيين الحجم
    pub fn with_size(mut self, width: f32, height: f32) -> Self {
        self.width = width;
        self.height = height;
        self
    }
    
    /// تعيين بيانات
    pub fn with_data(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.data.insert(key.into(), value.into());
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// عداد (Gauge)
// ═══════════════════════════════════════════════════════════════════════════════

/// عداد قياس
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gauge {
    /// العنصر الأساسي
    pub widget: HmiWidget,
    /// القيمة الحالية
    pub value: f64,
    /// الحد الأدنى
    pub min_value: f64,
    /// الحد الأقصى
    pub max_value: f64,
    /// الوحدة
    pub unit: String,
    /// حد التحذير الأدنى
    pub low_warning: Option<f64>,
    /// حد التحذير الأعلى
    pub high_warning: Option<f64>,
    /// حد الإنذار الأدنى
    pub low_alarm: Option<f64>,
    /// حد الإنذار الأعلى
    pub high_alarm: Option<f64>,
    /// اللون
    pub color: HmiColor,
    /// نوع العداد
    pub gauge_type: GaugeType,
}

/// نوع العداد
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GaugeType {
    /// دائري
    Circular,
    /// نصف دائري
    Semicircular,
    /// خطي
    Linear,
    /// رقمي
    Digital,
}

impl Gauge {
    /// إنشاء عداد جديد
    pub fn new(name: impl Into<String>, min: f64, max: f64) -> Self {
        Self {
            widget: HmiWidget::new(
                format!("gauge_{}", uuid::Uuid::new_v4()),
                name,
                WidgetType::Gauge,
            ),
            value: min,
            min_value: min,
            max_value: max,
            unit: String::new(),
            low_warning: None,
            high_warning: None,
            low_alarm: None,
            high_alarm: None,
            color: HmiColor::Blue,
            gauge_type: GaugeType::Circular,
        }
    }
    
    /// تعيين الوحدة
    pub fn with_unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = unit.into();
        self
    }
    
    /// تعيين حدود التحذير
    pub fn with_warning_limits(mut self, low: f64, high: f64) -> Self {
        self.low_warning = Some(low);
        self.high_warning = Some(high);
        self
    }
    
    /// تعيين حدود الإنذار
    pub fn with_alarm_limits(mut self, low: f64, high: f64) -> Self {
        self.low_alarm = Some(low);
        self.high_alarm = Some(high);
        self
    }
    
    /// تحديث القيمة
    pub fn set_value(&mut self, value: f64) {
        self.value = value.clamp(self.min_value, self.max_value);
        self.update_state();
    }
    
    /// تحديث الحالة بناءً على القيمة
    fn update_state(&mut self) {
        self.widget.state = WidgetState::Normal;
        
        if let Some(high) = self.high_alarm {
            if self.value >= high {
                self.widget.state = WidgetState::Alarm;
                self.color = HmiColor::Red;
                return;
            }
        }
        
        if let Some(low) = self.low_alarm {
            if self.value <= low {
                self.widget.state = WidgetState::Alarm;
                self.color = HmiColor::Red;
                return;
            }
        }
        
        if let Some(high) = self.high_warning {
            if self.value >= high {
                self.widget.state = WidgetState::Warning;
                self.color = HmiColor::Orange;
                return;
            }
        }
        
        if let Some(low) = self.low_warning {
            if self.value <= low {
                self.widget.state = WidgetState::Warning;
                self.color = HmiColor::Yellow;
                return;
            }
        }
    }
    
    /// نسبة القيمة من المدى
    pub fn percentage(&self) -> f64 {
        if self.max_value == self.min_value {
            return 0.0;
        }
        ((self.value - self.min_value) / (self.max_value - self.min_value) * 100.0)
            .clamp(0.0, 100.0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// رسم بياني (Chart)
// ═══════════════════════════════════════════════════════════════════════════════

/// نقطة بيانات
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
}

/// رسم بياني
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chart {
    /// العنصر الأساسي
    pub widget: HmiWidget,
    /// نقاط البيانات
    pub data_points: Vec<DataPoint>,
    /// الحد الأقصى للنقاط
    pub max_points: usize,
    /// اللون
    pub color: HmiColor,
    /// نوع الرسم
    pub chart_type: ChartType,
    /// إظهار الشبكة
    pub show_grid: bool,
    /// حد Y الأدنى
    pub y_min: Option<f64>,
    /// حد Y الأقصى
    pub y_max: Option<f64>,
}

/// نوع الرسم البياني
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChartType {
    Line,
    Area,
    Bar,
    Scatter,
}

impl Chart {
    /// إنشاء رسم بياني جديد
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            widget: HmiWidget::new(
                format!("chart_{}", uuid::Uuid::new_v4()),
                name,
                WidgetType::Chart,
            ),
            data_points: Vec::new(),
            max_points: 1000,
            color: HmiColor::Blue,
            chart_type: ChartType::Line,
            show_grid: true,
            y_min: None,
            y_max: None,
        }
    }
    
    /// إضافة نقطة بيانات
    pub fn add_point(&mut self, value: f64) {
        let point = DataPoint {
            timestamp: Utc::now(),
            value,
        };
        
        self.data_points.push(point);
        
        // إزالة النقاط القديمة
        if self.data_points.len() > self.max_points {
            self.data_points.remove(0);
        }
    }
    
    /// إضافة نقطة مع وقت محدد
    pub fn add_point_with_time(&mut self, value: f64, timestamp: DateTime<Utc>) {
        let point = DataPoint { timestamp, value };
        self.data_points.push(point);
        
        if self.data_points.len() > self.max_points {
            self.data_points.remove(0);
        }
    }
    
    /// مسح البيانات
    pub fn clear(&mut self) {
        self.data_points.clear();
    }
    
    /// آخر قيمة
    pub fn last_value(&self) -> Option<f64> {
        self.data_points.last().map(|p| p.value)
    }
    
    /// متوسط القيم
    pub fn average(&self) -> f64 {
        if self.data_points.is_empty() {
            return 0.0;
        }
        self.data_points.iter().map(|p| p.value).sum::<f64>() / self.data_points.len() as f64
    }
    
    /// أقصى قيمة
    pub fn max(&self) -> f64 {
        self.data_points.iter().map(|p| p.value).fold(f64::MIN, f64::max)
    }
    
    /// أدنى قيمة
    pub fn min(&self) -> f64 {
        self.data_points.iter().map(|p| p.value).fold(f64::MAX, f64::min)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// زر تحكم (Button)
// ═══════════════════════════════════════════════════════════════════════════════

/// زر تحكم
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Button {
    /// العنصر الأساسي
    pub widget: HmiWidget,
    /// النص
    pub text: String,
    /// اللون
    pub color: HmiColor,
    /// نوع الزر
    pub button_type: ButtonType,
    /// مضغوط
    pub pressed: bool,
}

/// نوع الزر
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ButtonType {
    /// زر عادي
    Normal,
    /// زر إيقاف طوارئ
    EmergencyStop,
    /// زر بدء
    Start,
    /// زر إيقاف
    Stop,
    /// زر إعادة تعيين
    Reset,
    /// زر تأكيد
    Confirm,
}

impl Button {
    /// إنشاء زر جديد
    pub fn new(name: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            widget: HmiWidget::new(
                format!("btn_{}", uuid::Uuid::new_v4()),
                name,
                WidgetType::Button,
            ),
            text: text.into(),
            color: HmiColor::Blue,
            button_type: ButtonType::Normal,
            pressed: false,
        }
    }
    
    /// إنشاء زر إيقاف طوارئ
    pub fn emergency_stop() -> Self {
        let mut btn = Self::new("emergency_stop", "إيقاف طوارئ");
        btn.button_type = ButtonType::EmergencyStop;
        btn.color = HmiColor::Red;
        btn
    }
    
    /// إنشاء زر بدء
    pub fn start() -> Self {
        let mut btn = Self::new("start", "بدء");
        btn.button_type = ButtonType::Start;
        btn.color = HmiColor::Green;
        btn
    }
    
    /// إنشاء زر إيقاف
    pub fn stop() -> Self {
        let mut btn = Self::new("stop", "إيقاف");
        btn.button_type = ButtonType::Stop;
        btn.color = HmiColor::Red;
        btn
    }
    
    /// ضغط الزر
    pub fn press(&mut self) {
        self.pressed = true;
    }
    
    /// تحرير الزر
    pub fn release(&mut self) {
        self.pressed = false;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// مؤشر LED (Indicator)
// ═══════════════════════════════════════════════════════════════════════════════

/// مؤشر LED
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Indicator {
    /// العنصر الأساسي
    pub widget: HmiWidget,
    /// الحالة
    pub is_on: bool,
    /// لون التشغيل
    pub on_color: HmiColor,
    /// لون الإيقاف
    pub off_color: HmiColor,
    /// الوميض
    pub blinking: bool,
    /// معدل الوميض (ms)
    pub blink_rate: u32,
}

impl Indicator {
    /// إنشاء مؤشر جديد
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            widget: HmiWidget::new(
                format!("ind_{}", uuid::Uuid::new_v4()),
                name,
                WidgetType::Indicator,
            ),
            is_on: false,
            on_color: HmiColor::Green,
            off_color: HmiColor::Gray,
            blinking: false,
            blink_rate: 500,
        }
    }
    
    /// تشغيل
    pub fn turn_on(&mut self) {
        self.is_on = true;
    }
    
    /// إيقاف
    pub fn turn_off(&mut self) {
        self.is_on = false;
        self.blinking = false;
    }
    
    /// تبديل الحالة
    pub fn toggle(&mut self) {
        self.is_on = !self.is_on;
    }
    
    /// بدء الوميض
    pub fn start_blinking(&mut self) {
        self.blinking = true;
    }
    
    /// إيقاف الوميض
    pub fn stop_blinking(&mut self) {
        self.blinking = false;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// تطبيق HMI
// ═══════════════════════════════════════════════════════════════════════════════

/// تطبيق HMI متكامل
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HmiApp {
    /// اسم التطبيق
    pub name: String,
    /// العناصر
    pub widgets: HashMap<String, HmiWidget>,
    /// العدادات
    pub gauges: HashMap<String, Gauge>,
    /// الرسوم البيانية
    pub charts: HashMap<String, Chart>,
    /// الأزرار
    pub buttons: HashMap<String, Button>,
    /// المؤشرات
    pub indicators: HashMap<String, Indicator>,
    /// العرض
    pub width: u32,
    /// الارتفاع
    pub height: u32,
    /// الثيم
    pub theme: HmiTheme,
}

/// ثيم HMI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HmiTheme {
    pub background_color: HmiColor,
    pub text_color: HmiColor,
    pub accent_color: HmiColor,
    pub font_family: String,
    pub font_size: u32,
}

impl Default for HmiTheme {
    fn default() -> Self {
        Self {
            background_color: HmiColor::Black,
            text_color: HmiColor::White,
            accent_color: HmiColor::Blue,
            font_family: "Arial".to_string(),
            font_size: 14,
        }
    }
}

impl HmiApp {
    /// إنشاء تطبيق جديد
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            widgets: HashMap::new(),
            gauges: HashMap::new(),
            charts: HashMap::new(),
            buttons: HashMap::new(),
            indicators: HashMap::new(),
            width: 1280,
            height: 720,
            theme: HmiTheme::default(),
        }
    }
    
    /// إضافة عداد
    pub fn add_gauge(&mut self, gauge: Gauge) -> String {
        let id = gauge.widget.id.clone();
        self.gauges.insert(id.clone(), gauge);
        id
    }
    
    /// إضافة رسم بياني
    pub fn add_chart(&mut self, chart: Chart) -> String {
        let id = chart.widget.id.clone();
        self.charts.insert(id.clone(), chart);
        id
    }
    
    /// إضافة زر
    pub fn add_button(&mut self, button: Button) -> String {
        let id = button.widget.id.clone();
        self.buttons.insert(id.clone(), button);
        id
    }
    
    /// إضافة مؤشر
    pub fn add_indicator(&mut self, indicator: Indicator) -> String {
        let id = indicator.widget.id.clone();
        self.indicators.insert(id.clone(), indicator);
        id
    }
    
    /// تحديث قيمة عداد
    pub fn update_gauge(&mut self, id: &str, value: f64) -> Result<(), String> {
        self.gauges.get_mut(id)
            .ok_or_else(|| format!("العداد '{}' غير موجود", id))
            .map(|g| g.set_value(value))
    }
    
    /// إضافة نقطة للرسم البياني
    pub fn add_chart_point(&mut self, id: &str, value: f64) -> Result<(), String> {
        self.charts.get_mut(id)
            .ok_or_else(|| format!("الرسم البياني '{}' غير موجود", id))
            .map(|c| c.add_point(value))
    }
    
    /// الحصول على قيمة عداد
    pub fn get_gauge_value(&self, id: &str) -> Option<f64> {
        self.gauges.get(id).map(|g| g.value)
    }
    
    /// تصدير كـ JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
    
    /// إنشاء HTML للعرض
    pub fn render_html(&self) -> String {
        let mut html = format!(
            r#"<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <title>{}</title>
    <style>
        body {{ 
            background-color: {}; 
            color: {}; 
            font-family: {}, sans-serif;
            font-size: {}px;
            margin: 0;
            padding: 20px;
        }}
        .hmi-container {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 20px;
        }}
        .gauge {{ 
            background: #1a1a1a; 
            border-radius: 10px; 
            padding: 20px;
            text-align: center;
        }}
        .gauge-value {{ 
            font-size: 2em; 
            font-weight: bold;
        }}
        .chart {{ 
            background: #1a1a1a; 
            border-radius: 10px; 
            padding: 20px;
        }}
        .button {{ 
            padding: 15px 30px; 
            border-radius: 8px;
            border: none;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
        }}
        .indicator {{ 
            width: 30px; 
            height: 30px; 
            border-radius: 50%;
            display: inline-block;
        }}
    </style>
</head>
<body>
    <h1>{}</h1>
    <div class="hmi-container">
"#,
            self.name,
            self.theme.background_color.hex(),
            self.theme.text_color.hex(),
            self.theme.font_family,
            self.theme.font_size,
            self.name
        );
        
        // إضافة العدادات
        for (id, gauge) in &self.gauges {
            html.push_str(&format!(
                r#"        <div class="gauge" id="{}">
            <div class="gauge-name">{}</div>
            <div class="gauge-value" style="color: {}">{:.1}</div>
            <div class="gauge-unit">{}</div>
        </div>
"#,
                id, gauge.widget.name, gauge.color.hex(), gauge.value, gauge.unit
            ));
        }
        
        html.push_str("    </div>\n</body>\n</html>");
        html
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gauge_creation() {
        let gauge = Gauge::new("السرعة", 0.0, 300.0)
            .with_unit("RPM")
            .with_warning_limits(200.0, 280.0)
            .with_alarm_limits(290.0, 295.0);
        
        assert_eq!(gauge.widget.name, "السرعة");
        assert_eq!(gauge.min_value, 0.0);
        assert_eq!(gauge.max_value, 300.0);
        assert_eq!(gauge.unit, "RPM");
    }
    
    #[test]
    fn test_gauge_value_update() {
        let mut gauge = Gauge::new("السرعة", 0.0, 100.0)
            .with_alarm_limits(10.0, 90.0);
        
        gauge.set_value(50.0);
        assert_eq!(gauge.value, 50.0);
        assert_eq!(gauge.widget.state, WidgetState::Normal);
        
        gauge.set_value(95.0);
        assert_eq!(gauge.widget.state, WidgetState::Alarm);
    }
    
    #[test]
    fn test_chart() {
        let mut chart = Chart::new("درجة الحرارة");
        
        chart.add_point(25.0);
        chart.add_point(26.0);
        chart.add_point(27.0);
        
        assert_eq!(chart.data_points.len(), 3);
        assert_eq!(chart.last_value(), Some(27.0));
        
        let avg = chart.average();
        assert!((avg - 26.0).abs() < 0.01);
    }
    
    #[test]
    fn test_button() {
        let btn = Button::new("start_btn", "بدء التشغيل");
        assert_eq!(btn.text, "بدء التشغيل");
        assert!(!btn.pressed);
    }
    
    #[test]
    fn test_emergency_stop_button() {
        let btn = Button::emergency_stop();
        assert_eq!(btn.button_type, ButtonType::EmergencyStop);
        assert_eq!(btn.color, HmiColor::Red);
    }
    
    #[test]
    fn test_indicator() {
        let mut ind = Indicator::new("status_led");
        
        assert!(!ind.is_on);
        
        ind.turn_on();
        assert!(ind.is_on);
        
        ind.toggle();
        assert!(!ind.is_on);
    }
    
    #[test]
    fn test_hmi_app() {
        let mut app = HmiApp::new("لوحة التحكم الرئيسية");
        
        let gauge = Gauge::new("السرعة", 0.0, 300.0);
        let gauge_id = app.add_gauge(gauge);
        
        app.update_gauge(&gauge_id, 150.0).unwrap();
        
        let value = app.get_gauge_value(&gauge_id);
        assert_eq!(value, Some(150.0));
    }
    
    #[test]
    fn test_hmi_render_html() {
        let mut app = HmiApp::new("Test App");
        
        let gauge = Gauge::new("Temperature", 0.0, 100.0)
            .with_unit("°C");
        app.add_gauge(gauge);
        
        let html = app.render_html();
        assert!(html.contains("Test App"));
        assert!(html.contains("Temperature"));
    }
    
    #[test]
    fn test_hmi_color() {
        let color = HmiColor::Red;
        assert_eq!(color.rgb(), (220, 0, 0));
        assert_eq!(color.hex(), "#DC0000");
    }
}
