//! مكتبة onnx للغة المرجع
//! Al-Marjaa Language onnx Library
//!
//! المخترع والمطور: رضوان دالي حمدوني

pub const LIBRARY_NAME: &str = "almarjaa-onnx";
pub const LIBRARY_VERSION: &str = "3.4.0";

#[no_mangle]
pub extern "C" fn almarjaa_onnx_init() -> bool {
    true
}
