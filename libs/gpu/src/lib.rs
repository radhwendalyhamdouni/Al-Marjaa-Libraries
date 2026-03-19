//! مكتبة gpu للغة المرجع
//! Al-Marjaa Language gpu Library
//!
//! المخترع والمطور: رضوان دالي حمدوني

pub const LIBRARY_NAME: &str = "almarjaa-gpu";
pub const LIBRARY_VERSION: &str = "3.4.0";

#[no_mangle]
pub extern "C" fn almarjaa_gpu_init() -> bool {
    true
}
