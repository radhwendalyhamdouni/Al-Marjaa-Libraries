//! مكتبة database للغة المرجع
//! Al-Marjaa Language database Library
//!
//! المخترع والمطور: رضوان دالي حمدوني

pub const LIBRARY_NAME: &str = "almarjaa-database";
pub const LIBRARY_VERSION: &str = "3.4.0";

#[no_mangle]
pub extern "C" fn almarjaa_database_init() -> bool {
    true
}
