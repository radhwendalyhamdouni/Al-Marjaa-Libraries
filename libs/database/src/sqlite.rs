// src/stdlib/database/sqlite.rs
// SQLite Driver - Production Ready with rusqlite
// مشغل SQLite - جاهز للإنتاج مع rusqlite

use super::{QueryResult, Row};
use std::collections::HashMap;

/// اتصال SQLite الحقيقي (مع feature database)
#[cfg(feature = "database")]
pub struct SqliteConnection {
    /// الاتصال الحقيقي
    connection: Arc<Mutex<rusqlite::Connection>>,
    /// مسار قاعدة البيانات
    database_path: String,
    /// هل الاتصال مفتوح
    connected: bool,
}

/// اتصال SQLite المحاكاة (بدون feature database)
#[cfg(not(feature = "database"))]
pub struct SqliteConnection {
    /// المسار
    database_path: String,
    /// الجداول المحاكاة
    tables: HashMap<String, Vec<(String, String)>>,
    /// البيانات المحاكاة
    data: HashMap<String, Vec<Row>>,
    /// هل متصل
    connected: bool,
}

impl SqliteConnection {
    /// إنشاء اتصال جديد
    pub fn new(path: &str) -> Self {
        #[cfg(feature = "database")]
        {
            Self::new_real(path)
        }

        #[cfg(not(feature = "database"))]
        {
            Self::new_simulation(path)
        }
    }

    /// إنشاء اتصال حقيقي (مع feature database)
    #[cfg(feature = "database")]
    fn new_real(path: &str) -> Self {
        let conn_result = if path == ":memory:" {
            rusqlite::Connection::open_in_memory()
        } else {
            rusqlite::Connection::open(path)
        };

        match conn_result {
            Ok(conn) => {
                println!(
                    "✅ [PRODUCTION] تم فتح قاعدة بيانات SQLite الحقيقية: {}",
                    path
                );
                Self {
                    connection: Arc::new(Mutex::new(conn)),
                    database_path: path.to_string(),
                    connected: true,
                }
            }
            Err(e) => {
                eprintln!(
                    "⚠️ [FALLBACK] فشل فتح قاعدة البيانات، استخدام المحاكاة: {}",
                    e
                );
                // Fallback للمحاكاة
                Self {
                    connection: Arc::new(Mutex::new(
                        rusqlite::Connection::open_in_memory().unwrap(),
                    )),
                    database_path: path.to_string(),
                    connected: false,
                }
            }
        }
    }

    /// إنشاء اتصال محاكاة (بدون feature database)
    #[cfg(not(feature = "database"))]
    fn new_simulation(path: &str) -> Self {
        println!("⚠️ [SIMULATION] وضع المحاكاة - rusqlite غير مفعّل");
        println!("📝 لتفعيل قاعدة البيانات الحقيقية، استخدم: --features database");
        Self {
            database_path: path.to_string(),
            tables: HashMap::new(),
            data: HashMap::new(),
            connected: true,
        }
    }

    /// الاتصال بقاعدة البيانات
    pub fn connect(&mut self) -> Result<(), String> {
        #[cfg(feature = "database")]
        {
            if self.connected {
                return Ok(());
            }

            let conn_result = if self.database_path == ":memory:" {
                rusqlite::Connection::open_in_memory()
            } else {
                rusqlite::Connection::open(&self.database_path)
            };

            match conn_result {
                Ok(conn) => {
                    self.connection = Arc::new(Mutex::new(conn));
                    self.connected = true;
                    println!("✅ [PRODUCTION] تم الاتصال بقاعدة البيانات");
                    Ok(())
                }
                Err(e) => Err(format!("فشل الاتصال بقاعدة البيانات: {}", e)),
            }
        }

        #[cfg(not(feature = "database"))]
        {
            self.connected = true;
            println!("⚠️ [SIMULATION] محاكاة الاتصال");
            Ok(())
        }
    }

    /// إنشاء جدول
    pub fn create_table(&mut self, name: &str, columns: &[(&str, &str)]) -> Result<(), String> {
        #[cfg(feature = "database")]
        {
            let cols_def: String = columns
                .iter()
                .map(|(col_name, col_type)| {
                    let sql_type = match col_type.to_lowercase().as_str() {
                        "integer" | "int" | "عدد_صحيح" => "INTEGER",
                        "real" | "float" | "عدد_عشري" => "REAL",
                        "text" | "string" | "نص" => "TEXT",
                        "blob" | "binary" | "ثنائي" => "BLOB",
                        "boolean" | "bool" | "منطقي" => "INTEGER",
                        _ => "TEXT",
                    };
                    format!("{} {}", col_name, sql_type)
                })
                .collect::<Vec<_>>()
                .join(", ");

            let sql = format!(
                "CREATE TABLE IF NOT EXISTS {} (id INTEGER PRIMARY KEY AUTOINCREMENT, {})",
                name, cols_def
            );

            let conn = self.connection.lock().unwrap();
            conn.execute(&sql, [])
                .map_err(|e| format!("فشل إنشاء الجدول: {}", e))?;

            println!("✅ [PRODUCTION] تم إنشاء الجدول: {}", name);
            Ok(())
        }

        #[cfg(not(feature = "database"))]
        {
            let cols: Vec<(String, String)> = columns
                .iter()
                .map(|(name, typ)| (name.to_string(), typ.to_string()))
                .collect();

            self.tables.insert(name.to_string(), cols);
            self.data.insert(name.to_string(), Vec::new());

            println!("⚠️ [SIMULATION] محاكاة إنشاء جدول: {}", name);
            Ok(())
        }
    }

    /// تنفيذ استعلام SQL
    pub fn execute(&self, query: &str) -> Result<QueryResult, String> {
        #[cfg(feature = "database")]
        {
            self.execute_real(query)
        }

        #[cfg(not(feature = "database"))]
        {
            self.execute_simulation(query)
        }
    }

    /// تنفيذ استعلام حقيقي (مع feature database)
    #[cfg(feature = "database")]
    fn execute_real(&self, query: &str) -> Result<QueryResult, String> {
        let query_upper = query.trim().to_uppercase();
        let conn = self.connection.lock().unwrap();

        if query_upper.starts_with("SELECT") || query_upper.starts_with("PRAGMA") {
            // استعلام SELECT
            let mut stmt = conn
                .prepare(query)
                .map_err(|e| format!("فشل تحضير الاستعلام: {}", e))?;

            let column_names: Vec<String> =
                stmt.column_names().iter().map(|s| s.to_string()).collect();
            let column_count = column_names.len();

            let rows_result = stmt.query_map([], |row| {
                let mut result_row = Row::new();
                for (i, col_name) in column_names.iter().enumerate().take(column_count) {
                    // محاولة قراءة القيمة بأنواع مختلفة
                    let value: Value = if let Ok(v) = row.get::<_, i64>(i) {
                        Value::Integer(v)
                    } else if let Ok(v) = row.get::<_, f64>(i) {
                        Value::Float(v)
                    } else if let Ok(v) = row.get::<_, String>(i) {
                        Value::Text(v)
                    } else if let Ok(v) = row.get::<_, bool>(i) {
                        Value::Boolean(v)
                    } else if let Ok(v) = row.get::<_, Vec<u8>>(i) {
                        Value::Binary(v)
                    } else {
                        Value::Null
                    };
                    result_row.insert(col_name, value);
                }
                Ok(result_row)
            });

            match rows_result {
                Ok(rows) => {
                    let mut result = QueryResult::new();
                    for row in rows {
                        match row {
                            Ok(r) => result.rows.push(r),
                            Err(e) => eprintln!("⚠️ تحذير: فشل قراءة صف: {}", e),
                        }
                    }
                    println!("✅ [PRODUCTION] تم تنفيذ الاستعلام، {} صف", result.rows.len());
                    Ok(result)
                }
                Err(e) => Err(format!("فشل تنفيذ الاستعلام: {}", e)),
            }
        } else {
            // استعلام تنفيذي (INSERT, UPDATE, DELETE, etc.)
            let result = conn
                .execute(query, [])
                .map_err(|e| format!("فشل تنفيذ الاستعلام: {}", e))?;

            let mut query_result = QueryResult::new();
            query_result.rows_affected = result as u64;

            // محاولة الحصول على last_insert_id
            if query_upper.starts_with("INSERT") {
                let last_id = conn.last_insert_rowid();
                query_result.last_insert_id = Some(last_id as u64);
            }

            println!("✅ [PRODUCTION] تم تنفيذ الاستعلام، {} صف متأثر", result);
            Ok(query_result)
        }
    }

    /// تنفيذ استعلام محاكاة (بدون feature database)
    #[cfg(not(feature = "database"))]
    fn execute_simulation(&self, query: &str) -> Result<QueryResult, String> {
        println!(
            "⚠️ [SIMULATION] محاكاة تنفيذ: {}...",
            &query[..query.len().min(50)]
        );
        Ok(QueryResult::new())
    }

    /// إدراج صف
    pub fn insert(&mut self, table: &str, row: Row) -> Result<(), String> {
        #[cfg(feature = "database")]
        {
            let columns: Vec<&str> = row.columns.keys().map(|s| s.as_str()).collect();
            let placeholders: Vec<&str> = columns.iter().map(|_| "?").collect();

            let sql = format!(
                "INSERT INTO {} ({}) VALUES ({})",
                table,
                columns.join(", "),
                placeholders.join(", ")
            );

            let conn = self.connection.lock().unwrap();
            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| format!("فشل تحضير الاستعلام: {}", e))?;

            // تحويل القيم
            let params: Vec<Box<dyn rusqlite::ToSql>> = columns
                .iter()
                .map(|col| match row.columns.get(*col) {
                    Some(Value::Integer(i)) => Box::new(*i) as Box<dyn rusqlite::ToSql>,
                    Some(Value::Float(f)) => Box::new(*f) as Box<dyn rusqlite::ToSql>,
                    Some(Value::Text(s)) => Box::new(s.clone()) as Box<dyn rusqlite::ToSql>,
                    Some(Value::Boolean(b)) => Box::new(*b as i64) as Box<dyn rusqlite::ToSql>,
                    Some(Value::Binary(b)) => Box::new(b.clone()) as Box<dyn rusqlite::ToSql>,
                    _ => Box::new(rusqlite::types::Null) as Box<dyn rusqlite::ToSql>,
                })
                .collect();

            let params_refs: Vec<&dyn rusqlite::ToSql> =
                params.iter().map(|p| p.as_ref()).collect();

            stmt.execute(&params_refs[..])
                .map_err(|e| format!("فشل إدراج الصف: {}", e))?;

            println!("✅ [PRODUCTION] تم إدراج صف في {}", table);
            Ok(())
        }

        #[cfg(not(feature = "database"))]
        {
            if let Some(rows) = self.data.get_mut(table) {
                rows.push(row);
                println!("⚠️ [SIMULATION] محاكاة إدراج صف في {}", table);
                Ok(())
            } else {
                Err(format!("الجدول {} غير موجود", table))
            }
        }
    }

    /// البحث في جدول
    pub fn select(&self, table: &str) -> Result<Vec<Row>, String> {
        #[cfg(feature = "database")]
        {
            let sql = format!("SELECT * FROM {}", table);
            let result = self.execute(&sql)?;
            Ok(result.rows)
        }

        #[cfg(not(feature = "database"))]
        {
            if let Some(rows) = self.data.get(table) {
                Ok(rows.clone())
            } else {
                Err(format!("الجدول {} غير موجود", table))
            }
        }
    }

    /// إغلاق الاتصال
    pub fn close(&mut self) {
        self.connected = false;
        println!("📝 تم إغلاق اتصال قاعدة البيانات");
    }

    /// هل متصل
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// الحصول على مسار قاعدة البيانات
    pub fn path(&self) -> &str {
        &self.database_path
    }
}

// ===== دوال عربية =====

/// اتصال SQLite جديد
pub fn اتصال_sqlite(path: &str) -> SqliteConnection {
    SqliteConnection::new(path)
}

/// إنشاء قاعدة بيانات في الذاكرة
pub fn قاعدة_بيانات_في_الذاكرة() -> SqliteConnection {
    SqliteConnection::new(":memory:")
}

// ===== اختبارات =====

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection_creation() {
        let conn = SqliteConnection::new(":memory:");
        assert!(conn.is_connected());
    }

    #[test]
    fn test_create_table() {
        let mut conn = SqliteConnection::new(":memory:");
        let result = conn.create_table("users", &[("name", "TEXT"), ("age", "INTEGER")]);
        assert!(result.is_ok());
    }
}
