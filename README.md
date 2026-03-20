<div align="center">

# مكتبات لغة المرجع
### Al-Marjaa Language Libraries

[![Version](https://img.shields.io/badge/version-3.4.0-blue.svg)](https://github.com/radhwendalyhamdouni/Al-Marjaa-Libraries)

**المخترع والمطور: رضوان دالي حمدوني**

</div>

---

## 🎯 نظرة عامة

هذا المستودع يحتوي على المكتبات المتخصصة للغة المرجع، وهي مكملة للنواة الأساسية.

| المستودع | الوصف |
|----------|-------|
| [Al-Marjaa-Core](https://github.com/radhwendalyhamdouni/Al-Marjaa-Core) | النواة الأساسية (Lexer, Parser, Interpreter, JIT) |
| [Al-Marjaa-Libraries](https://github.com/radhwendalyhamdouni/Al-Marjaa-Libraries) | المكتبات المتخصصة (AI, Database, Network, ...) |

---

## 📦 المكتبات المتاحة

| المكتبة | الوصف | التبعيات |
|---------|-------|----------|
| **ai** | الذكاء الاصطناعي، Vibe Coding، ONNX | - |
| **database** | SQLite, PostgreSQL, MySQL, MongoDB | rusqlite, sqlx |
| **network** | HTTP Client/Server, WebSocket | reqwest, axum, tokio |
| **onnx** | ONNX Runtime للنماذج العصبية | onnxruntime |
| **gpu** | تسريع GPU (CUDA, OpenCL) | cuda, opencl |
| **industrial** | SCADA, Modbus, MQTT, HMI | tokio, rumqttc |

---

## 🔗 التكامل مع النواة

### الطريقة 1: استخدام Cargo Workspace

```toml
# Cargo.toml
[workspace]
members = [
    "almarjaa-core",
    "almarjaa-libraries/libs/database",
    "almarjaa-libraries/libs/network",
]

[workspace.dependencies]
almarjaa = { path = "almarjaa-core" }
almarjaa-database = { path = "almarjaa-libraries/libs/database" }
almarjaa-network = { path = "almarjaa-libraries/libs/network" }
```

### الطريقة 2: استخدام Git

```toml
[dependencies]
# النواة الأساسية
almarjaa = { git = "https://github.com/radhwendalyhamdouni/Al-Marjaa-Core" }

# المكتبات المطلوبة
almarjaa-database = { git = "https://github.com/radhwendalyhamdouni/Al-Marjaa-Libraries" }
almarjaa-network = { git = "https://github.com/radhwendalyhamdouni/Al-Marjaa-Libraries" }
```

### الطريقة 3: التحميل الديناميكي (مخطط)

```mrj
# في كود المرجع
استيراد "قواعد_البيانات" كـ ديبي؛
استيراد "الشبكة" كـ نت؛

متغير db = ديبي.اتصل("app.db")؛
متغير رد = نت.احصل("https://api.example.com")؛
```

---

## 📥 التثبيت

### مع النواة
```bash
# تثبيت النواة
git clone https://github.com/radhwendalyhamdouni/Al-Marjaa-Core.git
cd Al-Marjaa-Core
cargo build --release
```

### مع المكتبات
```bash
# استنساخ المكتبات
git clone https://github.com/radhwendalyhamdouni/Al-Marjaa-Libraries.git

# بناء مكتبة محددة
cd Al-Marjaa-Libraries/libs/database
cargo build --release --features sqlite
```

---

## 💡 مثال الاستخدام

### مع Rust

```rust
use almarjaa::{Interpreter, Value};
use almarjaa_database::Database;
use almarjaa_network::HttpClient;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // إنشاء مفسر
    let mut interpreter = Interpreter::new();
    
    // استخدام قاعدة البيانات
    let db = Database::sqlite("app.db")?;
    db.execute("CREATE TABLE users (id INTEGER, name TEXT)")?;
    
    // استخدام الشبكة
    let client = HttpClient::new();
    let response = client.get("https://api.example.com").send()?;
    
    println!("Response: {}", response.text()?);
    Ok(())
}
```

---

## 🏗️ بنية المشروع

```
Al-Marjaa-Libraries/
├── Cargo.toml              # Workspace configuration
├── libs/
│   ├── ai/                 # الذكاء الاصطناعي
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── inference.rs
│   │   │   ├── arabic_nlp.rs
│   │   │   └── vibe_advanced.rs
│   │   └── Cargo.toml
│   │
│   ├── database/           # قواعد البيانات
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── sqlite.rs
│   │   │   ├── postgres.rs
│   │   │   └── mysql.rs
│   │   └── Cargo.toml
│   │
│   ├── network/            # الشبكات
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── client.rs
│   │   │   ├── server.rs
│   │   │   └── websocket.rs
│   │   └── Cargo.toml
│   │
│   ├── industrial/         # التطبيقات الصناعية
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── modbus.rs
│   │   │   ├── mqtt.rs
│   │   │   └── scada.rs
│   │   └── Cargo.toml
│   │
│   ├── onnx/               # ONNX Runtime
│   │   └── ...
│   │
│   └── gpu/                # GPU Computing
│       └── ...
│
└── README.md
```

---

## 📋 حالة المكتبات

| المكتبة | الحالة | ميزات |
|---------|--------|-------|
| database | ✅ جاهزة | SQLite, PostgreSQL, MySQL, MongoDB |
| network | ✅ جاهزة | HTTP Client, Server, WebSocket |
| ai | ✅ جاهزة | Vibe Coding, Arabic NLP |
| industrial | ✅ جاهزة | Modbus, MQTT, SCADA |
| onnx | ⚠️ تجريبية | ONNX Runtime |
| gpu | ⚠️ تجريبية | CUDA, OpenCL |

---

## 🔄 التطوير المستقبلي

- [ ] نشر على crates.io
- [ ] نظام تحميل ديناميكي للمكتبات
- [ ] دعم WebAssembly
- [ ] تحسين الأداء

---

## 👨‍💻 المؤلف

**رضوان دالي حمدوني**
- البريد: almarjaa.project@hotmail.com

---

**صُنع بـ ❤️ للعالم العربي**
