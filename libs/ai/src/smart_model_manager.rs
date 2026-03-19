// ═══════════════════════════════════════════════════════════════════════════════
// مدير النماذج الذكي العابر للمنصات - لغة المرجع
// ═══════════════════════════════════════════════════════════════════════════════
// يدعم: Windows, Linux, macOS
// الميزات:
// - تنزيل النماذج تلقائياً عند الحاجة
// - تشغيل النموذج فقط عند الطلب (Lazy Loading)
// - إيقاف النموذج بعد فترة من عدم الاستخدام لتوفير الموارد
// - كشف تلقائي لـ llama.cpp على جميع المنصات
// ═══════════════════════════════════════════════════════════════════════════════

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// معلومات النموذج المتاح
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub filename: String,
    pub url: String,
    pub size_mb: u64,
    pub description: String,
}

impl ModelInfo {
    /// النماذج المتاحة للتنزيل
    pub fn available_models() -> Vec<Self> {
        vec![
            Self {
                name: "Qwen 2.5 0.5B (صغير)".to_string(),
                filename: "qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string(),
                url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string(),
                size_mb: 469,
                description: "نموذج صغير وسريع (469 MB) - موصى به للمبتدئين".to_string(),
            },
            Self {
                name: "Qwen 2.5 1.5B (متوسط)".to_string(),
                filename: "qwen2.5-1.5b-instruct-q4_k_m.gguf".to_string(),
                url: "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf".to_string(),
                size_mb: 1100,
                description: "نموذج متوسط متوازن (1.1 GB) - أفضل جودة".to_string(),
            },
            Self {
                name: "Qwen 2.5 0.5B Q8 (دقة عالية)".to_string(),
                filename: "qwen2.5-0.5b-instruct-q8_0.gguf".to_string(),
                url: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf".to_string(),
                size_mb: 700,
                description: "نموذج بدقة أعلى (700 MB)".to_string(),
            },
        ]
    }

    /// النموذج الافتراضي
    pub fn default_model() -> Self {
        Self::available_models().into_iter().next().unwrap()
    }
}

/// حالة النموذج
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelState {
    /// غير موجود
    NotDownloaded,
    /// جاري التنزيل
    Downloading,
    /// موجود لكن غير محمل
    Downloaded,
    /// جاري التحميل
    Loading,
    /// جاهز للاستخدام
    Ready,
    /// جاري الإيقاف
    Stopping,
    /// خطأ
    Error,
}

/// إعدادات مدير النماذج
#[derive(Debug, Clone)]
pub struct ModelManagerConfig {
    /// مسار مجلد النماذج
    pub models_dir: PathBuf,
    /// مهلة عدم الاستخدام قبل الإيقاف (بالثواني)
    pub idle_timeout_secs: u64,
    /// المنفذ للسيرفر
    pub port: u16,
    /// درجة الحرارة
    pub temperature: f32,
    /// الحد الأقصى للتوكنات
    pub max_tokens: usize,
}

impl Default for ModelManagerConfig {
    fn default() -> Self {
        Self {
            models_dir: get_default_models_dir(),
            idle_timeout_secs: 120, // دقيقتان
            port: 8080,
            temperature: 0.7,
            max_tokens: 256,
        }
    }
}

/// الحصول على مسار مجلد النماذج الافتراضي
fn get_default_models_dir() -> PathBuf {
    // الأولوية: مجلد المشروع الحالي
    let local_models = PathBuf::from("models");
    if local_models.exists() {
        return local_models;
    }

    // ثم: مجلد المستخدم
    if let Some(home) = dirs::home_dir() {
        let user_models = home.join(".almarjaa").join("models");
        if user_models.exists() {
            return user_models;
        }
        // إنشاء المجلد إذا لم يكن موجوداً
        if fs::create_dir_all(&user_models).is_ok() {
            return user_models;
        }
    }

    // الافتراضي
    PathBuf::from("models")
}

/// نتيجة الاستدلال
#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub text: String,
    pub tokens_generated: usize,
    pub duration_ms: u64,
    pub model_used: String,
}

/// كاشف المنصة
#[derive(Debug, Clone, Copy)]
pub enum Platform {
    Windows,
    Linux,
    MacOS,
}

impl Platform {
    /// كشف المنصة الحالية
    pub fn current() -> Self {
        #[cfg(target_os = "windows")]
        {
            Platform::Windows
        }

        #[cfg(target_os = "linux")]
        {
            Platform::Linux
        }

        #[cfg(target_os = "macos")]
        {
            Platform::MacOS
        }

        #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
        {
            Platform::Linux
        } // افتراضي
    }

    /// الحصول على مسارات llama.cpp المحتملة
    pub fn llama_server_paths(&self) -> Vec<&'static str> {
        match self {
            Platform::Windows => vec![
                "llama-server.exe",
                "server.exe",
                ".\\llama-server.exe",
                ".\\llama.cpp\\build\\bin\\Release\\llama-server.exe",
                ".\\llama.cpp\\build\\bin\\Debug\\llama-server.exe",
                "C:\\Program Files\\llama.cpp\\llama-server.exe",
                "C:\\llama.cpp\\build\\bin\\Release\\llama-server.exe",
                // مسارات إضافية لـ Windows
                "%LOCALAPPDATA%\\llama.cpp\\llama-server.exe",
                "%USERPROFILE%\\llama.cpp\\build\\bin\\Release\\llama-server.exe",
            ],
            Platform::Linux => vec![
                "llama-server",
                "server",
                "./llama-server",
                "./llama.cpp/build/bin/llama-server",
                "/usr/local/bin/llama-server",
                "/usr/bin/llama-server",
                "~/.local/bin/llama-server",
                "/opt/llama.cpp/llama-server",
            ],
            Platform::MacOS => vec![
                "llama-server",
                "server",
                "./llama-server",
                "./llama.cpp/build/bin/llama-server",
                "/usr/local/bin/llama-server",
                "/opt/homebrew/bin/llama-server",
                "/usr/local/opt/llama.cpp/bin/llama-server",
                "~/llama.cpp/build/bin/llama-server",
            ],
        }
    }

    /// الحصول على مسارات llama-cli المحتملة
    pub fn llama_cli_paths(&self) -> Vec<&'static str> {
        match self {
            Platform::Windows => vec![
                "llama-cli.exe",
                "main.exe",
                ".\\llama-cli.exe",
                ".\\llama.cpp\\build\\bin\\Release\\llama-cli.exe",
                ".\\llama.cpp\\build\\bin\\Release\\main.exe",
                "C:\\Program Files\\llama.cpp\\llama-cli.exe",
            ],
            Platform::Linux => vec![
                "llama-cli",
                "main",
                "./llama-cli",
                "./llama.cpp/build/bin/llama-cli",
                "/usr/local/bin/llama-cli",
                "/usr/bin/llama-cli",
            ],
            Platform::MacOS => vec![
                "llama-cli",
                "main",
                "./llama-cli",
                "./llama.cpp/build/bin/llama-cli",
                "/usr/local/bin/llama-cli",
                "/opt/homebrew/bin/llama-cli",
            ],
        }
    }

    /// الحصول على أداة التنزيل المناسبة
    pub fn download_tools(&self) -> Vec<DownloadTool> {
        match self {
            Platform::Windows => vec![
                DownloadTool::PowerShell,
                DownloadTool::Curl,
                DownloadTool::Reqwest,
            ],
            Platform::Linux | Platform::MacOS => vec![
                DownloadTool::Wget,
                DownloadTool::Curl,
                DownloadTool::Reqwest,
            ],
        }
    }
}

/// أدوات التنزيل المتاحة
#[derive(Debug, Clone, Copy)]
pub enum DownloadTool {
    Wget,
    Curl,
    PowerShell,
    Reqwest,
}

/// مدير النماذج الذكي
pub struct SmartModelManager {
    config: ModelManagerConfig,
    current_model: Option<ModelInfo>,
    state: Arc<Mutex<ModelState>>,
    server_process: Option<Child>,
    last_used: Arc<Mutex<Option<Instant>>>,
    download_progress: Arc<Mutex<u8>>,
    error_message: Option<String>,
    platform: Platform,
    idle_monitor_handle: Option<thread::JoinHandle<()>>,
    shutdown_flag: Arc<Mutex<bool>>,
}

impl SmartModelManager {
    /// إنشاء مدير نماذج جديد
    pub fn new() -> Self {
        Self {
            config: ModelManagerConfig::default(),
            current_model: None,
            state: Arc::new(Mutex::new(ModelState::NotDownloaded)),
            server_process: None,
            last_used: Arc::new(Mutex::new(None)),
            download_progress: Arc::new(Mutex::new(0)),
            error_message: None,
            platform: Platform::current(),
            idle_monitor_handle: None,
            shutdown_flag: Arc::new(Mutex::new(false)),
        }
    }

    /// إنشاء مدير مع إعدادات مخصصة
    pub fn with_config(config: ModelManagerConfig) -> Self {
        Self {
            config,
            current_model: None,
            state: Arc::new(Mutex::new(ModelState::NotDownloaded)),
            server_process: None,
            last_used: Arc::new(Mutex::new(None)),
            download_progress: Arc::new(Mutex::new(0)),
            error_message: None,
            platform: Platform::current(),
            idle_monitor_handle: None,
            shutdown_flag: Arc::new(Mutex::new(false)),
        }
    }

    /// الحصول على المنصة الحالية
    pub fn get_platform(&self) -> Platform {
        self.platform
    }

    /// الحصول على حالة النموذج
    pub fn get_state(&self) -> ModelState {
        *self.state.lock().unwrap()
    }

    /// تعيين حالة النموذج
    fn set_state(&self, new_state: ModelState) {
        *self.state.lock().unwrap() = new_state;
    }

    /// الحصول على رسالة الخطأ
    pub fn get_error(&self) -> Option<&str> {
        self.error_message.as_deref()
    }

    /// التحقق من وجود النموذج
    pub fn is_model_downloaded(&self, model: &ModelInfo) -> bool {
        let model_path = self.config.models_dir.join(&model.filename);
        model_path.exists()
    }

    /// الحصول على النموذج المحمل حالياً
    pub fn get_current_model(&self) -> Option<&ModelInfo> {
        self.current_model.as_ref()
    }

    /// قائمة النماذج المتاحة مع حالتها
    pub fn list_models(&self) -> Vec<(ModelInfo, bool)> {
        ModelInfo::available_models()
            .into_iter()
            .map(|m| {
                let downloaded = self.is_model_downloaded(&m);
                (m, downloaded)
            })
            .collect()
    }

    /// البحث عن llama-server
    pub fn find_llama_server(&self) -> Option<String> {
        for path in self.platform.llama_server_paths() {
            // توسيع المتغيرات البيئية في Windows
            #[cfg(target_os = "windows")]
            let expanded_path = shellexpand::env(path).unwrap_or_default().to_string();

            #[cfg(not(target_os = "windows"))]
            let expanded_path = path.to_string();

            let p = Path::new(&expanded_path);
            if p.exists() {
                return Some(expanded_path);
            }

            // محاولة تشغيل الأمر للتحقق من وجوده في PATH
            if Command::new(path).arg("--version").output().is_ok() {
                return Some(path.to_string());
            }
        }
        None
    }

    /// البحث عن llama-cli
    pub fn find_llama_cli(&self) -> Option<String> {
        for path in self.platform.llama_cli_paths() {
            let p = Path::new(path);
            if p.exists() {
                return Some(path.to_string());
            }

            if Command::new(path).arg("--version").output().is_ok() {
                return Some(path.to_string());
            }
        }
        None
    }

    /// تنزيل النموذج
    pub fn download_model(&mut self, model: &ModelInfo) -> Result<(), String> {
        println!("\n📥 بدء تنزيل النموذج: {}", model.name);
        println!(
            "📁 المسار: {}/{}",
            self.config.models_dir.display(),
            model.filename
        );
        println!("📊 الحجم: {} MB", model.size_mb);
        println!("🖥️ المنصة: {:?}\n", self.platform);

        // إنشاء مجلد النماذج إذا لم يكن موجوداً
        if !self.config.models_dir.exists() {
            fs::create_dir_all(&self.config.models_dir)
                .map_err(|e| format!("فشل إنشاء مجلد النماذج: {}", e))?;
        }

        let model_path = self.config.models_dir.join(&model.filename);

        // التحقق من وجود النموذج
        if model_path.exists() {
            println!("✅ النموذج موجود بالفعل!");
            self.set_state(ModelState::Downloaded);
            return Ok(());
        }

        self.set_state(ModelState::Downloading);
        *self.download_progress.lock().unwrap() = 0;

        // محاولة أدوات التنزيل بالترتيب
        for tool in self.platform.download_tools() {
            let result = match tool {
                DownloadTool::Wget => self.download_with_wget(&model.url, &model_path),
                DownloadTool::Curl => self.download_with_curl(&model.url, &model_path),
                DownloadTool::PowerShell => self.download_with_powershell(&model.url, &model_path),
                DownloadTool::Reqwest => self.download_with_reqwest(&model.url, &model_path),
            };

            if result.is_ok() {
                self.set_state(ModelState::Downloaded);
                *self.download_progress.lock().unwrap() = 100;
                println!("\n✅ تم تنزيل النموذج بنجاح!");
                return Ok(());
            }
        }

        self.set_state(ModelState::Error);
        self.error_message = Some("فشل تنزيل النموذج. تأكد من اتصالك بالإنترنت.".to_string());
        Err(format!(
            "فشل تنزيل النموذج. حاول تنزيله يدوياً من:\n{}",
            model.url
        ))
    }

    /// التنزيل باستخدام wget (Linux/macOS)
    fn download_with_wget(&self, url: &str, path: &Path) -> Result<(), String> {
        println!("🔄 محاولة التنزيل باستخدام wget...");

        let output = Command::new("wget")
            .args([
                "-c", // استئناف التنزيل
                "--progress=bar:force",
                "-O",
            ])
            .arg(path.to_str().unwrap())
            .arg(url)
            .status()
            .map_err(|e| format!("wget غير متاح: {}", e))?;

        if output.success() {
            Ok(())
        } else {
            Err("فشل wget".to_string())
        }
    }

    /// التنزيل باستخدام curl (جميع المنصات)
    fn download_with_curl(&self, url: &str, path: &Path) -> Result<(), String> {
        println!("🔄 محاولة التنزيل باستخدام curl...");

        let output = Command::new("curl")
            .args([
                "-L", // تتبع إعادة التوجيه
                "-C", "-", // استئناف التنزيل
                "-o",
            ])
            .arg(path.to_str().unwrap())
            .arg(url)
            .status()
            .map_err(|e| format!("curl غير متاح: {}", e))?;

        if output.success() {
            Ok(())
        } else {
            Err("فشل curl".to_string())
        }
    }

    /// التنزيل باستخدام PowerShell (Windows)
    fn download_with_powershell(&self, url: &str, path: &Path) -> Result<(), String> {
        println!("🔄 محاولة التنزيل باستخدام PowerShell...");

        let path_str = path.to_str().unwrap();
        let script = format!(
            "Invoke-WebRequest -Uri '{}' -OutFile '{}' -UseBasicParsing",
            url, path_str
        );

        let output = Command::new("powershell")
            .args(["-Command", &script])
            .status()
            .map_err(|e| format!("PowerShell غير متاح: {}", e))?;

        if output.success() {
            Ok(())
        } else {
            Err("فشل PowerShell".to_string())
        }
    }

    /// التنزيل باستخدام reqwest (جميع المنصات - مدمج)
    fn download_with_reqwest(&mut self, url: &str, path: &Path) -> Result<(), String> {
        println!("🔄 محاولة التنزيل المدمج (reqwest)...");

        #[cfg(feature = "network")]
        {
            let response = reqwest::blocking::Client::new()
                .get(url)
                .header("User-Agent", "AlMarjaa-Language/3.4.0")
                .send()
                .map_err(|e| format!("فشل الاتصال: {}", e))?;

            if !response.status().is_success() {
                return Err(format!("خطأ HTTP: {}", response.status()));
            }

            let total_size = response.content_length().unwrap_or(0);

            let mut file = fs::File::create(path).map_err(|e| format!("فشل إنشاء الملف: {}", e))?;

            // استخدام bytes() مباشرة بدلاً من التدفق
            let bytes = response
                .bytes()
                .map_err(|e| format!("فشل قراءة البيانات: {}", e))?;

            file.write_all(&bytes)
                .map_err(|e| format!("فشل الكتابة: {}", e))?;

            let downloaded = bytes.len() as u64;

            if total_size > 0 {
                let progress = ((downloaded as f64 / total_size as f64) * 100.0) as u8;
                *self.download_progress.lock().unwrap() = progress;
                println!(
                    "📊 التقدم: {}% ({:.1} MB / {:.1} MB)",
                    progress,
                    downloaded as f64 / 1_000_000.0,
                    total_size as f64 / 1_000_000.0
                );
            }

            Ok(())
        }

        #[cfg(not(feature = "network"))]
        {
            Err("التنزيل المدمج غير متاح. قم بتفعيل feature 'network' في Cargo.toml".to_string())
        }
    }

    /// تحميل النموذج (تشغيل السيرفر) - Lazy Loading
    pub fn load_model(&mut self, model: &ModelInfo) -> Result<(), String> {
        // التحقق من وجود النموذج
        let model_path = self.config.models_dir.join(&model.filename);

        if !model_path.exists() {
            println!("📥 النموذج غير موجود. جاري التنزيل...");
            self.download_model(model)?;
        }

        println!("\n🚀 تحميل النموذج: {}", model.name);
        println!("🖥️ المنصة: {:?}", self.platform);
        self.set_state(ModelState::Loading);

        // التحقق من وجود سيرفر يعمل
        if self.check_server_health() {
            println!("✅ السيرفر يعمل بالفعل!");
            self.set_state(ModelState::Ready);
            self.current_model = Some(model.clone());
            *self.last_used.lock().unwrap() = Some(Instant::now());
            return Ok(());
        }

        // البحث عن llama-server
        let server_path = self.find_llama_server().ok_or_else(|| {
            let install_url = match self.platform {
                Platform::Windows => "https://github.com/ggerganov/llama.cpp/releases",
                Platform::Linux => {
                    "sudo apt install llama.cpp أو https://github.com/ggerganov/llama.cpp"
                }
                Platform::MacOS => {
                    "brew install llama.cpp أو https://github.com/ggerganov/llama.cpp"
                }
            };
            format!(
                "❌ llama.cpp server غير موجود.\n\
                     📥 قم بتثبيته من: {}\n\
                     💡 أو تأكد من وجوده في PATH",
                install_url
            )
        })?;

        println!("🔧 استخدام: {}", server_path);

        // تشغيل السيرفر
        let server_result = self.start_server(&server_path, &model_path);

        match server_result {
            Ok(_) => {
                self.set_state(ModelState::Ready);
                self.current_model = Some(model.clone());
                *self.last_used.lock().unwrap() = Some(Instant::now());

                // بدء مراقب الخمول
                self.start_idle_monitor();

                println!("✅ تم تحميل النموذج بنجاح!");
                Ok(())
            }
            Err(e) => {
                self.set_state(ModelState::Error);
                self.error_message = Some(e.clone());
                Err(e)
            }
        }
    }

    /// تشغيل السيرفر
    fn start_server(&mut self, server_path: &str, model_path: &Path) -> Result<(), String> {
        println!("⏳ انتظار جاهزية السيرفر...");

        #[cfg(target_os = "windows")]
        let creation_flags = 0x08000000; // CREATE_NO_WINDOW

        #[cfg(not(target_os = "windows"))]
        let child = Command::new(server_path)
            .args([
                "-m",
                model_path.to_str().unwrap(),
                "--port",
                &self.config.port.to_string(),
                "-c",
                "2048",
                "-ngl",
                "99",
                "--host",
                "127.0.0.1",
                "--log-disable",
            ])
            .spawn()
            .map_err(|e| format!("فشل تشغيل السيرفر: {}", e))?;

        #[cfg(target_os = "windows")]
        let child = Command::new(server_path)
            .args([
                "-m",
                model_path.to_str().unwrap(),
                "--port",
                &self.config.port.to_string(),
                "-c",
                "2048",
                "-ngl",
                "99",
                "--host",
                "127.0.0.1",
                "--log-disable",
            ])
            .creation_flags(creation_flags)
            .spawn()
            .map_err(|e| format!("فشل تشغيل السيرفر: {}", e))?;

        self.server_process = Some(child);

        // انتظار جاهزية السيرفر
        for i in 1..=30 {
            thread::sleep(Duration::from_secs(1));
            print!("\r⏳ انتظار السيرفر... {}/30", i);
            io::stdout().flush().ok();

            if self.check_server_health() {
                println!();
                return Ok(());
            }
        }

        Err("انتهت مهلة انتظار السيرفر".to_string())
    }

    /// التحقق من حالة السيرفر
    fn check_server_health(&self) -> bool {
        // محاولة باستخدام curl
        let curl_result = Command::new("curl")
            .args([
                "-s",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                "--connect-timeout",
                "2",
                &format!("http://localhost:{}/health", self.config.port),
            ])
            .output()
            .map(|o| {
                let stdout = String::from_utf8_lossy(&o.stdout);
                stdout.trim() == "200"
            })
            .unwrap_or(false);

        if curl_result {
            return true;
        }

        // في Windows، محاولة باستخدام PowerShell
        #[cfg(target_os = "windows")]
        {
            let ps_script = format!(
                "try {{ (Invoke-WebRequest -Uri 'http://localhost:{}/health' -TimeoutSec 2 -UseBasicParsing).StatusCode }} catch {{ 0 }}",
                self.config.port
            );

            Command::new("powershell")
                .args(["-Command", &ps_script])
                .output()
                .map(|o| {
                    let stdout = String::from_utf8_lossy(&o.stdout);
                    stdout.trim() == "200"
                })
                .unwrap_or(false)
        }

        #[cfg(not(target_os = "windows"))]
        false
    }

    /// بدء مراقب الخمول - يوقف النموذج بعد فترة من عدم الاستخدام
    fn start_idle_monitor(&mut self) {
        let state = Arc::clone(&self.state);
        let last_used = Arc::clone(&self.last_used);
        let shutdown = Arc::clone(&self.shutdown_flag);
        let idle_timeout = self.config.idle_timeout_secs;
        let port = self.config.port;

        *self.shutdown_flag.lock().unwrap() = false;

        let handle = thread::spawn(move || {
            loop {
                thread::sleep(Duration::from_secs(10));

                // التحقق من طلب الإيقاف
                if *shutdown.lock().unwrap() {
                    break;
                }

                // التحقق من الخمول
                let current_state = *state.lock().unwrap();
                if current_state == ModelState::Ready {
                    if let Some(last) = *last_used.lock().unwrap() {
                        let elapsed = last.elapsed().as_secs();
                        if elapsed >= idle_timeout {
                            println!(
                                "\n⏰ مرت {} ثانية منذ آخر استخدام. جاري إيقاف النموذج...",
                                elapsed
                            );

                            // إرسال طلب إيقاف للسيرفر
                            #[cfg(target_os = "windows")]
                            {
                                let _ = Command::new("powershell")
                                    .args(["-Command", &format!(
                                        "Invoke-WebRequest -Uri 'http://localhost:{}/stop' -Method POST -UseBasicParsing",
                                        port
                                    )])
                                    .output();
                            }

                            #[cfg(not(target_os = "windows"))]
                            {
                                let _ = Command::new("curl")
                                    .args([
                                        "-s",
                                        "-X",
                                        "POST",
                                        &format!("http://localhost:{}/stop", port),
                                    ])
                                    .output();
                            }

                            *state.lock().unwrap() = ModelState::Downloaded;
                            println!("✅ تم إيقاف النموذج لتوفير الموارد");
                            break;
                        }
                    }
                }
            }
        });

        self.idle_monitor_handle = Some(handle);
    }

    /// تشغيل الاستدلال
    pub fn infer(&mut self, prompt: &str) -> Result<InferenceResult, String> {
        let current_state = self.get_state();

        if current_state != ModelState::Ready {
            // محاولة التحميل التلقائي
            if current_state == ModelState::Downloaded {
                if let Some(ref model) = self.current_model.clone() {
                    self.load_model(model)?;
                } else {
                    return Err("النموذج غير محدد. قم بتحميل نموذج أولاً.".to_string());
                }
            } else {
                return Err("النموذج غير جاهز. قم بتحميله أولاً.".to_string());
            }
        }

        // تحديث وقت آخر استخدام
        *self.last_used.lock().unwrap() = Some(Instant::now());

        let start = Instant::now();

        // بناء الـ prompt
        let formatted_prompt = self.format_prompt(prompt);

        // إرسال الطلب للسيرفر
        let request_body = serde_json::json!({
            "prompt": formatted_prompt,
            "n_predict": self.config.max_tokens,
            "temperature": self.config.temperature,
        });

        // محاولة باستخدام curl
        let output = Command::new("curl")
            .args([
                "-s",
                "-X",
                "POST",
                "-H",
                "Content-Type: application/json",
                "-d",
                &serde_json::to_string(&request_body).unwrap(),
                &format!("http://localhost:{}/completion", self.config.port),
            ])
            .output();

        let output = match output {
            Ok(o) => o,
            Err(_) => {
                // في Windows، محاولة باستخدام PowerShell
                #[cfg(target_os = "windows")]
                {
                    let ps_script = format!(
                        "$body = '{}'; Invoke-RestMethod -Uri 'http://localhost:{}/completion' -Method POST -Body $body -ContentType 'application/json' | ConvertTo-Json",
                        serde_json::to_string(&request_body).unwrap(),
                        self.config.port
                    );

                    Command::new("powershell")
                        .args(["-Command", &ps_script])
                        .output()
                        .map_err(|e| format!("فشل إرسال الطلب: {}", e))?
                }

                #[cfg(not(target_os = "windows"))]
                return Err("فشل إرسال الطلب للسيرفر".to_string());
            }
        };

        // تحليل الاستجابة
        let response: serde_json::Value = serde_json::from_slice(&output.stdout)
            .map_err(|e| format!("فشل تحليل الاستجابة: {}", e))?;

        let text = response["content"].as_str().unwrap_or("").to_string();
        let tokens = response["tokens_evaluated"].as_u64().unwrap_or(0) as usize;

        Ok(InferenceResult {
            text,
            tokens_generated: tokens,
            duration_ms: start.elapsed().as_millis() as u64,
            model_used: self
                .current_model
                .as_ref()
                .map(|m| m.name.clone())
                .unwrap_or_default(),
        })
    }

    /// تنسيق الـ prompt
    fn format_prompt(&self, user_input: &str) -> String {
        format!(
            "<|im_start|>system\n\
            أنت مساعد برمجي عربي متخصص في لغة المرجع - لغة برمجة عربية.\n\
            قم بتحويل النص العربي إلى كود بلغة المرجع.\n\n\
            قواعد لغة المرجع:\n\
            - المتغيرات: متغير اسم = قيمة؛\n\
            - الدوال: دالة اسم() {{ }}\n\
            - الشرط: إذا شرط {{ }} وإلا {{ }}\n\
            - الحلقة: طالما شرط {{ }}\n\
            - الطباعة: اطبع(\"نص\")؛\n\
            - الأرجاع: أعطِ قيمة؛\n\n\
            أعد فقط الكود بدون شرح إضافي.<|im_end|>\n\
            <|im_start|>user\n{}<|im_end|>\n\
            <|im_start|>assistant\n",
            user_input
        )
    }

    /// إيقاف النموذج
    pub fn stop_model(&mut self) {
        // إيقاف مراقب الخمول
        *self.shutdown_flag.lock().unwrap() = true;

        if let Some(handle) = self.idle_monitor_handle.take() {
            // لن ننتظر الخيط لأنه قد يستغرق وقتاً
            let _ = handle.join();
        }

        if let Some(ref mut child) = self.server_process {
            let _ = child.kill();
            println!("🛑 تم إيقاف النموذج");
        }
        self.server_process = None;
        self.set_state(ModelState::Downloaded);
        self.current_model = None;
    }

    /// تحويل نص إلى كود
    pub fn text_to_code(&mut self, text: &str) -> Result<String, String> {
        let result = self.infer(text)?;
        Ok(result.text)
    }

    /// الحصول على معلومات الحالة
    pub fn status(&self) -> String {
        let state = self.get_state();
        let state_emoji = match state {
            ModelState::NotDownloaded => "⬇️",
            ModelState::Downloading => "📥",
            ModelState::Downloaded => "💾",
            ModelState::Loading => "⏳",
            ModelState::Ready => "✅",
            ModelState::Stopping => "🛑",
            ModelState::Error => "❌",
        };

        let last_used_str = self
            .last_used
            .lock()
            .unwrap()
            .map(|t| format!("منذ {} ثانية", t.elapsed().as_secs()))
            .unwrap_or("غير مستخدم".to_string());

        format!(
            "{} حالة: {:?}\n\
             📦 النموذج: {}\n\
             📁 المسار: {}\n\
             🖥️ المنصة: {:?}\n\
             🕐 آخر استخدام: {}",
            state_emoji,
            state,
            self.current_model
                .as_ref()
                .map(|m| m.name.as_str())
                .unwrap_or("غير محدد"),
            self.config.models_dir.display(),
            self.platform,
            last_used_str,
        )
    }

    /// الحصول على تقدم التنزيل
    pub fn get_download_progress(&self) -> u8 {
        *self.download_progress.lock().unwrap()
    }
}

impl Default for SmartModelManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for SmartModelManager {
    fn drop(&mut self) {
        self.stop_model();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// دوال مساعدة
// ═══════════════════════════════════════════════════════════════════════════════

/// الحصول على معلومات المنصة
pub fn get_platform_info() -> String {
    let platform = Platform::current();
    let mut info = format!("🖥️ المنصة: {:?}\n", platform);
    info.push_str("📁 مسارات llama-server المحتملة:\n");
    for path in platform.llama_server_paths() {
        info.push_str(&format!("  - {}\n", path));
    }
    info
}

/// التحقق من توفر llama.cpp
pub fn check_llama_cpp_availability() -> (bool, Option<String>) {
    let manager = SmartModelManager::new();
    match manager.find_llama_server() {
        Some(path) => (true, Some(path)),
        None => (false, None),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// اختبارات
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manager_creation() {
        let manager = SmartModelManager::new();
        assert_eq!(manager.get_state(), ModelState::NotDownloaded);
    }

    #[test]
    fn test_available_models() {
        let models = ModelInfo::available_models();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_model_info() {
        let default = ModelInfo::default_model();
        assert!(default.filename.contains("qwen"));
    }

    #[test]
    fn test_list_models() {
        let manager = SmartModelManager::new();
        let list = manager.list_models();
        assert!(!list.is_empty());
    }

    #[test]
    fn test_platform_detection() {
        let platform = Platform::current();
        // يجب أن يكون أحد الأنظمة المدعومة
        let is_supported = matches!(
            platform,
            Platform::Windows | Platform::Linux | Platform::MacOS
        );
        assert!(is_supported);
    }

    #[test]
    fn test_llama_paths() {
        let platform = Platform::current();
        let paths = platform.llama_server_paths();
        assert!(!paths.is_empty());
    }

    #[test]
    fn test_download_tools() {
        let platform = Platform::current();
        let tools = platform.download_tools();
        assert!(!tools.is_empty());
    }
}
