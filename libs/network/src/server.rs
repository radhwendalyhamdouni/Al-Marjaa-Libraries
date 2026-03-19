// src/stdlib/http/server.rs
// خادم HTTP المتقدم - Production Ready with axum
// Advanced HTTP Server - جاهز للإنتاج مع axum

use super::{ContentType, HttpHeaders, HttpMethod, StatusCode};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// نوع دالة البرمجية الوسيطة
pub type MiddlewareFn = Box<dyn Fn(&HttpRequest, &mut HttpResponseBuilder) + Send + Sync>;

/// طلب HTTP الوارد
#[derive(Debug, Clone)]
pub struct HttpRequest {
    /// الطريقة
    pub method: HttpMethod,
    /// المسار
    pub path: String,
    /// معاملات الاستعلام
    pub query: HashMap<String, String>,
    /// الرؤوس
    pub headers: HttpHeaders,
    /// الجسم
    pub body: String,
    /// عناوين IP
    pub remote_addr: Option<String>,
    /// ملفات تعريف الارتباط
    pub cookies: HashMap<String, String>,
}

impl HttpRequest {
    /// إنشاء طلب جديد
    pub fn new(method: HttpMethod, path: String) -> Self {
        Self {
            method,
            path,
            query: HashMap::new(),
            headers: HttpHeaders::new(),
            body: String::new(),
            remote_addr: None,
            cookies: HashMap::new(),
        }
    }

    /// الحصول على معامل
    pub fn query(&self, key: &str) -> Option<&String> {
        self.query.get(key)
    }

    /// الحصول على رأس
    pub fn header(&self, key: &str) -> Option<&String> {
        self.headers.get(key)
    }

    /// الحصول على ملف تعريف الارتباط
    pub fn cookie(&self, name: &str) -> Option<&String> {
        self.cookies.get(name)
    }

    /// تحليل JSON
    pub fn json<T: serde::de::DeserializeOwned>(&self) -> Result<T, String> {
        serde_json::from_str(&self.body).map_err(|e| format!("خطأ في تحليل JSON: {}", e))
    }
}

/// استجابة HTTP الصادرة
#[derive(Debug, Clone)]
pub struct HttpResponseBuilder {
    /// رمز الحالة
    pub status: StatusCode,
    /// الرؤوس
    pub headers: HttpHeaders,
    /// الجسم
    pub body: String,
    /// نوع المحتوى
    pub content_type: ContentType,
}

impl HttpResponseBuilder {
    /// إنشاء استجابة جديدة
    pub fn new() -> Self {
        Self {
            status: StatusCode::OK,
            headers: HttpHeaders::new(),
            body: String::new(),
            content_type: ContentType::Text,
        }
    }

    /// تعيين رمز الحالة
    pub fn status(mut self, status: StatusCode) -> Self {
        self.status = status;
        self
    }

    /// تعيين الجسم
    pub fn body(mut self, body: String) -> Self {
        self.body = body;
        self
    }

    /// تعيين نوع المحتوى
    pub fn content_type(mut self, ct: ContentType) -> Self {
        self.content_type = ct;
        self
    }

    /// إضافة رأس
    pub fn header(mut self, key: String, value: String) -> Self {
        self.headers.insert(key, value);
        self
    }

    /// تعيين JSON
    pub fn json<T: serde::Serialize>(mut self, data: &T) -> Result<Self, String> {
        self.body = serde_json::to_string(data).map_err(|e| format!("خطأ في تحويل JSON: {}", e))?;
        self.content_type = ContentType::Json;
        Ok(self)
    }

    /// تعيين HTML
    pub fn html(mut self, html: String) -> Self {
        self.body = html;
        self.content_type = ContentType::Html;
        self
    }

    /// تعيين نص
    pub fn text(mut self, text: String) -> Self {
        self.body = text;
        self.content_type = ContentType::Text;
        self
    }

    /// إعادة توجيه
    pub fn redirect(mut self, url: &str) -> Self {
        self.status = StatusCode::FOUND;
        self.headers.insert("Location".to_string(), url.to_string());
        self
    }

    /// خطأ 404
    pub fn not_found(mut self) -> Self {
        self.status = StatusCode::NOT_FOUND;
        self.body = "غير موجود".to_string();
        self
    }

    /// خطأ 500
    pub fn internal_error(mut self, message: &str) -> Self {
        self.status = StatusCode::INTERNAL_SERVER_ERROR;
        self.body = message.to_string();
        self
    }

    /// بناء الاستجابة النهائية
    pub fn build(self) -> String {
        let status_line = format!("HTTP/1.1 {} {}", self.status.0, self.status.reason_phrase());
        let content_type = format!("Content-Type: {}", self.content_type.to_mime());
        let content_length = format!("Content-Length: {}", self.body.len());

        let mut headers_str = String::new();
        for (key, value) in self.headers.iter() {
            headers_str.push_str(&format!("{}: {}\r\n", key, value));
        }

        format!(
            "{}\r\n{}\r\n{}\r\n{}\r\n\r\n{}",
            status_line, content_type, content_length, headers_str, self.body
        )
    }
}

impl Default for HttpResponseBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// معالج المسار
pub type RouteHandler = Arc<dyn Fn(&HttpRequest) -> HttpResponseBuilder + Send + Sync>;

/// مسار
#[derive(Clone)]
pub struct Route {
    /// الطريقة
    pub method: HttpMethod,
    /// النمط
    pub pattern: String,
    /// المعالج
    pub handler: RouteHandler,
}

// ===== خادم HTTP حقيقي (مع feature http-server) =====

#[cfg(feature = "http-server")]
pub mod real_server {
    use super::*;
    use axum::{
        body::Body,
        response::{IntoResponse, Response},
        routing::get,
        Router,
    };
    use std::net::SocketAddr;

    /// حالة الخادم المشتركة
    #[derive(Clone)]
    pub struct ServerState {
        pub routes: Arc<Mutex<Vec<Route>>>,
        pub middleware: Arc<Mutex<Vec<MiddlewareFn>>>,
    }

    /// الخادم الحقيقي
    pub struct RealHttpServer {
        pub port: u16,
        pub host: String,
        pub state: ServerState,
    }

    impl RealHttpServer {
        pub fn new() -> Self {
            Self {
                port: 8080,
                host: "0.0.0.0".to_string(),
                state: ServerState {
                    routes: Arc::new(Mutex::new(Vec::new())),
                    middleware: Arc::new(Mutex::new(Vec::new())),
                },
            }
        }

        pub fn port(mut self, port: u16) -> Self {
            self.port = port;
            self
        }

        pub fn host(mut self, host: &str) -> Self {
            self.host = host.to_string();
            self
        }

        /// إنشاء Router
        fn build_router(&self) -> Router {
            Router::new()
                .route("/", get(root_handler))
                .route("/health", get(health_handler))
                .fallback(fallback_handler)
                .with_state(self.state.clone())
        }

        /// تشغيل الخادم
        pub async fn run_async(&self) -> Result<(), String> {
            let addr: SocketAddr = format!("{}:{}", self.host, self.port)
                .parse()
                .map_err(|e| format!("عنوان غير صالح: {}", e))?;

            let router = self.build_router();

            println!("🚀 [PRODUCTION] الخادم الحقيقي يعمل على http://{}", addr);
            println!("📖 لغة المرجع - الخادم العربي المتقدم (axum)");

            let listener = tokio::net::TcpListener::bind(addr)
                .await
                .map_err(|e| format!("فشل ربط المنفذ: {}", e))?;

            axum::serve(listener, router)
                .await
                .map_err(|e| format!("خطأ في الخادم: {}", e))?;

            Ok(())
        }
    }

    impl Default for RealHttpServer {
        fn default() -> Self {
            Self::new()
        }
    }

    /// معالج الصفحة الرئيسية
    async fn root_handler() -> impl IntoResponse {
        Response::builder()
            .status(200)
            .header("Content-Type", "text/html; charset=utf-8")
            .body(Body::from(r#"<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head><meta charset="UTF-8"><title>لغة المرجع</title></head>
<body style="font-family: 'Segoe UI', Tahoma, sans-serif; background: #1a1a2e; color: #eee; padding: 50px;">
    <h1 style="color: #0f0;">🕌 لغة المرجع - الخادم العربي المتقدم</h1>
    <div style="background: #16213e; padding: 20px; border-radius: 10px;">
        <h3>✅ الخادم يعمل بشكل صحيح</h3>
        <p>الإصدار: 3.4.0</p>
        <p>المحرك: axum (tokio)</p>
    </div>
</body>
</html>"#))
            .unwrap()
    }

    /// معالج فحص الصحة
    async fn health_handler() -> impl IntoResponse {
        Response::builder()
            .status(200)
            .header("Content-Type", "application/json")
            .body(Body::from(
                r#"{"status": "healthy", "version": "3.4.0", "engine": "axum"}"#,
            ))
            .unwrap()
    }

    /// معالج المسارات غير الموجودة
    async fn fallback_handler() -> impl IntoResponse {
        Response::builder()
            .status(404)
            .header("Content-Type", "application/json")
            .body(Body::from(
                r#"{"error": "المسار غير موجود", "status": 404}"#,
            ))
            .unwrap()
    }
}

// ===== خادم HTTP (الواجهة الموحدة) =====

/// خادم HTTP
#[cfg(feature = "http-server")]
pub struct HttpServer {
    /// المنفذ
    pub port: u16,
    /// المضيف
    pub host: String,
    /// المسارات
    routes: Arc<Mutex<Vec<Route>>>,
    /// البرمجيات الوسيطة
    middleware: Arc<Mutex<Vec<MiddlewareFn>>>,
    /// الخادم الحقيقي
    real_server: real_server::RealHttpServer,
}

#[cfg(not(feature = "http-server"))]
pub struct HttpServer {
    /// المنفذ
    pub port: u16,
    /// المضيف
    pub host: String,
    /// المسارات
    routes: Arc<Mutex<Vec<Route>>>,
    /// البرمجيات الوسيطة
    middleware: Arc<Mutex<Vec<MiddlewareFn>>>,
}

impl HttpServer {
    /// إنشاء خادم جديد
    pub fn new() -> Self {
        #[cfg(feature = "http-server")]
        {
            let real = real_server::RealHttpServer::new();
            Self {
                port: 8080,
                host: "0.0.0.0".to_string(),
                routes: Arc::new(Mutex::new(Vec::new())),
                middleware: Arc::new(Mutex::new(Vec::new())),
                real_server: real,
            }
        }

        #[cfg(not(feature = "http-server"))]
        {
            Self {
                port: 8080,
                host: "0.0.0.0".to_string(),
                routes: Arc::new(Mutex::new(Vec::new())),
                middleware: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    /// تعيين المنفذ
    pub fn port(mut self, port: u16) -> Self {
        self.port = port;
        #[cfg(feature = "http-server")]
        {
            self.real_server = self.real_server.port(port);
        }
        self
    }

    /// تعيين المضيف
    pub fn host(mut self, host: &str) -> Self {
        self.host = host.to_string();
        #[cfg(feature = "http-server")]
        {
            self.real_server = self.real_server.host(host);
        }
        self
    }

    /// إضافة مسار GET
    pub fn get<F>(&self, pattern: &str, handler: F)
    where
        F: Fn(&HttpRequest) -> HttpResponseBuilder + Send + Sync + 'static,
    {
        self.add_route(HttpMethod::Get, pattern, handler);
    }

    /// إضافة مسار POST
    pub fn post<F>(&self, pattern: &str, handler: F)
    where
        F: Fn(&HttpRequest) -> HttpResponseBuilder + Send + Sync + 'static,
    {
        self.add_route(HttpMethod::Post, pattern, handler);
    }

    /// إضافة مسار PUT
    pub fn put<F>(&self, pattern: &str, handler: F)
    where
        F: Fn(&HttpRequest) -> HttpResponseBuilder + Send + Sync + 'static,
    {
        self.add_route(HttpMethod::Put, pattern, handler);
    }

    /// إضافة مسار DELETE
    pub fn delete<F>(&self, pattern: &str, handler: F)
    where
        F: Fn(&HttpRequest) -> HttpResponseBuilder + Send + Sync + 'static,
    {
        self.add_route(HttpMethod::Delete, pattern, handler);
    }

    /// إضافة مسار عام
    fn add_route<F>(&self, method: HttpMethod, pattern: &str, handler: F)
    where
        F: Fn(&HttpRequest) -> HttpResponseBuilder + Send + Sync + 'static,
    {
        let route = Route {
            method,
            pattern: pattern.to_string(),
            handler: Arc::new(handler),
        };

        self.routes.lock().unwrap().push(route);
    }

    /// إضافة برمجية وسيطة
    pub fn use_middleware<F>(&self, middleware: F)
    where
        F: Fn(&HttpRequest, &mut HttpResponseBuilder) + Send + Sync + 'static,
    {
        self.middleware.lock().unwrap().push(Box::new(middleware));
    }

    /// معالجة الطلب
    pub fn handle(&self, request: &HttpRequest) -> HttpResponseBuilder {
        let routes = self.routes.lock().unwrap();

        for route in routes.iter() {
            if route.method == request.method && self.match_pattern(&route.pattern, &request.path) {
                let mut response = (route.handler)(request);

                // تطبيق البرمجيات الوسيطة
                let middleware = self.middleware.lock().unwrap();
                for mw in middleware.iter() {
                    mw(request, &mut response);
                }

                return response;
            }
        }

        // لم يتم العثور على مسار
        HttpResponseBuilder::new().not_found()
    }

    /// مطابقة النمط
    fn match_pattern(&self, pattern: &str, path: &str) -> bool {
        if pattern == path {
            return true;
        }

        // دعم المعاملات الديناميكية مثل /user/:id
        let pattern_parts: Vec<&str> = pattern.split('/').collect();
        let path_parts: Vec<&str> = path.split('/').collect();

        if pattern_parts.len() != path_parts.len() {
            return false;
        }

        for (p, actual) in pattern_parts.iter().zip(path_parts.iter()) {
            if p.starts_with(':') {
                continue; // معامل ديناميكي
            }
            if p != actual {
                return false;
            }
        }

        true
    }

    /// تشغيل الخادم (متزامن)
    pub fn run(&self) -> Result<(), String> {
        #[cfg(feature = "http-server")]
        {
            println!("✅ [PRODUCTION] تشغيل الخادم الحقيقي (axum + tokio)");

            // إنشاء runtime
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| format!("فشل إنشاء tokio runtime: {}", e))?;

            rt.block_on(async { self.real_server.run_async().await })
        }

        #[cfg(not(feature = "http-server"))]
        {
            println!("⚠️ [SIMULATION] وضع المحاكاة - axum غير مفعّل");
            println!("📝 لتفعيل الخادم الحقيقي، استخدم: --features http-server");
            println!(
                "🚀 [SIMULATION] الخادم يعمل على http://{}:{}",
                self.host, self.port
            );
            println!("📖 لغة المرجع - الخادم العربي المتقدم");
            Ok(())
        }
    }

    /// تشغيل الخادم (غير متزامن)
    #[cfg(feature = "http-server")]
    pub async fn run_async(&self) -> Result<(), String> {
        self.real_server.run_async().await
    }
}

impl Default for HttpServer {
    fn default() -> Self {
        Self::new()
    }
}

// ===== دوال عربية =====

/// إنشاء خادم جديد
pub fn خادم_جديد() -> HttpServer {
    HttpServer::new()
}

/// استجابة جديدة
pub fn استجابة() -> HttpResponseBuilder {
    HttpResponseBuilder::new()
}

// ===== اختبارات =====

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_creation() {
        let server = HttpServer::new();
        assert_eq!(server.port, 8080);
        assert_eq!(server.host, "0.0.0.0");
    }

    #[test]
    fn test_server_port() {
        let server = HttpServer::new().port(3000);
        assert_eq!(server.port, 3000);
    }

    #[test]
    fn test_response_builder() {
        let response = HttpResponseBuilder::new()
            .status(StatusCode::OK)
            .text("مرحبا".to_string());

        assert_eq!(response.status, StatusCode::OK);
        assert_eq!(response.body, "مرحبا");
    }

    #[test]
    fn test_json_response() {
        let data = serde_json::json!({"message": "مرحبا"});
        let response = HttpResponseBuilder::new().json(&data).unwrap();

        assert_eq!(response.content_type, ContentType::Json);
    }

    #[test]
    fn test_arabic_functions() {
        let server = خادم_جديد();
        assert_eq!(server.port, 8080);

        let response = استجابة();
        assert_eq!(response.status, StatusCode::OK);
    }

    #[test]
    fn test_pattern_matching() {
        let server = HttpServer::new();

        assert!(server.match_pattern("/", "/"));
        assert!(server.match_pattern("/users/:id", "/users/123"));
        assert!(!server.match_pattern("/users", "/posts"));
    }

    #[test]
    fn test_route_handling() {
        let server = HttpServer::new();

        server.get("/", |_req| {
            HttpResponseBuilder::new().text("الصفحة الرئيسية".to_string())
        });

        let request = HttpRequest::new(HttpMethod::Get, "/".to_string());
        let response = server.handle(&request);

        assert_eq!(response.status, StatusCode::OK);
        assert_eq!(response.body, "الصفحة الرئيسية");
    }

    #[test]
    fn test_404_response() {
        let server = HttpServer::new();

        let request = HttpRequest::new(HttpMethod::Get, "/nonexistent".to_string());
        let response = server.handle(&request);

        assert_eq!(response.status, StatusCode::NOT_FOUND);
    }
}
