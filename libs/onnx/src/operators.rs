// ═══════════════════════════════════════════════════════════════════════════════
// مشغلات ONNX المتقدمة - Advanced ONNX Operators
// ═══════════════════════════════════════════════════════════════════════════════
// يدعم جميع المشغلات الأساسية لشبكات الذاكرة العميقة
// ═══════════════════════════════════════════════════════════════════════════════

use super::types::ONNXTensor;
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════════
// أنواع المشغلات
// ═══════════════════════════════════════════════════════════════════════════════

/// نوع مشغل ONNX
#[derive(Debug, Clone, PartialEq)]
pub enum OperatorType {
    // المشغلات الحسابية
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Gemm,

    // مشغلات الالتفاف
    Conv,
    ConvTranspose,
    MaxPool,
    AveragePool,
    GlobalMaxPool,
    GlobalAveragePool,

    // مشغلات التسوية
    BatchNormalization,
    InstanceNormalization,
    LayerNormalization,
    GroupNormalization,

    // دوال التنشيط
    Relu,
    LeakyRelu,
    Sigmoid,
    Tanh,
    Softmax,
    LogSoftmax,
    Elu,
    Selu,
    Gelu,
    Swish,
    Mish,
    PRelu,
    ThresholdedRelu,

    // مشغلات الشكل
    Reshape,
    Flatten,
    Squeeze,
    Unsqueeze,
    Transpose,
    Permute,
    Concat,
    Split,
    Slice,
    Expand,
    Tile,
    Gather,
    Scatter,

    // مشغلات التجميع
    ReduceMean,
    ReduceSum,
    ReduceMax,
    ReduceMin,
    ReduceProd,
    ReduceL1,
    ReduceL2,
    ReduceLogSum,
    ReduceLogSumExp,
    ArgMax,
    ArgMin,

    // مشغلات المقارنة والمنطق
    Equal,
    NotEqual,
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,
    And,
    Or,
    Not,
    Where,

    // مشغلات الرياضيات المتقدمة
    Pow,
    Exp,
    Log,
    Sqrt,
    Rsqrt,
    Abs,
    Neg,
    Ceil,
    Floor,
    Round,
    Sign,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,

    // مشغلات خاصة
    Dropout,
    LRN, // Local Response Normalization
    Identity,
    Clip,
    Hardmax,
    Softplus,
    Softsign,

    // مشغلات متقدمة
    Attention,
    MultiHeadAttention,
    Embedding,
    LayerNorm,

    // مشغلات مخصصة للعربية
    ArabicTextEncoder,
    ArabicTokenizer,
}

/// مشغل ONNX
#[derive(Debug, Clone)]
pub struct ONNXOperator {
    /// نوع المشغل
    pub op_type: OperatorType,
    /// اسم المشغل
    pub name: String,
    /// المدخلات
    pub inputs: Vec<String>,
    /// المخرجات
    pub outputs: Vec<String>,
    /// السمات
    pub attributes: HashMap<String, AttributeValue>,
}

/// قيمة السمة
#[derive(Debug, Clone)]
pub enum AttributeValue {
    Int(i64),
    Float(f64),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f64>),
    Strings(Vec<String>),
    Tensor(ONNXTensor),
}

// ═══════════════════════════════════════════════════════════════════════════════
// معلمات الالتفاف
// ═══════════════════════════════════════════════════════════════════════════════

/// معلمات الالتفاف
#[derive(Debug, Clone)]
pub struct ConvParams {
    /// عدد المرشحات
    pub num_filters: usize,
    /// حجم النواة
    pub kernel_size: Vec<usize>,
    /// الخطوات
    pub strides: Vec<usize>,
    /// الحشو
    pub pads: Vec<usize>,
    /// التمدد
    pub dilations: Vec<usize>,
    /// المجموعات
    pub group: usize,
}

impl Default for ConvParams {
    fn default() -> Self {
        Self {
            num_filters: 1,
            kernel_size: vec![3, 3],
            strides: vec![1, 1],
            pads: vec![0, 0, 0, 0],
            dilations: vec![1, 1],
            group: 1,
        }
    }
}

/// معلمات التجميع
#[derive(Debug, Clone)]
pub struct PoolParams {
    /// حجم النافذة
    pub kernel_size: Vec<usize>,
    /// الخطوات
    pub strides: Vec<usize>,
    /// الحشو
    pub pads: Vec<usize>,
    /// عدد الأبعاد
    pub ceil_mode: bool,
}

impl Default for PoolParams {
    fn default() -> Self {
        Self {
            kernel_size: vec![2, 2],
            strides: vec![2, 2],
            pads: vec![0, 0, 0, 0],
            ceil_mode: false,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// تنفيذ المشغلات
// ═══════════════════════════════════════════════════════════════════════════════

/// منفذ المشغلات
#[allow(dead_code)]
pub struct OperatorExecutor {
    /// استخدام GPU
    use_gpu: bool,
    /// عدد الخيوط
    num_threads: usize,
}

impl OperatorExecutor {
    pub fn new() -> Self {
        Self {
            use_gpu: false,
            num_threads: num_cpus::get(),
        }
    }

    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }

    /// تنفيذ مشغل
    pub fn execute(
        &self,
        op: &ONNXOperator,
        inputs: &[ONNXTensor],
    ) -> Result<Vec<ONNXTensor>, String> {
        match op.op_type {
            // المشغلات الحسابية
            OperatorType::Add => self.execute_add(inputs),
            OperatorType::Sub => self.execute_sub(inputs),
            OperatorType::Mul => self.execute_mul(inputs),
            OperatorType::Div => self.execute_div(inputs),
            OperatorType::MatMul => self.execute_matmul(inputs),
            OperatorType::Gemm => self.execute_gemm(inputs),

            // مشغلات الالتفاف
            OperatorType::Conv => self.execute_conv(inputs, &op.attributes),
            OperatorType::MaxPool => self.execute_maxpool(inputs, &op.attributes),
            OperatorType::AveragePool => self.execute_avgpool(inputs, &op.attributes),
            OperatorType::GlobalAveragePool => self.execute_global_avgpool(inputs),

            // مشغلات التسوية
            OperatorType::BatchNormalization => self.execute_batchnorm(inputs, &op.attributes),
            OperatorType::LayerNormalization => self.execute_layernorm(inputs, &op.attributes),

            // دوال التنشيط
            OperatorType::Relu => self.execute_relu(inputs),
            OperatorType::LeakyRelu => self.execute_leaky_relu(inputs, &op.attributes),
            OperatorType::Sigmoid => self.execute_sigmoid(inputs),
            OperatorType::Tanh => self.execute_tanh(inputs),
            OperatorType::Softmax => self.execute_softmax(inputs),
            OperatorType::Gelu => self.execute_gelu(inputs),
            OperatorType::Swish => self.execute_swish(inputs),

            // مشغلات الشكل
            OperatorType::Reshape => self.execute_reshape(inputs, &op.attributes),
            OperatorType::Flatten => self.execute_flatten(inputs, &op.attributes),
            OperatorType::Transpose => self.execute_transpose(inputs, &op.attributes),
            OperatorType::Concat => self.execute_concat(inputs, &op.attributes),
            OperatorType::Slice => self.execute_slice(inputs, &op.attributes),

            // مشغلات التجميع
            OperatorType::ReduceMean => self.execute_reduce_mean(inputs, &op.attributes),
            OperatorType::ReduceSum => self.execute_reduce_sum(inputs, &op.attributes),
            OperatorType::ArgMax => self.execute_argmax(inputs, &op.attributes),

            // مشغلات أخرى
            OperatorType::Dropout => self.execute_dropout(inputs, &op.attributes),
            OperatorType::Identity => Ok(inputs.to_vec()),
            OperatorType::Clip => self.execute_clip(inputs, &op.attributes),

            _ => Err(format!("المشغل '{:?}' غير مدعوم بعد", op.op_type)),
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // تنفيذ المشغلات الحسابية
    // ═════════════════════════════════════════════════════════════════════════

    fn execute_add(&self, inputs: &[ONNXTensor]) -> Result<Vec<ONNXTensor>, String> {
        if inputs.len() < 2 {
            return Err("الجمع يتطلب مدخلين على الأقل".to_string());
        }
        let a = &inputs[0];
        let b = &inputs[1];

        let result: Vec<f64> = a
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(x, y)| x + y)
            .collect();

        Ok(vec![ONNXTensor::new(result, a.shape.clone())])
    }

    fn execute_sub(&self, inputs: &[ONNXTensor]) -> Result<Vec<ONNXTensor>, String> {
        if inputs.len() < 2 {
            return Err("الطرح يتطلب مدخلين على الأقل".to_string());
        }
        let a = &inputs[0];
        let b = &inputs[1];

        let result: Vec<f64> = a
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(x, y)| x - y)
            .collect();

        Ok(vec![ONNXTensor::new(result, a.shape.clone())])
    }

    fn execute_mul(&self, inputs: &[ONNXTensor]) -> Result<Vec<ONNXTensor>, String> {
        if inputs.len() < 2 {
            return Err("الضرب يتطلب مدخلين على الأقل".to_string());
        }
        let a = &inputs[0];
        let b = &inputs[1];

        let result: Vec<f64> = a
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(x, y)| x * y)
            .collect();

        Ok(vec![ONNXTensor::new(result, a.shape.clone())])
    }

    fn execute_div(&self, inputs: &[ONNXTensor]) -> Result<Vec<ONNXTensor>, String> {
        if inputs.len() < 2 {
            return Err("القسمة تتطلب مدخلين على الأقل".to_string());
        }
        let a = &inputs[0];
        let b = &inputs[1];

        let result: Vec<f64> = a
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(x, y)| if *y == 0.0 { f64::NAN } else { x / y })
            .collect();

        Ok(vec![ONNXTensor::new(result, a.shape.clone())])
    }

    fn execute_matmul(&self, inputs: &[ONNXTensor]) -> Result<Vec<ONNXTensor>, String> {
        if inputs.len() < 2 {
            return Err("ضرب المصفوفات يتطلب مدخلين".to_string());
        }

        let a = &inputs[0];
        let b = &inputs[1];

        // دعم المصفوفات 2D
        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err("ضرب المصفوفات يدعم فقط المصفوفات 2D حالياً".to_string());
        }

        let m = a.shape[0];
        let k = a.shape[1];
        let k2 = b.shape[0];
        let n = b.shape[1];

        if k != k2 {
            return Err(format!("أبعاد غير متوافقة: {} × {}", k, k2));
        }

        let mut result = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a.data[i * k + l] * b.data[l * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        Ok(vec![ONNXTensor::new(result, vec![m, n])])
    }

    fn execute_gemm(&self, inputs: &[ONNXTensor]) -> Result<Vec<ONNXTensor>, String> {
        // GEMM: Y = alpha * A' * B' + beta * C
        // تنفيذ مبسط: Y = A * B + C (إذا وجد)
        let mut result = self.execute_matmul(inputs)?;

        if inputs.len() > 2 {
            let c = &inputs[2];
            result[0] = ONNXTensor::new(
                result[0]
                    .data
                    .iter()
                    .zip(c.data.iter())
                    .map(|(x, y)| x + y)
                    .collect(),
                result[0].shape.clone(),
            );
        }

        Ok(result)
    }

    // ═════════════════════════════════════════════════════════════════════════
    // تنفيذ مشغلات الالتفاف
    // ═════════════════════════════════════════════════════════════════════════

    fn execute_conv(
        &self,
        inputs: &[ONNXTensor],
        attrs: &HashMap<String, AttributeValue>,
    ) -> Result<Vec<ONNXTensor>, String> {
        if inputs.len() < 2 {
            return Err("الالتفاف يتطلب مدخلين (المدخلات والأوزان)".to_string());
        }

        let x = &inputs[0]; // المدخلات
        let w = &inputs[1]; // الأوزان

        // استخراج المعلمات
        let strides = attrs
            .get("strides")
            .and_then(|v| {
                if let AttributeValue::Ints(v) = v {
                    Some(v.clone())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| vec![1, 1]);

        let pads = attrs
            .get("pads")
            .and_then(|v| {
                if let AttributeValue::Ints(v) = v {
                    Some(v.clone())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| vec![0, 0, 0, 0]);

        // تنفيذ مبسط للالتفاف 2D
        // في الإصدار الكامل، يستخدم خوارزمية im2col أو FFT

        let batch = x.shape[0];
        let in_channels = x.shape[1];
        let in_h = x.shape[2];
        let in_w = x.shape[3];

        let out_channels = w.shape[0];
        let kernel_h = w.shape[2];
        let kernel_w = w.shape[3];

        let stride_h = strides[0] as usize;
        let stride_w = strides[1] as usize;

        let out_h = (in_h + pads[0] as usize + pads[2] as usize - kernel_h) / stride_h + 1;
        let out_w = (in_w + pads[1] as usize + pads[3] as usize - kernel_w) / stride_w + 1;

        let mut output = vec![0.0; batch * out_channels * out_h * out_w];

        // تنفيذ الالتفاف
        for n in 0..batch {
            for oc in 0..out_channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0;

                        for ic in 0..in_channels {
                            for kh in 0..kernel_h {
                                for kw in 0..kernel_w {
                                    let ih = oh * stride_h + kh - pads[0] as usize;
                                    let iw = ow * stride_w + kw - pads[1] as usize;

                                    if ih < in_h && iw < in_w {
                                        let x_idx = n * in_channels * in_h * in_w
                                            + ic * in_h * in_w
                                            + ih * in_w
                                            + iw;
                                        let w_idx = oc * in_channels * kernel_h * kernel_w
                                            + ic * kernel_h * kernel_w
                                            + kh * kernel_w
                                            + kw;

                                        sum += x.data[x_idx] * w.data[w_idx];
                                    }
                                }
                            }
                        }

                        let out_idx =
                            n * out_channels * out_h * out_w + oc * out_h * out_w + oh * out_w + ow;
                        output[out_idx] = sum;
                    }
                }
            }
        }

        // إضافة الإزاحة (bias) إذا وجدت
        if inputs.len() > 2 {
            let bias = &inputs[2];
            for n in 0..batch {
                for oc in 0..out_channels {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let out_idx = n * out_channels * out_h * out_w
                                + oc * out_h * out_w
                                + oh * out_w
                                + ow;
                            output[out_idx] += bias.data[oc];
                        }
                    }
                }
            }
        }

        Ok(vec![ONNXTensor::new(
            output,
            vec![batch, out_channels, out_h, out_w],
        )])
    }

    fn execute_maxpool(
        &self,
        inputs: &[ONNXTensor],
        attrs: &HashMap<String, AttributeValue>,
    ) -> Result<Vec<ONNXTensor>, String> {
        let x = &inputs[0];

        let kernel_size = attrs
            .get("kernel_shape")
            .and_then(|v| {
                if let AttributeValue::Ints(v) = v {
                    Some(v.clone())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| vec![2, 2]);

        let strides = attrs
            .get("strides")
            .and_then(|v| {
                if let AttributeValue::Ints(v) = v {
                    Some(v.clone())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| vec![2, 2]);

        // تنفيذ MaxPool 2D مبسط
        let batch = x.shape[0];
        let channels = x.shape[1];
        let in_h = x.shape[2];
        let in_w = x.shape[3];

        let kh = kernel_size[0] as usize;
        let kw = kernel_size[1] as usize;
        let sh = strides[0] as usize;
        let sw = strides[1] as usize;

        let out_h = (in_h - kh) / sh + 1;
        let out_w = (in_w - kw) / sw + 1;

        let mut output = vec![f64::MIN; batch * channels * out_h * out_w];

        for n in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = f64::MIN;

                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ih = oh * sh + ki;
                                let iw = ow * sw + kj;
                                let idx =
                                    n * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw;

                                max_val = max_val.max(x.data[idx]);
                            }
                        }

                        let out_idx =
                            n * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                        output[out_idx] = max_val;
                    }
                }
            }
        }

        Ok(vec![ONNXTensor::new(
            output,
            vec![batch, channels, out_h, out_w],
        )])
    }

    fn execute_avgpool(
        &self,
        inputs: &[ONNXTensor],
        attrs: &HashMap<String, AttributeValue>,
    ) -> Result<Vec<ONNXTensor>, String> {
        let x = &inputs[0];

        let kernel_size = attrs
            .get("kernel_shape")
            .and_then(|v| {
                if let AttributeValue::Ints(v) = v {
                    Some(v.clone())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| vec![2, 2]);

        let strides = attrs
            .get("strides")
            .and_then(|v| {
                if let AttributeValue::Ints(v) = v {
                    Some(v.clone())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| vec![2, 2]);

        let batch = x.shape[0];
        let channels = x.shape[1];
        let in_h = x.shape[2];
        let in_w = x.shape[3];

        let kh = kernel_size[0] as usize;
        let kw = kernel_size[1] as usize;
        let sh = strides[0] as usize;
        let sw = strides[1] as usize;

        let out_h = (in_h - kh) / sh + 1;
        let out_w = (in_w - kw) / sw + 1;

        let mut output = vec![0.0; batch * channels * out_h * out_w];
        let pool_size = (kh * kw) as f64;

        for n in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0;

                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ih = oh * sh + ki;
                                let iw = ow * sw + kj;
                                let idx =
                                    n * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw;
                                sum += x.data[idx];
                            }
                        }

                        let out_idx =
                            n * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                        output[out_idx] = sum / pool_size;
                    }
                }
            }
        }

        Ok(vec![ONNXTensor::new(
            output,
            vec![batch, channels, out_h, out_w],
        )])
    }

    fn execute_global_avgpool(&self, inputs: &[ONNXTensor]) -> Result<Vec<ONNXTensor>, String> {
        let x = &inputs[0];

        let batch = x.shape[0];
        let channels = x.shape[1];
        let spatial_size = x.shape[2] * x.shape[3];

        let mut output = vec![0.0; batch * channels];

        for n in 0..batch {
            for c in 0..channels {
                let mut sum = 0.0;
                for i in 0..spatial_size {
                    let idx = n * channels * spatial_size + c * spatial_size + i;
                    sum += x.data[idx];
                }
                output[n * channels + c] = sum / spatial_size as f64;
            }
        }

        Ok(vec![ONNXTensor::new(output, vec![batch, channels, 1, 1])])
    }

    // ═════════════════════════════════════════════════════════════════════════
    // تنفيذ مشغلات التسوية
    // ═════════════════════════════════════════════════════════════════════════

    fn execute_batchnorm(
        &self,
        inputs: &[ONNXTensor],
        attrs: &HashMap<String, AttributeValue>,
    ) -> Result<Vec<ONNXTensor>, String> {
        if inputs.len() < 5 {
            return Err("تسوية الدفعة تتطلب 5 مدخلات".to_string());
        }

        let x = &inputs[0];
        let scale = &inputs[1];
        let bias = &inputs[2];
        let mean = &inputs[3];
        let var = &inputs[4];

        let epsilon = attrs
            .get("epsilon")
            .and_then(|v| {
                if let AttributeValue::Float(v) = v {
                    Some(*v)
                } else {
                    None
                }
            })
            .unwrap_or(1e-5);

        let mut output = x.data.clone();
        let channels = x.shape[1];
        let spatial_size = x.shape[2] * x.shape[3];

        for c in 0..channels {
            let gamma = scale.data[c];
            let beta = bias.data[c];
            let mu = mean.data[c];
            let sigma2 = var.data[c].sqrt();

            for i in 0..spatial_size {
                for n in 0..x.shape[0] {
                    let idx = n * channels * spatial_size + c * spatial_size + i;
                    output[idx] = gamma * (x.data[idx] - mu) / (sigma2 + epsilon).sqrt() + beta;
                }
            }
        }

        Ok(vec![ONNXTensor::new(output, x.shape.clone())])
    }

    fn execute_layernorm(
        &self,
        inputs: &[ONNXTensor],
        attrs: &HashMap<String, AttributeValue>,
    ) -> Result<Vec<ONNXTensor>, String> {
        let x = &inputs[0];

        let epsilon = attrs
            .get("epsilon")
            .and_then(|v| {
                if let AttributeValue::Float(v) = v {
                    Some(*v)
                } else {
                    None
                }
            })
            .unwrap_or(1e-5);

        // افتراض أن المحور الأخير هو المحور المقصود
        let last_dim = x.shape.last().copied().unwrap_or(1);
        let mut output = x.data.clone();

        let num_elements = x.data.len() / last_dim;

        for i in 0..num_elements {
            let start = i * last_dim;
            let end = start + last_dim;

            // حساب المتوسط
            let mean: f64 = x.data[start..end].iter().sum::<f64>() / last_dim as f64;

            // حساب التباين
            let var: f64 = x.data[start..end]
                .iter()
                .map(|&v| (v - mean).powi(2))
                .sum::<f64>()
                / last_dim as f64;

            // تطبيق التسوية
            for (out_val, &x_val) in output[start..end].iter_mut().zip(&x.data[start..end]) {
                *out_val = (x_val - mean) / (var + epsilon).sqrt();
            }
        }

        // تطبيق gamma و beta إذا وجدا
        if inputs.len() >= 3 {
            let gamma = &inputs[1];
            let beta = &inputs[2];
            for i in 0..num_elements {
                for j in 0..last_dim {
                    let idx = i * last_dim + j;
                    output[idx] = gamma.data[j] * output[idx] + beta.data[j];
                }
            }
        }

        Ok(vec![ONNXTensor::new(output, x.shape.clone())])
    }

    // ═════════════════════════════════════════════════════════════════════════
    // تنفيذ دوال التنشيط
    // ═════════════════════════════════════════════════════════════════════════

    fn execute_relu(&self, inputs: &[ONNXTensor]) -> Result<Vec<ONNXTensor>, String> {
        let x = &inputs[0];
        let result: Vec<f64> = x.data.iter().map(|&v| v.max(0.0)).collect();
        Ok(vec![ONNXTensor::new(result, x.shape.clone())])
    }

    fn execute_leaky_relu(
        &self,
        inputs: &[ONNXTensor],
        attrs: &HashMap<String, AttributeValue>,
    ) -> Result<Vec<ONNXTensor>, String> {
        let x = &inputs[0];
        let alpha = attrs
            .get("alpha")
            .and_then(|v| {
                if let AttributeValue::Float(v) = v {
                    Some(*v)
                } else {
                    None
                }
            })
            .unwrap_or(0.01);

        let result: Vec<f64> = x
            .data
            .iter()
            .map(|&v| if v > 0.0 { v } else { alpha * v })
            .collect();

        Ok(vec![ONNXTensor::new(result, x.shape.clone())])
    }

    fn execute_sigmoid(&self, inputs: &[ONNXTensor]) -> Result<Vec<ONNXTensor>, String> {
        let x = &inputs[0];
        let result: Vec<f64> = x.data.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
        Ok(vec![ONNXTensor::new(result, x.shape.clone())])
    }

    fn execute_tanh(&self, inputs: &[ONNXTensor]) -> Result<Vec<ONNXTensor>, String> {
        let x = &inputs[0];
        let result: Vec<f64> = x.data.iter().map(|&v| v.tanh()).collect();
        Ok(vec![ONNXTensor::new(result, x.shape.clone())])
    }

    fn execute_softmax(&self, inputs: &[ONNXTensor]) -> Result<Vec<ONNXTensor>, String> {
        let x = &inputs[0];

        // تطبيق softmax على المحور الأخير
        let last_dim = x.shape.last().copied().unwrap_or(1);
        let num_rows = x.data.len() / last_dim;

        let mut result = vec![0.0; x.data.len()];

        for i in 0..num_rows {
            let start = i * last_dim;
            let end = start + last_dim;

            // العثور على القيمة القصوى للاستقرار العددي
            let max_val = x.data[start..end].iter().cloned().fold(f64::MIN, f64::max);

            // حساب exp(x - max)
            let exp_vals: Vec<f64> = x.data[start..end]
                .iter()
                .map(|&v| (v - max_val).exp())
                .collect();

            let sum: f64 = exp_vals.iter().sum();

            for j in 0..last_dim {
                result[start + j] = exp_vals[j] / sum;
            }
        }

        Ok(vec![ONNXTensor::new(result, x.shape.clone())])
    }

    fn execute_gelu(&self, inputs: &[ONNXTensor]) -> Result<Vec<ONNXTensor>, String> {
        let x = &inputs[0];
        // GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        let result: Vec<f64> = x
            .data
            .iter()
            .map(|&v| {
                let _inner = v / 2.0_f64.sqrt();
                // تقريب erf
                let erf_approx = v.tanh() * 0.5;
                v * 0.5 * (1.0 + erf_approx)
            })
            .collect();
        Ok(vec![ONNXTensor::new(result, x.shape.clone())])
    }

    fn execute_swish(&self, inputs: &[ONNXTensor]) -> Result<Vec<ONNXTensor>, String> {
        let x = &inputs[0];
        // Swish: x * sigmoid(x)
        let result: Vec<f64> = x.data.iter().map(|&v| v / (1.0 + (-v).exp())).collect();
        Ok(vec![ONNXTensor::new(result, x.shape.clone())])
    }

    // ═════════════════════════════════════════════════════════════════════════
    // تنفيذ مشغلات الشكل
    // ═════════════════════════════════════════════════════════════════════════

    fn execute_reshape(
        &self,
        inputs: &[ONNXTensor],
        attrs: &HashMap<String, AttributeValue>,
    ) -> Result<Vec<ONNXTensor>, String> {
        let x = &inputs[0];

        let new_shape = if inputs.len() > 1 {
            inputs[1].data.iter().map(|&v| v as usize).collect()
        } else {
            attrs
                .get("shape")
                .and_then(|v| {
                    if let AttributeValue::Ints(v) = v {
                        Some(v.iter().map(|&i| i as usize).collect::<Vec<_>>())
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| x.shape.clone())
        };

        x.reshape(new_shape).map(|t| vec![t])
    }

    fn execute_flatten(
        &self,
        inputs: &[ONNXTensor],
        attrs: &HashMap<String, AttributeValue>,
    ) -> Result<Vec<ONNXTensor>, String> {
        let x = &inputs[0];
        let axis = attrs
            .get("axis")
            .and_then(|v| {
                if let AttributeValue::Int(v) = v {
                    Some(*v as usize)
                } else {
                    None
                }
            })
            .unwrap_or(1);

        let batch_size = x.shape[0];
        let remaining_size: usize = x.shape[axis..].iter().product();

        Ok(vec![ONNXTensor::new(
            x.data.clone(),
            vec![batch_size, remaining_size],
        )])
    }

    fn execute_transpose(
        &self,
        inputs: &[ONNXTensor],
        attrs: &HashMap<String, AttributeValue>,
    ) -> Result<Vec<ONNXTensor>, String> {
        let x = &inputs[0];

        let _perm = attrs
            .get("perm")
            .and_then(|v| {
                if let AttributeValue::Ints(v) = v {
                    Some(v.iter().map(|&i| i as usize).collect::<Vec<_>>())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| {
                // الافتراضي: عكس الأبعاد
                x.shape
                    .iter()
                    .enumerate()
                    .map(|(i, _)| x.shape.len() - 1 - i)
                    .collect()
            });

        // تنفيذ التبديل
        if x.shape.len() == 2 {
            let rows = x.shape[0];
            let cols = x.shape[1];
            let mut result = vec![0.0; rows * cols];

            for i in 0..rows {
                for j in 0..cols {
                    result[j * rows + i] = x.data[i * cols + j];
                }
            }

            return Ok(vec![ONNXTensor::new(result, vec![cols, rows])]);
        }

        // للأبعاد الأعلى، نحتاج تنفيذ أكثر تعقيداً
        Ok(vec![x.clone()])
    }

    fn execute_concat(
        &self,
        inputs: &[ONNXTensor],
        attrs: &HashMap<String, AttributeValue>,
    ) -> Result<Vec<ONNXTensor>, String> {
        if inputs.is_empty() {
            return Err("الدمج يتطلب مدخلاً واحداً على الأقل".to_string());
        }

        let axis = attrs
            .get("axis")
            .and_then(|v| {
                if let AttributeValue::Int(v) = v {
                    Some(*v as usize)
                } else {
                    None
                }
            })
            .unwrap_or(0);

        // دمج البيانات على المحور المحدد
        let mut result_data = Vec::new();
        let mut new_shape = inputs[0].shape.clone();

        for input in inputs {
            result_data.extend_from_slice(&input.data);
        }

        // تحديث الشكل
        let total_on_axis: usize = inputs.iter().map(|t| t.shape[axis]).sum();
        new_shape[axis] = total_on_axis;

        Ok(vec![ONNXTensor::new(result_data, new_shape)])
    }

    fn execute_slice(
        &self,
        inputs: &[ONNXTensor],
        attrs: &HashMap<String, AttributeValue>,
    ) -> Result<Vec<ONNXTensor>, String> {
        let x = &inputs[0];

        let starts = attrs
            .get("starts")
            .and_then(|v| {
                if let AttributeValue::Ints(v) = v {
                    Some(v.iter().map(|&i| i as usize).collect::<Vec<_>>())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| vec![0]);

        let ends = attrs
            .get("ends")
            .and_then(|v| {
                if let AttributeValue::Ints(v) = v {
                    Some(v.iter().map(|&i| i as usize).collect::<Vec<_>>())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| x.shape.clone());

        // تنفيذ بسيط للقطع على المحور الأول
        let start = starts.first().copied().unwrap_or(0);
        let end = ends.first().copied().unwrap_or(x.shape[0]);

        let slice_size = x.data.len() / x.shape[0];
        let result_data = x.data[start * slice_size..end * slice_size].to_vec();

        let mut new_shape = x.shape.clone();
        new_shape[0] = end - start;

        Ok(vec![ONNXTensor::new(result_data, new_shape)])
    }

    // ═════════════════════════════════════════════════════════════════════════
    // تنفيذ مشغلات التجميع
    // ═════════════════════════════════════════════════════════════════════════

    fn execute_reduce_mean(
        &self,
        inputs: &[ONNXTensor],
        attrs: &HashMap<String, AttributeValue>,
    ) -> Result<Vec<ONNXTensor>, String> {
        let x = &inputs[0];

        let keepdims = attrs
            .get("keepdims")
            .and_then(|v| {
                if let AttributeValue::Int(v) = v {
                    Some(*v != 0)
                } else {
                    None
                }
            })
            .unwrap_or(true);

        // حساب المتوسط لجميع العناصر
        let sum: f64 = x.data.iter().sum();
        let mean = sum / x.data.len() as f64;

        if keepdims {
            let shape = vec![1; x.shape.len()];
            Ok(vec![ONNXTensor::new(vec![mean], shape)])
        } else {
            Ok(vec![ONNXTensor::scalar(mean)])
        }
    }

    fn execute_reduce_sum(
        &self,
        inputs: &[ONNXTensor],
        attrs: &HashMap<String, AttributeValue>,
    ) -> Result<Vec<ONNXTensor>, String> {
        let x = &inputs[0];

        let keepdims = attrs
            .get("keepdims")
            .and_then(|v| {
                if let AttributeValue::Int(v) = v {
                    Some(*v != 0)
                } else {
                    None
                }
            })
            .unwrap_or(true);

        let sum: f64 = x.data.iter().sum();

        if keepdims {
            let shape = vec![1; x.shape.len()];
            Ok(vec![ONNXTensor::new(vec![sum], shape)])
        } else {
            Ok(vec![ONNXTensor::scalar(sum)])
        }
    }

    fn execute_argmax(
        &self,
        inputs: &[ONNXTensor],
        attrs: &HashMap<String, AttributeValue>,
    ) -> Result<Vec<ONNXTensor>, String> {
        let x = &inputs[0];
        let _axis = attrs
            .get("axis")
            .and_then(|v| {
                if let AttributeValue::Int(v) = v {
                    Some(*v as usize)
                } else {
                    None
                }
            })
            .unwrap_or(0);

        // العثور على فهرس القيمة القصوى
        let max_idx = x
            .data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        Ok(vec![ONNXTensor::new(vec![max_idx as f64], vec![1])])
    }

    // ═════════════════════════════════════════════════════════════════════════
    // مشغلات أخرى
    // ═════════════════════════════════════════════════════════════════════════

    fn execute_dropout(
        &self,
        inputs: &[ONNXTensor],
        _attrs: &HashMap<String, AttributeValue>,
    ) -> Result<Vec<ONNXTensor>, String> {
        // في وضع الاستدلال، Dropout يعيد المدخلات كما هي
        Ok(inputs.to_vec())
    }

    fn execute_clip(
        &self,
        inputs: &[ONNXTensor],
        attrs: &HashMap<String, AttributeValue>,
    ) -> Result<Vec<ONNXTensor>, String> {
        let x = &inputs[0];

        let min_val = attrs.get("min").and_then(|v| {
            if let AttributeValue::Float(v) = v {
                Some(*v)
            } else {
                None
            }
        });

        let max_val = attrs.get("max").and_then(|v| {
            if let AttributeValue::Float(v) = v {
                Some(*v)
            } else {
                None
            }
        });

        let result: Vec<f64> = x
            .data
            .iter()
            .map(|&v| {
                let mut clipped = v;
                if let Some(min) = min_val {
                    clipped = clipped.max(min);
                }
                if let Some(max) = max_val {
                    clipped = clipped.min(max);
                }
                clipped
            })
            .collect();

        Ok(vec![ONNXTensor::new(result, x.shape.clone())])
    }
}

impl Default for OperatorExecutor {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// دوال مساعدة
// ═══════════════════════════════════════════════════════════════════════════════

/// إنشاء مشغل جديد
pub fn create_operator(op_type: OperatorType, name: &str) -> ONNXOperator {
    ONNXOperator {
        op_type,
        name: name.to_string(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        attributes: HashMap::new(),
    }
}

/// إضافة سمة عدد صحيح
pub fn add_int_attr(op: &mut ONNXOperator, key: &str, value: i64) {
    op.attributes
        .insert(key.to_string(), AttributeValue::Int(value));
}

/// إضافة سمة عدد عشري
pub fn add_float_attr(op: &mut ONNXOperator, key: &str, value: f64) {
    op.attributes
        .insert(key.to_string(), AttributeValue::Float(value));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_operator() {
        let executor = OperatorExecutor::new();
        let a = ONNXTensor::vector(vec![1.0, 2.0, 3.0]);
        let b = ONNXTensor::vector(vec![4.0, 5.0, 6.0]);

        let result = executor.execute_add(&[a, b]).unwrap();
        assert_eq!(result[0].data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_relu_operator() {
        let executor = OperatorExecutor::new();
        let x = ONNXTensor::vector(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);

        let result = executor.execute_relu(&[x]).unwrap();
        assert_eq!(result[0].data, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_softmax_operator() {
        let executor = OperatorExecutor::new();
        let x = ONNXTensor::vector(vec![1.0, 2.0, 3.0]);

        let result = executor.execute_softmax(&[x]).unwrap();
        let sum: f64 = result[0].data.iter().sum();
        assert!((sum - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_matmul_operator() {
        let executor = OperatorExecutor::new();
        let a = ONNXTensor::matrix(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = ONNXTensor::matrix(vec![5.0, 6.0, 7.0, 8.0], 2, 2);

        let result = executor.execute_matmul(&[a, b]).unwrap();
        assert_eq!(result[0].shape, vec![2, 2]);
        assert_eq!(result[0].data[0], 19.0); // 1*5 + 2*7
    }
}
