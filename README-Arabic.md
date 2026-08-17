# open-cuda

*English*: [README-English.md](README-English.md) ·
*لغات أخرى*: [Deutsch](README-German.md) · [Italiano](README-Italian.md) ·
[Français](README-French.md) · [Русский](README-Russian.md) ·
[Українська](README-Ukrainian.md) · [עברית](README-Hebrew.md) ·
[فارسی](README-Persian.md) · [العربية](README-Arabic.md)

> **آخر تحديث (2026-08-10)**: تمت إضافة `generate_with_repetition_penalty`
> (عقوبة التكرار على طراز CTRL — القيمة penalty>1.0 تُضعف قيم logits
> للرموز التي ظهرت بالفعل) إلى `open-cuda-llm::GptModel`. أصبحت الدالة
> الحالية `generate()` الآن غلافًا رقيقًا يستدعيها بـ `penalty=1.0`
> (سلوك مطابق تمامًا بايتًا بايت، دون أي تراجع في الاختبارات/المستدعين
> الحاليين). هذا يعالج مباشرة خللًا مُبلَّغًا عنه في `aruaru-llm` حيث
> يدخل فك الترميز الجشع لنموذج GPT-2 الأساسي (دون ضبط دقيق للحوار) في
> حلقة لا نهائية لنفس السلسلة النصية (مثل "Student: Hello"). أُضيف
> اختبار جديد على أوزان GPT-2 124M حقيقية،
> `repetition_penalty_reduces_degenerate_loop_on_real_gpt2_weights`،
> يؤكد أن الحلقة تتكرر بالفعل دون العقوبة، وتتوقف بالفعل (مُنتِجةً نصًا
> طبيعيًا نحويًا) عند `penalty=1.3`. أصبحت `/v1/generate` في
> `aruaru-llm` تستدعي الآن هذه الواجهة الجديدة بقيمة افتراضية
> `penalty=1.3` (قابلة للتجاوز عبر `ARUARU_LLM_REPETITION_PENALTY`).
> راجع إدخال HANDOFF بتاريخ 2026-08-10 في [CLAUDE.md](CLAUDE.md).

> **آخر تحديث (2026-08-08)**: أُعيد التحقق على أجهزة حقيقية من ملاحظة
> "تم تنفيذ MLA" بتاريخ 2026-08-06
> (`cargo test -p opencuda-blas mla -- --nocapture` → `1 passed; 0
> failed`، تم اجتياز مسار Vulkan الحقيقي على GT730). تم بحث الدقة
> المختلطة FP8 و DeepSeekMoE لكن **تقرر عدم تنفيذ أيهما**: لا يملك FP8
> أي دعم عتادي حقيقي على وحدة معالجة الرسومات الوحيدة في هذا الجهاز
> (GT730، Kepler، CC 3.5 — لا توجد أنوية Tensor لـ FP8، وسيكون
> محاكاة برمجية فقط)؛ لا يملك DeepSeekMoE نقطة تكامل حقيقية لأن
> `DecoderLayer` في `open-cuda-llm` يحتوي فقط على FFN كثيف واحد
> (بدون بنية خبراء/موجّه)، ولا توجد نقطة تفتيش MoE حقيقية. راجع إدخال
> HANDOFF بتاريخ 2026-08-08 في [CLAUDE.md](CLAUDE.md) للتفاصيل.

> **آخر تحديث (2026-08-07)**: تم توصيل نواة flash-attention المدمجة
> (`flash_attention_with_spirv`) بـ `DecoderLayer` في `open-cuda-llm`،
> عبر `GptModel::set_flash_attention_spirv()` — تعود إلى المسار
> الحالي ذي الإرسالات الثلاثة عند عدم التعيين (متوافقة تمامًا مع
> الإصدارات السابقة). تم التحقق على عتاد حقيقي من NVIDIA GT 730: سلاسل
> الرموز المُولَّدة عبر Vulkan مطابقة بايتًا بايت لمسار المعالج. راجع
> HANDOFF في [CLAUDE.md](CLAUDE.md) للتفاصيل.

> **تم التنفيذ في 2026-08-06**: ضغط ذاكرة تخزين KV منخفض الرتبة، مستوحى
> من Multi-Head Latent Attention (MLA) الخاص بـ DeepSeek-V3 —
> `opencuda-blas::mla_compress_kv`/`mla_decompress_kv`. بعد دراسة
> التقرير التقني ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437))
> ومدونات التنفيذ باليابانية والإنجليزية، تم بناء آلية الإسقاط منخفض
> الرتبة (التصميم وراء تخفيض ذاكرة KV بنسبة 93.3% المُبلَّغ عنه من قِبل
> DeepSeek) فوق نظام `sgemm` الحالي الذي سبق التحقق منه على عتاد حقيقي.
> تم التحقق على عتاد حقيقي (GT730) من تطابق مساري CPU و Vulkan
> رقميًا. لا يحمل أوزانًا مدرَّبة (هذا يوضح الآلية، وليس جودة الضغط
> المدرَّبة — راجع قسم HANDOFF في `CLAUDE.md` للإفصاح الصادق). لا يزال
> تطبيق تقنيات Toshiba SBM / DeepSeek على المستودعات السبعة الأخرى قيد
> الدراسة.

> **تم التحديث في 2026-07-25**: تمت إعادة تسمية عنوان ملف سياسة التطوير
> (`CLAUDE.md`) من "سياسة التطوير وقواعد بيئة التطوير" إلى "فلسفة
> التصميم وسياسة التطوير وقواعد بيئة التطوير"، لفصل فلسفة تصميم
> المشروع (ما نُقدّره) وسياسة التطوير (كيف نعمل) وقواعد بيئة التطوير
> (الاتفاقيات التشغيلية الملموسة) بوضوح أكبر. راجع `CLAUDE.md` للتفاصيل.

**بداية التطوير: 2026-06-26** (تاريخ إنشاء هذا المستودع على GitHub)

"CUDA الثاني" — أساس لتجريد/حوسبة وحدة معالجة الرسومات (تصميم
`OmniGPU`) يهدف إلى التوافق مع Windows/macOS/Linux و
Intel/AMD/NVIDIA. مُقترن ("SET") مع `aruaru-llm`، وهو المستهلك
التنفيذي لخط أنابيب تنفيذ GPU/CPU.

## ما هذا

- **`opencuda-core`**: السمة (trait) `GpuDevice` المشتركة بين جميع
  الواجهات الخلفية (مكافئ CUDA Runtime API: `alloc`/`memcpy`/
  `launch_kernel`).
- **`opencuda-cpu`**: الواجهة الخلفية للمعالج (توازي البيانات عبر
  `rayon`).
- **`opencuda-vulkan`**: الواجهة الخلفية Vulkan Compute (عابرة
  للمنصات، أصلية على Windows/Linux/Android، وmacOS/iOS عبر MoltenVK).
  تم التحقق من GEMM/Attention/تكميم INT4·INT8 على تنفيذ Vulkan حقيقي.
- **`opencuda-directx`** (أُضيفت في 2026-07-23): واجهة خلفية DirectX
  12 Compute (خاصة بـ Windows فقط، واجهة خلفية اختيارية تتعايش مع
  Vulkan). تم التحقق من إرسال GPU لـ `vector_add`/`matmul`/`ChaCha20`
  على عتاد حقيقي (NVIDIA GT 730) — يطابق الناتج تمامًا تطبيقات
  المرجع الخاصة بالمعالج (مثل حزمة `chacha20` الخاصة بـ RustCrypto)
  في الاختبارات. كما تم تنفيذ الحصول الحقيقي على اسم المورّد/سعة
  VRAM عبر تعداد مهايئات DXGI.
- **`opencuda-blas`**: مكافئ NumPy (GEMM/Attention/تكميم).
- **`open-cuda-bert`**: المرور الأمامي لمشفّرات عائلة BERT (يدعم
  multilingual-e5-small).
- **`open-cuda-llm`**: مكافئ vLLM (فك ترميز جشع مع ذاكرة تخزين KV).
  ينفّذ `GptModel::load`، الذي يحمّل ملفات `safetensors` لـ GPT-2
  (Hugging Face `openai-community/gpt2`) (2026-07-25، بنفس تصميم
  `open-cuda-bert::BertModel::load`). تم التحقق على عتاد حقيقي: تنزيل
  وتحميل أوزان GPT-2 124M الحقيقية يُنتج إنجليزية مفكوكة الترميز
  بطريقة جشعة أكثر طلاقة بوضوح من التهيئة العشوائية (ناتج عديم
  المعنى) — مثل "The quick brown fox" → "es are a great way to get a
  little bit of a". راجع HANDOFF في `CLAUDE.md` للتفاصيل.
- **`open-cuda-whisper`** (أُضيفت في 2026-07-31): مكافئ Whisper
  (التعرف على الكلام، المرتبة #6 في خريطة طريق أبحاث السوق). استخراج
  log-mel-spectrogram + مشفّر (نفس تصميم Multi-Head Attention
  الخاص بـ `open-cuda-bert`) + مفكّك ترميز بذاكرة تخزين KV (نفس تصميم
  `open-cuda-llm`) + cross-attention. **حاليًا مجرد MVP بتهيئة
  عشوائية فقط** (لا يوجد بعد محمّل لأوزان Whisper المدرَّبة — راجع
  HANDOFF في `CLAUDE.md` للتفاصيل).

## لماذا لدينا كل من DirectX و Vulkan (قرار تقني بتاريخ 2026-07-23)

في البداية، كان يُعتقد أن هذا المشروع "قيد التطوير كإضافة DirectX".
بعد التحقق من ذلك عبر البحث على الويب باليابانية/الإنجليزية، تبيّن أن
DXVK/vkd3d-proton (التقنية التي يستخدمها Proton من Valve فعليًا) كلاهما
يحوّل في اتجاه "DirectX (واجهة برمجة خاصة بـ Windows فقط) → Vulkan
(واجهة برمجة عابرة للمنصات)" — ولم يُعثر على أمثلة للاتجاه المعاكس.
**لهدف الدعم العابر للمنصات، يُعد نهج Vulkan Compute الحالي تقنيًا
المسار الأكثر مباشرة.** بناءً على ذلك، تم اعتماد سياسة "الاحتفاظ
بـ Vulkan، وإضافة DirectX لـ Windows بالتعايش معه"، وتم تنفيذ
`opencuda-directx`.

## إفصاح صادق

- **الدعم العابر للمنصات لا يزال عملًا قيد التقدم**: صُمم Vulkan
  Compute لدعم أصلي على Windows/Linux/Android ودعم قائم على MoltenVK
  على macOS/iOS، لكن التحقق على عتاد حقيقي تم فقط على هذا الجهاز
  (Windows، NVIDIA GT 730).
- **إرسال نوى `opencuda-directx` يغطي جزئيًا فقط المرحلة 2**: تم
  تنفيذ `vector_add` و`matmul` و`ChaCha20` (التشفير فقط — لا يشمل
  حساب علامة المصادقة Poly1305). كما تم تنفيذ اكتشاف المورّد عبر
  تعداد مهايئات DXGI (`GpuVendor::Nvidia` وما إلى ذلك)، لكن المعلومات
  التفصيلية مثل `compute_capability` تظل نائبًا (placeholder) لأنه لا
  يمكن الحصول عليها من DXGI.
- **الفائدة الحقيقية من الضغط/التشفير على GPU غير مؤكدة**: بالنسبة
  لحمولة صغيرة مثل إطار نفق واحد، هناك قلق تقني من أن النفقات العامة
  لنقل البيانات بين المضيف والجهاز قد تُلغي ميزة الأداء الحسابي لوحدة
  معالجة الرسومات (اختبار الأداء الحقيقي يبقى مهمة مستقبلية).

## العلاقات ضمن هذا النظام البيئي

- [aruaru-llm](https://github.com/aon-co-jp/aruaru-llm) — تنفيذ مرجعي
  لهذا المستودع (مستهلك خط أنابيب تنفيذ GPU/CPU).
- [RS-LinkFusion](https://github.com/aon-co-jp/RS-LinkFusion) — يقيّم
  استخدام تسريع الضغط/التشفير على GPU (نواة ChaCha20 في
  `opencuda-directx`).
- [open-raid-z](https://github.com/aon-co-jp/open-raid-z) — المصدر
  المعتمد لقواعد سياسة التطوير.

## البناء والاختبار

```bash
cargo build --workspace
cargo test --workspace

# اختبارات عتاد حقيقي لـ DirectX 12 (خاص بـ Windows فقط، ميزة real-dx12)
cargo test -p opencuda-directx --features real-dx12
```

### جرّبه على وحدة معالجة الرسومات الخاصة بك (أُضيف في 2026-07-27)

إذا كنت تريد فقط تشغيل شيء واحد أولًا للتحقق من أنه يعمل، فيمكن تشغيل
كل حزمة فرعية ضمن `examples/` (عضو في مساحة العمل) باستخدام
`cargo run -p <الاسم>`. `vulkan_info` على وجه الخصوص مثال بسيط جدًا
يقوم فقط بتعداد وطباعة أجهزة Vulkan الفعلية الحقيقية (اسم مورّد GPU،
سعة VRAM) على جهازك، مما يجعله أفضل أمر أولي للتحقق مما إذا كان يتم
اكتشاف GPU في بيئتك:

```bash
cargo run -p vulkan_info
```

يمكن تشغيل أمثلة أخرى (`matmul`، `matmul_vulkan_real`، `vector_add`،
`vector_add_vulkan`، `vector_add_vulkan_real`، `vector_add_omniir`)
بالمثل باستخدام `cargo run -p <الاسم>`. راجع `OmniGPU-Design.md`
القسم 8.5 لمصفوفة حالة الدعم حسب المورّد (Intel/AMD/NVIDIA وما إلى
ذلك).

## الترخيص

Apache-2.0
