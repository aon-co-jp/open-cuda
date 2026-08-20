# פילוסופיית עיצוב ומדיניות פיתוח (open-cuda)

> **הערה**: זהו תרגום מקוצר של המצב הנוכחי. יומן השינויים ההיסטורי
> המלא של HANDOFF (עשרות רשומות מאז 2026-06-26) נשאר זמין רק ביפנית
> ב-[CLAUDE.md](CLAUDE.md) לשם תמציתיות — עיינו שם לפרטים על כל סשן.

כונן העבודה: `F:\runo`. מאגר GitHub:
[aon-co-jp/open-cuda](https://github.com/aon-co-jp/open-cuda). תחילת
הפיתוח: 2026-06-26.

## תפקיד הפרויקט הזה

"ה-CUDA השני" — בסיס הפשטה/חישוב GPU (עיצוב `OmniGPU`, ראו
`OmniGPU-Design.md`) השואף לתאימות עם Windows/macOS/Linux ו-
Intel/AMD/NVIDIA. יוצר "SET" עם `aruaru-llm`, שהוא הצרכן האמיתי של
צינור הביצוע GPU/CPU.

## ארכיטקטורת ה-crates

- **`opencuda-core`**: תכונת (trait) `GpuDevice` משותפת (מקבילה
  ל-CUDA Runtime API).
- **`opencuda-cpu`**: backend מבוסס CPU (מקביליות נתונים דרך `rayon`).
- **`opencuda-vulkan`**: backend Vulkan Compute חוצה-פלטפורמות
  (ילידי ב-Windows/Linux/Android, macOS/iOS דרך MoltenVK). GEMM/
  Attention/כימות INT4·INT8 אומתו על חומרה אמיתית.
- **`opencuda-directx`**: backend מבוסס DirectX 12 Compute (Windows
  בלבד, קיים במקביל ל-Vulkan). דיספאצ'ינג הגרעינים vector_add/matmul/
  ChaCha20/Poly1305 אומת על חומרה אמיתית (GT 730); מיפוי מתאמי DXGI
  לזיהוי יצרן/VRAM מיושם.
- **`opencuda-blas`**: מקביל ל-NumPy (GEMM/Attention/כימות/
  Flash Attention/דחיסת KV מסוג MLA).
- **`open-cuda-bert`**: מעבר קדימה עבור מקודדי BERT
  (multilingual-e5-small).
- **`open-cuda-llm`**: מקביל ל-vLLM — מפענח GPT-2 עם מטמון KV, עונש
  חזרתיות, דחיסת MLA, flash attention ו(התכונה החדשה ביותר) פענוח
  ספקולטיבי.
- **`open-cuda-whisper`**: מקביל ל-Whisper (לוג-mel-ספקטרוגרם + מקודד
  + מפענח עם מטמון KV + cross-attention), כרגע רק MVP עם אתחול אקראי.

## מצב גילוי כן

- **cuBLAS/rocBLAS/oneMKL נותרים כ-stubs שלא אומתו** — למכונה זו אין
  שרשרת כלים של CUDA/ROCm/oneAPI לאימות.
- **FP8 נדחה**: ה-GPU היחיד של מכונה זו (GT730, Kepler, CC 3.5) אינו
  כולל Tensor Cores של FP8 — יישום היה רק אמולציית תוכנה ללא תועלת
  אמיתית.
- **DeepSeekMoE נדחה**: ל-`DecoderLayer` יש רק FFN צפוף בודד (ללא
  מבנה מומחים/מנתב), ואין נקודת ביקורת MoE אמיתית.
- **דחיסה/הצפנה על GPU**: התועלת האמיתית עבור מטענים קטנים אינה
  מאומתת (התקורה בין המארח למכשיר עלולה לבטל את יתרון ה-GPU).
- **ממצא ממכשיר Android אמיתי (2026-08-15)**: על מכשיר Android
  (moto g53y 5G, Adreno 619), ה-GPU של הטלפון עלה הן על ה-CPU והן על
  ההשוואה CPU/GPU של GT730 השולחני בגדלי מטריצות גדולים יותר (עד פי
  ~6 ב-512×512), מה שאישש את ההשערה ש"GPUs של טלפונים עשויים להיות
  מהירים באופן מפתיע". הסתייגות: חישובים בודדים קטנים עדיין מוגבלים
  על ידי תקורת האתחול של ה-GPU (58–63 מ"ש).

## מטריצת תמיכת יצרנים

שלוש שכבות: האינטגרציה הפועלת של Vulkan / המספר (enum) `GpuVendor`
כשכבת דיווח בלבד (כולל Qualcomm/ARM/ImaginationPowerVr שלא אומתו) /
שכבת ה-stub של ספריות היצרנים. פרטים ב-`OmniGPU-Design.md` §8.5.

## רשומות HANDOFF אחרונות רלוונטיות

- **2026-08-10**: `generate_with_repetition_penalty` (עונש חזרתיות
  בסגנון CTRL) פותר באג אמיתי של לולאה אינסופית בפענוח החמדני של
  GPT-2 ב-`aruaru-llm`. ברירת מחדל `penalty=1.3` דרך
  `ARUARU_LLM_REPETITION_PENALTY`.
- **העבודה האחרונה ביותר (commit `0c43ba3`)**: פונקציה חדשה
  `GptModel::generate_speculative` — פענוח ספקולטיבי חסר-אובדן בסגנון
  DSpark/Leviathan (מודל טיוטה מציע טוקנים, מודל היעד מאמת דרך
  batch-prefill מרוכז). אומת כזהה סיבית-לביט ל-`generate()` על
  fixtures סינתטיים ומשקלים אמיתיים. **גילוי כן**: בנתיב ה-CPU של
  מכונה זו איטי יותר מ-`generate()` הרגיל (draft_k=4: רגיל 4.63
  שניות מול ספקולטיבי 7.65 שניות), מכיוון של-CPU GEMM נאיבי יש מעט
  תקורת דיספאצ'ינג לפזר עליה. מקרה היעד האמיתי — מהירות על Vulkan
  אמיתי — עדיין לא נמדד.
- **2026-08-19 — עדכון אוטומטי**: נבדקה הטמעת מנגנון עדכון אוטומטי
  (בדומה ל-`self_update.rs` של `open-english`) — נדחתה: המאגר הזה
  מכיל רק ארגזי ספרייה ובינארי דוגמאות חד-פעמיים, אין שירות שוכן.
  **הפרכת הערה**: היעדר שירות שוכן אינו פגם — ה-DirectX/CUDA
  האמיתיים של Microsoft/NVIDIA פועלים בעצמם כספריות זמן ריצה
  המקושרות על ידי כל תהליך, לא כשירותי רקע (`nvidia-persistenced`
  הוא חריג מוגבל למטמון מצב אתחול ה-GPU, לא בורר בין תהליכים).
- **2026-08-20 — תיקון**: הרשומה הקודמת בדבר "קישור בשלב עיצוב" בין
  `open-directx` ל-`open-cuda` הייתה לא מדויקת. למעשה מדובר בשני
  פרויקטים בלתי קשורים בעלי שם זהה: (1) ה-crate המובנה
  `opencuda-directx` של מאגר זה, ו-(2) המאגר העצמאי
  `aon-co-jp/open-directx`.
- **2026-08-20 — קוונטיזציית INT6 בסגנון FlexQ**: נוספו
  `quantize_int6`/`dequantize_int6`/`QuantizedInt6Tensor` ל-
  `opencuda-blas`, האורזים 4 ערכי 6-סיביות ל-3 בייטים. PuzzleMoE
  נדחה מכיוון שאין ארכיטקטורת MoE במודלים של מאגר זה (FFN צפוף יחיד
  בלבד).
