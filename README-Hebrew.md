# open-cuda

*English*: [README-English.md](README-English.md) ·
*שפות נוספות*: [Deutsch](README-German.md) · [Italiano](README-Italian.md) ·
[Français](README-French.md) · [Русский](README-Russian.md) ·
[Українська](README-Ukrainian.md) · [עברית](README-Hebrew.md) ·
[فارسی](README-Persian.md) · [العربية](README-Arabic.md)

> **עדכון אחרון (2026-08-10)**: נוספה הפונקציה
> `generate_with_repetition_penalty` (עונש חזרתיות בסגנון CTRL —
> penalty>1.0 מחליש את הלוגיטים של טוקנים שכבר הופיעו) ל-
> `open-cuda-llm::GptModel`. הפונקציה הקיימת `generate()` היא כעת עטיפה
> דקה שקוראת לה עם `penalty=1.0` (התנהגות זהה בית לבית, ללא נסיגה
> בבדיקות/קריאות קיימות). זה פותר ישירות באג מדווח ב-`aruaru-llm` שבו
> פענוח חמדני (greedy) של GPT-2 הבסיסי (ללא כוונון-עדין לדיאלוג) חוזר
> באופן אינסופי על אותו מחרוזת (למשל "Student: Hello"). נוספה בדיקה
> חדשה על משקלים אמיתיים של GPT-2 124M,
> `repetition_penalty_reduces_degenerate_loop_on_real_gpt2_weights`,
> המאשרת שהלולאה אכן משוחזרת ללא עונש, ואכן נעצרת (ומייצרת טקסט טבעי
> מבחינה דקדוקית) עם `penalty=1.3`. `/v1/generate` של `aruaru-llm` קורא
> כעת ל-API החדש הזה עם `penalty=1.3` כברירת מחדל (ניתן לשינוי דרך
> `ARUARU_LLM_REPETITION_PENALTY`). ראו את רשומת ה-HANDOFF מ-2026-08-10
> ב-[CLAUDE.md](CLAUDE.md).

> **עדכון אחרון (2026-08-08)**: אומתה מחדש על חומרה אמיתית ההערה
> "MLA מיושם" מ-2026-08-06 (`cargo test -p opencuda-blas mla --
> --nocapture` → `1 passed; 0 failed`, עברה בנתיב Vulkan האמיתי של
> ה-GT730). נחקרו FP8 בדיוק מעורב ו-DeepSeekMoE, אך **הוחלט שלא לממש
> אף אחד מהם**: ל-FP8 אין תמיכת חומרה אמיתית ב-GPU היחיד של מכונה זו
> (GT730, Kepler, CC 3.5 — ללא Tensor Cores של FP8, זו הייתה רק
> אמולציית תוכנה); ל-DeepSeekMoE אין נקודת שילוב אמיתית מכיוון של-
> `DecoderLayer` ב-`open-cuda-llm` יש רק FFN צפוף בודד (ללא מבנה
> מומחים/מנתב), ואין נקודת ביקורת MoE אמיתית. לפרטים ראו את רשומת
> ה-HANDOFF מ-2026-08-08 ב-[CLAUDE.md](CLAUDE.md).

> **עדכון אחרון (2026-08-07)**: גרעין ה-flash-attention המאוחד
> (`flash_attention_with_spirv`) חובר ל-`DecoderLayer` של
> `open-cuda-llm`, דרך `GptModel::set_flash_attention_spirv()` — חוזר
> לנתיב הקיים של 3 דיספאצ'ים כאשר לא מוגדר (תואם לאחור באופן מלא).
> אומת על חומרה אמיתית של NVIDIA GT 730: רצפי הטוקנים שנוצרו דרך Vulkan
> זהים בית לבית לנתיב ה-CPU. לפרטים ראו HANDOFF ב-[CLAUDE.md](CLAUDE.md).

> **מומש ב-2026-08-06**: דחיסת מטמון KV בדרגה נמוכה, בהשראת
> Multi-Head Latent Attention (MLA) של DeepSeek-V3 —
> `opencuda-blas::mla_compress_kv`/`mla_decompress_kv`. לאחר חקירת
> הדוח הטכני ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437))
> ובלוגי יישום ביפנית ובאנגלית, נבנה מנגנון ההיטל בדרגה נמוכה (העיצוב
> שמאחורי צמצום מטמון ה-KV בשיעור 93.3% שדווח על ידי DeepSeek) מעל
> בסיס `sgemm` הקיים שכבר אומת על חומרה אמיתית. אומת על חומרה אמיתית
> (GT730) שהנתיבים של CPU ו-Vulkan תואמים מספרית. אינו נושא משקלים
> מאומנים (זה מדגים את המנגנון, לא את איכות הדחיסה המאומנת — ראו את
> סעיף ה-HANDOFF של `CLAUDE.md` לגילוי הכן). יישום טכניקות
> Toshiba SBM / DeepSeek ל-7 המאגרים הנוספים עדיין נשקל.

> **עודכן ב-2026-07-25**: כותרת קובץ מדיניות הפיתוח (`CLAUDE.md`)
> שונתה מ"מדיניות פיתוח וכללי סביבת פיתוח" ל"פילוסופיית עיצוב ומדיניות
> פיתוח וכללי סביבת פיתוח", כדי להפריד בבירור רבה יותר בין פילוסופיית
> העיצוב של הפרויקט (מה אנו מעריכים), מדיניות הפיתוח (איך אנו עובדים)
> וכללי סביבת הפיתוח (מוסכמות תפעוליות קונקרטיות). לפרטים ראו
> `CLAUDE.md`.

**תחילת הפיתוח: 2026-06-26** (תאריך יצירת מאגר GitHub זה)

"ה-CUDA השני" — בסיס הפשטה/חישוב GPU (עיצוב `OmniGPU`) השואף לתאימות
Windows/macOS/Linux ולתאימות Intel/AMD/NVIDIA. מותאם ("SET") עם
`aruaru-llm`, שהוא הצרכן-יישום של צינור הביצוע GPU/CPU.

## מה זה

- **`opencuda-core`**: התכונה (trait) `GpuDevice` המשותפת לכל
  ה-backends (מקבילה ל-CUDA Runtime API: `alloc`/`memcpy`/
  `launch_kernel`).
- **`opencuda-cpu`**: backend מבוסס CPU (מקביליות נתונים דרך `rayon`).
- **`opencuda-vulkan`**: backend מבוסס Vulkan Compute (חוצה-פלטפורמות,
  ילידי ב-Windows/Linux/Android, macOS/iOS דרך MoltenVK). GEMM/
  Attention/כימות INT4·INT8 אומתו על ריצת Vulkan אמיתית.
- **`opencuda-directx`** (נוסף ב-2026-07-23): backend מבוסס
  DirectX 12 Compute (Windows בלבד, backend אופציונלי שקיים במקביל
  ל-Vulkan). דיספאצ'ינג GPU של `vector_add`/`matmul`/`ChaCha20` אומת
  על חומרה אמיתית (NVIDIA GT 730) — הפלט תואם במדויק ליישומי הייחוס
  של CPU (למשל ה-crate `chacha20` של RustCrypto) בבדיקות. גם קבלת שם
  היצרן/קיבולת ה-VRAM האמיתיים דרך מיפוי מתאמי DXGI מיושמת.
- **`opencuda-blas`**: מקביל ל-NumPy (GEMM/Attention/כימות).
- **`open-cuda-bert`**: מעבר קדימה (forward pass) עבור מקודדי משפחת
  BERT (תומך ב-multilingual-e5-small).
- **`open-cuda-llm`**: מקביל ל-vLLM (פענוח חמדני עם מטמון KV). מיישם
  את `GptModel::load`, הטוען `safetensors` של GPT-2 (Hugging Face
  `openai-community/gpt2`) (2026-07-25, אותו עיצוב כמו
  `open-cuda-bert::BertModel::load`). אומת על חומרה אמיתית: הורדה
  וטעינה של משקלים אמיתיים של GPT-2 124M מייצרת אנגלית מפוענחת-חמדנית
  שוטפת בבירור יותר מאתחול אקראי (פלט חסר משמעות) — למשל "The quick
  brown fox" → "es are a great way to get a little bit of a". לפרטים
  ראו את ה-HANDOFF של `CLAUDE.md`.
- **`open-cuda-whisper`** (נוסף ב-2026-07-31): מקביל ל-Whisper (זיהוי
  דיבור, מקום #6 במפת הדרכים של מחקר השוק). חילוץ לוג-mel-ספקטרוגרם +
  מקודד (אותו עיצוב Multi-Head Attention כמו `open-cuda-bert`) + מפענח
  עם מטמון KV (אותו עיצוב כמו `open-cuda-llm`) + cross-attention.
  **כרגע רק MVP עם אתחול אקראי** (טוען משקלי Whisper מאומנים עדיין לא
  קיים — לפרטים ראו HANDOFF של `CLAUDE.md`).

## מדוע יש לנו גם DirectX וגם Vulkan (החלטה טכנית מ-2026-07-23)

בתחילה, הפרויקט הזה נחשב כ"בפיתוח כתוסף DirectX". לאחר אימות זאת
באמצעות חיפוש ברשת ביפנית/אנגלית, התברר ש-DXVK/vkd3d-proton (הטכנולוגיה
ש-Proton של Valve אכן משתמשת בה) שניהם ממירים בכיוון "DirectX (API
ל-Windows בלבד) → Vulkan (API חוצה-פלטפורמות)" — לא נמצאו דוגמאות
לכיוון ההפוך. **למטרת תמיכה חוצה-פלטפורמות, גישת Vulkan Compute
הקיימת היא מבחינה טכנית הדרך הישירה יותר.** על בסיס זה, אומצה המדיניות
"לשמור על Vulkan, ולהוסיף DirectX עבור Windows בנוסף, בקיום משותף",
ו-`opencuda-directx` יושם.

## גילוי כן

- **התמיכה חוצה-הפלטפורמות היא עבודה בתהליך**: Vulkan Compute מעוצב
  לתמיכה ילידית ב-Windows/Linux/Android ותמיכה מבוססת MoltenVK
  ב-macOS/iOS, אך אימות על חומרה אמיתית בוצע רק על מכונה זו (Windows,
  NVIDIA GT 730).
- **דיספאצ'ינג הגרעינים של `opencuda-directx` מכסה רק חלקית את
  שלב 2**: `vector_add`, `matmul` ו-`ChaCha20` (הצפנה בלבד — לא כולל
  חישוב תג האימות Poly1305) מיושמים. זיהוי יצרן דרך מיפוי מתאמי DXGI
  (`GpuVendor::Nvidia` וכו') מיושם גם כן, אך מידע מפורט כמו
  `compute_capability` נשאר placeholder מכיוון שלא ניתן להשגה מ-DXGI.
- **התועלת האמיתית מדחיסה/הצפנה על ה-GPU אינה מאומתת**: עבור מטען
  קטן כמו פריים מנהרה בודד, קיים חשש טכני שתקורת ההעברה בין המארח
  למכשיר עלולה לבטל את יתרון החישוב של ה-GPU (בדיקות ביצועים אמיתיות
  נותרות משימה עתידית).

## קשרים בתוך המערכת האקולוגית הזו

- [aruaru-llm](https://github.com/aon-co-jp/aruaru-llm) — יישום ייחוס
  של מאגר זה (צרכן של צינור הביצוע GPU/CPU).
- [RS-LinkFusion](https://github.com/aon-co-jp/RS-LinkFusion) — בוחן
  שימוש בהאצת דחיסה/הצפנה על GPU (גרעין ה-ChaCha20 ב-`opencuda-directx`).
- [open-raid-z](https://github.com/aon-co-jp/open-raid-z) — המקור
  הקנוני לכללי מדיניות הפיתוח.

## בנייה ובדיקה

```bash
cargo build --workspace
cargo test --workspace

# בדיקות חומרה אמיתית של DirectX 12 (Windows בלבד, פיצ'ר real-dx12)
cargo test -p opencuda-directx --features real-dx12
```

### נסו זאת על ה-GPU שלכם (נוסף ב-2026-07-27)

אם אתם פשוט רוצים להריץ קודם דבר אחד כדי לבדוק שהוא עובד, ניתן להריץ
כל תת-crate תחת `examples/` (חבר במרחב העבודה) עם `cargo run -p <שם>`.
`vulkan_info` בפרט הוא דוגמה מינימלית שרק ממפה ומדפיסה את התקני
ה-Vulkan הפיזיים האמיתיים (שם יצרן GPU, קיבולת VRAM) על המכונה שלכם,
מה שהופך אותו לפקודה הראשונה הטובה ביותר לבדיקה האם GPU מזוהה בסביבה
שלכם:

```bash
cargo run -p vulkan_info
```

דוגמאות נוספות (`matmul`, `matmul_vulkan_real`, `vector_add`,
`vector_add_vulkan`, `vector_add_vulkan_real`, `vector_add_omniir`)
ניתן להריץ באותו אופן עם `cargo run -p <שם>`. ראו `OmniGPU-Design.md`
§8.5 למטריצת מצב התמיכה לפי יצרן (Intel/AMD/NVIDIA וכו').

## רישיון

Apache-2.0
