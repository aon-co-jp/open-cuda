# Designphilosophie & Entwicklungsrichtlinien (open-cuda)

> **Hinweis**: Dies ist eine kondensierte Übersetzung des aktuellen
> Zustands. Das ausführliche historische HANDOFF-Änderungsprotokoll
> (Dutzende Einträge seit 2026-06-26) bleibt aus Gründen der Kürze nur
> auf Japanisch in [CLAUDE.md](CLAUDE.md) verfügbar — siehe dort für
> Details zu einzelnen Sitzungen.

Arbeitslaufwerk: `F:\runo`. GitHub-Repo:
[aon-co-jp/open-cuda](https://github.com/aon-co-jp/open-cuda).
Entwicklungsbeginn: 2026-06-26.

## Rolle dieses Projekts

„Das zweite CUDA" — eine GPU-Abstraktions-/Compute-Grundlage (das
`OmniGPU`-Design, siehe `OmniGPU-Design.md`), die Windows/macOS/Linux-
und Intel/AMD/NVIDIA-Kompatibilität anstrebt. Bildet ein „SET" mit
`aruaru-llm`, welches die GPU/CPU-Ausführungspipeline tatsächlich
nutzt.

## Crate-Architektur

- **`opencuda-core`**: gemeinsamer `GpuDevice`-Trait (Äquivalent zur
  CUDA Runtime API).
- **`opencuda-cpu`**: CPU-Backend (Datenparallelität via `rayon`).
- **`opencuda-vulkan`**: Vulkan-Compute-Backend, plattformübergreifend
  (Windows/Linux/Android nativ, macOS/iOS via MoltenVK). GEMM/
  Attention/INT4·INT8-Quantisierung auf echter Hardware verifiziert.
- **`opencuda-directx`**: DirectX-12-Compute-Backend (nur Windows,
  koexistiert mit Vulkan). Kernel-Dispatch für vector_add/matmul/
  ChaCha20/Poly1305 auf echter Hardware (GT 730) verifiziert; DXGI-
  Adapter-Enumeration für Hersteller-/VRAM-Erkennung implementiert.
- **`opencuda-blas`**: NumPy-Äquivalent (GEMM/Attention/Quantisierung/
  Flash Attention/MLA-KV-Kompression).
- **`open-cuda-bert`**: Forward Pass für BERT-Encoder
  (multilingual-e5-small).
- **`open-cuda-llm`**: vLLM-Äquivalent — GPT-2-Decoder mit KV-Cache,
  Repetition Penalty, MLA-Kompression, Flash Attention und (neuestes
  Feature) spekulativer Dekodierung.
- **`open-cuda-whisper`**: Whisper-Äquivalent (Log-Mel-Spektrogramm +
  Encoder + KV-gecachter Decoder + Cross-Attention), aktuell nur ein
  zufallsinitialisiertes MVP.

## Ehrlicher Status der Offenlegung

- **cuBLAS/rocBLAS/oneMKL bleiben unverifizierte Stubs** — diese
  Maschine hat keine CUDA/ROCm/oneAPI-Toolchain zur Verifikation.
- **FP8 wurde abgelehnt**: die einzige GPU dieser Maschine (GT730,
  Kepler, CC 3.5) besitzt keine FP8-Tensor-Cores — eine Implementierung
  wäre reine Software-Emulation ohne echten Nutzen.
- **DeepSeekMoE wurde abgelehnt**: `DecoderLayer` hat nur ein einziges
  dichtes FFN (keine Experten-/Router-Struktur) und es existiert kein
  echter MoE-Checkpoint.
- **GPU-Kompression/-Verschlüsselung**: der reale Nutzen bei kleinen
  Payloads ist unverifiziert (Host↔Device-Overhead könnte den
  GPU-Vorteil zunichtemachen).
- **Android-Realgerätebefund (2026-08-15)**: Auf einem Android-Gerät
  (moto g53y 5G, Adreno 619) übertraf die Telefon-GPU bei größeren
  Matrixgrößen (bis zu ~6× bei 512×512) sowohl CPU als auch die
  Desktop-GT730-GPU-Vergleiche und bestätigte damit die Hypothese
  „Telefon-GPUs könnten überraschend schnell sein". Vorbehalt: einmalige
  kleine Berechnungen bleiben weiterhin durch den GPU-Initialisierungs-
  Overhead begrenzt (58–63ms).

## Hersteller-Support-Matrix

Drei Schichten: die funktionierende Vulkan-Integration / die
`GpuVendor`-Aufzählung als reine Meldeschicht (inkl. unverifiziertem
Qualcomm/ARM/ImaginationPowerVr) / die Stub-Schicht der
Hersteller-Bibliotheken. Details siehe `OmniGPU-Design.md` §8.5.

## Jüngste relevante HANDOFF-Einträge

- **2026-08-10**: `generate_with_repetition_penalty` (CTRL-artige
  Wiederholungsstrafe) behebt einen realen Endlosschleifen-Bug in
  `aruaru-llm`s greedy GPT-2-Decodierung. Standardwert `penalty=1.3`
  via `ARUARU_LLM_REPETITION_PENALTY`.
- **Neueste Arbeit (Commit `0c43ba3`)**: Neues
  `GptModel::generate_speculative` — verlustfreie spekulative
  Dekodierung im DSpark/Leviathan-Stil (Draft-Modell schlägt Tokens
  vor, Ziel-Modell verifiziert per gebündeltem Batch-Prefill). Auf
  synthetischen Fixtures und echten Gewichten als bitidentisch zu
  `generate()` verifiziert. **Ehrliche Offenlegung**: auf dem CPU-Pfad
  dieser Maschine langsamer als einfaches `generate()` (draft_k=4:
  plain 4,63s vs. spekulativ 7,65s), da naive CPU-GEMM kaum
  Dispatch-Overhead zum Amortisieren bietet. Der eigentliche
  Ziel-Fall — Geschwindigkeit auf echtem Vulkan — ist noch nicht
  gemessen.
- **2026-08-19 — Auto-Update**: Untersuchung eines automatischen
  Update-Mechanismus (nach dem Vorbild von `open-english`s
  `self_update.rs`) — verworfen: dieses Repo besteht nur aus
  Bibliotheks-Crates und Wegwerf-Beispielbinaries, kein residenter
  Dienst. **Widerlegung eines Einwands**: das Fehlen eines residenten
  Dienstes ist kein Mangel — das echte Microsoft DirectX/NVIDIA CUDA
  arbeitet selbst als Laufzeitbibliotheken, die von jedem Prozess
  gelinkt werden, nicht als Hintergrunddienste (`nvidia-persistenced`
  ist eine begrenzte Ausnahme zur Zwischenspeicherung des
  GPU-Initialisierungszustands, kein Schiedsrichter zwischen Prozessen).
- **2026-08-20 — Korrektur**: die frühere Notiz über eine „in
  Entwicklung befindliche Verbindung" zwischen `open-directx` und
  `open-cuda` war ungenau. Tatsächlich handelt es sich um zwei
  gleichnamige, unabhängige Projekte: (1) das interne
  `opencuda-directx`-Crate dieses Repos, und (2) das eigenständige
  Repository `aon-co-jp/open-directx`.
- **2026-08-20 — FlexQ-ähnliche INT6-Quantisierung**: Hinzufügen von
  `quantize_int6`/`dequantize_int6`/`QuantizedInt6Tensor` in
  `opencuda-blas`, das 4 6-Bit-Werte in 3 Bytes packt. PuzzleMoE wurde
  verworfen, da in den Modellen dieses Repos keine MoE-Architektur
  existiert (nur ein einzelnes dichtes FFN).
