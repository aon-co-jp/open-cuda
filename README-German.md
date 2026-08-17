# open-cuda

*English*: [README-English.md](README-English.md) ·
*Andere Sprachen*: [Deutsch](README-German.md) · [Italiano](README-Italian.md) ·
[Français](README-French.md) · [Русский](README-Russian.md) ·
[Українська](README-Ukrainian.md) · [עברית](README-Hebrew.md) ·
[فارسی](README-Persian.md) · [العربية](README-Arabic.md)

> **Letztes Update (2026-08-10)**: `generate_with_repetition_penalty`
> (Wiederholungsstrafe im CTRL-Stil — penalty>1.0 schwächt die Logits
> bereits erschienener Tokens ab) wurde zu `open-cuda-llm::GptModel`
> hinzugefügt. Das bestehende `generate()` ist jetzt ein dünner Wrapper,
> der `penalty=1.0` aufruft (byteidentisches Verhalten, keine Regression
> für bestehende Tests/Aufrufer). Dies behebt direkt einen gemeldeten
> `aruaru-llm`-Fehler, bei dem das Basis-GPT-2 (ohne Dialog-Finetuning)
> beim gierigen Decodieren denselben String endlos wiederholt (z. B.
> "Student: Hello"). Ein neuer Test mit echten GPT-2-124M-Gewichten,
> `repetition_penalty_reduces_degenerate_loop_on_real_gpt2_weights`,
> bestätigt, dass die Schleife ohne Strafe tatsächlich auftritt und bei
> `penalty=1.3` tatsächlich endet (grammatikalisch natürlicher Text
> entsteht). `/v1/generate` von `aruaru-llm` ruft diese neue API jetzt
> standardmäßig mit `penalty=1.3` auf (überschreibbar via
> `ARUARU_LLM_REPETITION_PENALTY`). Siehe den HANDOFF-Eintrag vom
> 2026-08-10 in [CLAUDE.md](CLAUDE.md).

> **Letztes Update (2026-08-08)**: Der Hinweis "MLA implementiert" vom
> 2026-08-06 wurde auf echter Hardware erneut verifiziert
> (`cargo test -p opencuda-blas mla -- --nocapture` → `1 passed; 0
> failed`, durchlief den echten Vulkan-Pfad der GT730). FP8-Mixed-Precision
> und DeepSeekMoE wurden untersucht, **eine Implementierung wurde jedoch
> in beiden Fällen abgelehnt**: FP8 hat auf der einzigen GPU dieser
> Maschine (GT730, Kepler, CC 3.5 — keine FP8-Tensor-Cores) keine echte
> Hardware-Unterstützung und wäre nur Software-Emulation; DeepSeekMoE hat
> keinen echten Integrationspunkt, da `DecoderLayer` von `open-cuda-llm`
> nur ein einziges dichtes FFN besitzt (keine Experten-/Router-Struktur)
> und kein echter MoE-Checkpoint existiert. Details siehe HANDOFF-Eintrag
> vom 2026-08-08 in [CLAUDE.md](CLAUDE.md).

> **Letztes Update (2026-08-07)**: Der fusionierte Flash-Attention-Kernel
> (`flash_attention_with_spirv`) wurde in `DecoderLayer` von
> `open-cuda-llm` verdrahtet, via `GptModel::set_flash_attention_spirv()`
> — fällt bei fehlender Einstellung auf den bestehenden 3-Dispatch-Pfad
> zurück (vollständig rückwärtskompatibel). Auf echter NVIDIA-GT-730-
> Hardware verifiziert: Vulkan-generierte Token-Sequenzen sind
> byteidentisch zum CPU-Pfad. Details siehe HANDOFF in
> [CLAUDE.md](CLAUDE.md).

> **Implementiert am 2026-08-06**: Low-Rank-KV-Cache-Kompression,
> inspiriert von DeepSeek-V3s Multi-Head Latent Attention (MLA) —
> `opencuda-blas::mla_compress_kv`/`mla_decompress_kv`. Nach Recherche
> des technischen Berichts
> ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437)) und von
> Implementierungs-Blogs auf Japanisch und Englisch wurde der
> Low-Rank-Projektionsmechanismus (das Design hinter DeepSeeks berichteter
> KV-Cache-Reduktion um 93,3 %) auf dem bestehenden, hardwareverifizierten
> `sgemm`-Backend aufgebaut. Auf echter Hardware (GT730) verifiziert, dass
> CPU- und Vulkan-Pfad numerisch übereinstimmen. Enthält keine trainierten
> Gewichte (dies demonstriert den Mechanismus, nicht die trainierte
> Kompressionsqualität — ehrliche Offenlegung siehe HANDOFF-Abschnitt von
> `CLAUDE.md`). Die Anwendung von Toshiba-SBM-/DeepSeek-Techniken auf die
> übrigen 7 Repos wird noch geprüft.

> **Aktualisiert am 2026-07-25**: Die Überschrift der Entwicklungsrichtlinien-
> Datei (`CLAUDE.md`) wurde von „Entwicklungsrichtlinien & Regeln der
> Entwicklungsumgebung" zu „Designphilosophie & Entwicklungsrichtlinien &
> Regeln der Entwicklungsumgebung" umbenannt, um Designphilosophie (was
> wir wertschätzen), Entwicklungsrichtlinien (wie wir arbeiten) und
> Umgebungsregeln (konkrete operative Konventionen) klarer zu trennen.
> Details siehe `CLAUDE.md`.

**Entwicklungsbeginn: 2026-06-26** (GitHub-Erstellungsdatum dieses Repos)

„Das zweite CUDA" — eine GPU-Abstraktions- und Compute-Grundlage (das
`OmniGPU`-Design), die Windows-/macOS-/Linux-Kompatibilität sowie
Intel-/AMD-/NVIDIA-Kompatibilität anstrebt. Bildet zusammen mit
`aruaru-llm` ein „SET" (das die GPU/CPU-Ausführungspipeline tatsächlich
nutzt).

## Was das ist

- **`opencuda-core`**: Der von allen Backends gemeinsam genutzte
  `GpuDevice`-Trait (Äquivalent zur CUDA-Runtime-API: `alloc`/`memcpy`/
  `launch_kernel`).
- **`opencuda-cpu`**: Das CPU-Backend (Datenparallelität via `rayon`).
- **`opencuda-vulkan`**: Das Vulkan-Compute-Backend (plattformübergreifend,
  nativ auf Windows/Linux/Android, macOS/iOS via MoltenVK). GEMM/
  Attention/INT4·INT8-Quantisierung auf echter Vulkan-Ausführung
  verifiziert.
- **`opencuda-directx`** (hinzugefügt 2026-07-23): Ein DirectX-12-
  Compute-Backend (nur Windows, optional, koexistiert mit Vulkan).
  GPU-Dispatch von `vector_add`/`matmul`/`ChaCha20` auf echter Hardware
  (NVIDIA GT 730) verifiziert — Ausgabe stimmt in Tests exakt mit
  CPU-Referenzimplementierungen überein (z. B. der RustCrypto-Crate
  `chacha20`). Auch die Ermittlung von echtem Herstellernamen/
  VRAM-Kapazität via DXGI-Adapter-Enumeration ist implementiert.
- **`opencuda-blas`**: NumPy-Äquivalent (GEMM/Attention/Quantisierung).
- **`open-cuda-bert`**: Forward Pass für BERT-artige Encoder (unterstützt
  multilingual-e5-small).
- **`open-cuda-llm`**: vLLM-Äquivalent (gieriges Decodieren mit
  KV-Cache). Implementiert `GptModel::load`, das GPT-2 (Hugging Face
  `openai-community/gpt2`) `safetensors` lädt (2026-07-25, gleiches
  Design wie `open-cuda-bert::BertModel::load`). Auf echter Hardware
  verifiziert: Herunterladen und Laden echter GPT-2-124M-Gewichte
  erzeugt deutlich flüssigeres, gierig decodiertes Englisch als
  Zufallsinitialisierung (bedeutungsloser Output) — z. B. "The quick
  brown fox" → "es are a great way to get a little bit of a". Details
  siehe HANDOFF in `CLAUDE.md`.
- **`open-cuda-whisper`** (hinzugefügt 2026-07-31): Whisper-Äquivalent
  (Spracherkennung, Platz 6 der Marktrecherche-Roadmap). Log-Mel-
  Spektrogramm-Extraktion + Encoder (gleiches Multi-Head-Attention-Design
  wie `open-cuda-bert`) + KV-gecachter Decoder (gleiches Design wie
  `open-cuda-llm`) + Cross-Attention. **Aktuell nur ein
  zufallsinitialisiertes MVP** (noch kein Lader für trainierte
  Whisper-Gewichte — Details siehe HANDOFF in `CLAUDE.md`).

## Warum sowohl DirectX als auch Vulkan (technische Entscheidung vom
2026-07-23)

Zunächst wurde angenommen, dieses Projekt sei „als DirectX-Plugin in
Entwicklung". Nach Rückversicherung durch japanische/englische
Web-Recherche stellte sich heraus, dass DXVK/vkd3d-proton (die
Technologie, die Valves Proton tatsächlich nutzt) beide nur in Richtung
„DirectX (nur Windows) → Vulkan (plattformübergreifend)" konvertieren —
Beispiele für die umgekehrte Richtung wurden nicht gefunden. **Für das
Ziel der Plattformübergreifenheit ist der bestehende Vulkan-Compute-Ansatz
technisch der direktere Weg.** Auf dieser Grundlage wurde die Richtlinie
„Vulkan behalten und zusätzlich DirectX für Windows koexistierend
hinzufügen" gewählt und `opencuda-directx` implementiert.

## Ehrliche Offenlegung

- **Plattformübergreifende Unterstützung ist noch auf halbem Weg**:
  Vulkan Compute ist so ausgelegt, dass es auf Windows/Linux/Android
  nativ und auf macOS/iOS via MoltenVK läuft, aber die Verifizierung auf
  echter Hardware wurde bisher nur auf dieser Maschine (Windows, NVIDIA
  GT 730) durchgeführt.
- **Der Kernel-Dispatch von `opencuda-directx` deckt nur einen Teil von
  Phase 2 ab**: `vector_add`, `matmul` und `ChaCha20` (nur Verschlüsselung
  — ohne Poly1305-Authentifizierungs-Tag) sind implementiert. Die
  Herstellererkennung via DXGI-Adapter-Enumeration (`GpuVendor::Nvidia`
  usw.) ist ebenfalls implementiert, aber Detailinformationen wie
  `compute_capability` bleiben ein Platzhalter, da DXGI sie nicht liefert.
- **Der reale Nutzen von GPU-Kompression/-Verschlüsselung ist unverifiziert**:
  Bei einer so kleinen Payload wie einem Tunnel-Frame besteht die
  technische Sorge, dass der Übertragungs-Overhead zwischen Host und
  Device den Rechenvorteil der GPU zunichtemachen könnte (echtes
  Benchmarking bleibt eine zukünftige Aufgabe).

## Zusammenhänge in diesem Ökosystem

- [aruaru-llm](https://github.com/aon-co-jp/aruaru-llm) — eine
  Referenzimplementierung dieses Repos (Nutzer der GPU/CPU-
  Ausführungspipeline).
- [RS-LinkFusion](https://github.com/aon-co-jp/RS-LinkFusion) — prüft
  den Einsatz von GPU-Kompressions-/Verschlüsselungsbeschleunigung
  (den ChaCha20-Kernel in `opencuda-directx`).
- [open-raid-z](https://github.com/aon-co-jp/open-raid-z) — die
  kanonische Quelle für Entwicklungsrichtlinien.

## Build & Test

```bash
cargo build --workspace
cargo test --workspace

# DirectX-12-Hardwaretests (nur Windows, real-dx12 feature)
cargo test -p opencuda-directx --features real-dx12
```

### Selbst auf der eigenen GPU testen (hinzugefügt 2026-07-27)

Wer zunächst nur etwas ausprobieren möchte, um zu prüfen, ob es
funktioniert, kann jedes Unter-Crate unter `examples/` (ein
Workspace-Mitglied) mit `cargo run -p <Name>` ausführen. `vulkan_info`
ist ein minimales Beispiel, das lediglich die echten physischen
Vulkan-Geräte (Hersteller, VRAM-Kapazität) auf der eigenen Maschine
auflistet und ausgibt — der beste erste Befehl, um zu prüfen, ob eine
GPU in der eigenen Umgebung erkannt wird:

```bash
cargo run -p vulkan_info
```

Weitere Beispiele (`matmul`, `matmul_vulkan_real`, `vector_add`,
`vector_add_vulkan`, `vector_add_vulkan_real`, `vector_add_omniir`)
lassen sich ebenso mit `cargo run -p <Name>` ausführen. Die
Unterstützungsmatrix je Hersteller (Intel/AMD/NVIDIA usw.) findet sich
in `OmniGPU-Design.md` §8.5.

## Lizenz

Apache-2.0
