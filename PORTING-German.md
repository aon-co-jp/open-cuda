# open-cuda — Portierungsleitfaden (Kurzfassung)

> **Hinweis**: Dies ist eine kondensierte Übersetzung der aktuell
> wiederverwendbaren Muster. Das ausführliche historische
> HANDOFF-Änderungsprotokoll bleibt nur auf Japanisch in
> [PORTING.md](PORTING.md) verfügbar — dort nachschlagen, bevor ein
> Muster tatsächlich übernommen wird.

Zusammenfassung der wiederverwendbaren Design-Muster aus diesem
Projekt, falls sie in ein anderes Projekt portiert werden sollen.

1. **`GpuDevice`-Trait (Backend-unabhängiges Design)**: minimaler
   CUDA-Runtime-API-Vertrag (`alloc`/`free`/`memcpy_h2d`/`memcpy_d2h`/
   `memcpy_d2d`/`launch_kernel`/`synchronize`) plus Fähigkeits-Flags
   (`supports_spirv`/`supports_dxil`, Standard `false`). Neue
   Hardware-Backends implementieren diesen Vertrag und erweitern das
   `KernelSource`-Enum nicht-brechend (z. B. `Dxil(Vec<u8>)`).
2. **Zweistufiges „Mock → echte Hardware"-Muster**: Phase 1 (Mock-
   Gerät, läuft ohne Hardware, verifiziert Verträge auch in
   GPU-losen CI-Umgebungen), Phase 1,5–2 (echte Implementierung
   hinter einem Cargo-Feature, standardmäßig aus, überspringt sich
   selbst ehrlich, wenn keine echte Hardware vorhanden ist).
3. **HLSL-cbuffer-Array-Padding-Falle**: Skalar-Arrays in `cbuffer`
   werden auf 16-Byte-Grenzen aufgefüllt — dicht gepackte Rust/C++-
   Konstanten passen dann nicht zum HLSL-Layout (führte real zu
   einem "Ausgabe bleibt Klartext"-Bug im ChaCha20-Kernel). Lösung:
   einzelne Skalarfelder (`key0`…`key7`) statt Array-Deklarationen.
4. **HLSL-eingebettete Root-Signature**: `[RootSignature(...)]`-
   Attribut direkt im Shader lässt `dxc` die Root-Signature ins DXIL
   einbetten — spart manuelles Root-Signature-Deskriptor-Bauen im
   Rust/C++-Code. Root-UAV-Deskriptoren direkt binden statt über
   Descriptor-Heaps.
5. **DXGI-Adapter-Enumeration für Herstellererkennung**:
   `D3D12CreateDevice(None, ...)` liefert keine Hersteller-/VRAM-
   Info. `IDXGIFactory1::EnumAdapters1(0)` → `DXGI_ADAPTER_DESC1`
   liefert PCIe-Vendor-ID (NVIDIA/AMD/Intel). Bei Fehlschlag sicher
   auf `None` zurückfallen.
6. **Ehrlicher Hinweis zu GPU-Kompression/-Verschlüsselung**: bei
   kleinen Payloads (Netzwerk-MTU-Größe) kann der Host↔Device-
   Transfer-Overhead den GPU-Rechenvorteil zunichtemachen — vor
   Integration echte Benchmarks für die Ziel-Payload-Größe fahren.
7. **RAID6-Paritätskernel-Muster**: variable Anzahl Datenplatten als
   eine zusammenhängende Puffer-Bindung (statt einzelner Bindings
   pro Platte). Q-Parity (Reed-Solomon) GF(2^8)-Multiplikation als
   eigenständige `gf_mul`-Funktion (Russian-Peasant-Multiplikation,
   irreduzibles Polynom `0x11D`) im Shader.
8. **64-Bit-unabhängige GPU-Implementierung**: 64-Bit-Ganzzahlen sind
   in DXIL SM6.0 optionale Hardware-Features. Poly1305 implementiert
   32×32→64-Bit-Paarmultiplikation/-addition/-shift rein mit 32-Bit-
   Operationen — Muster für Krypto/Bignum-Portierung auf unsichere
   64-Bit-Ziele.
9. **DeepSeek-V3-artige MLA-KV-Cache-Kompression**: Down-/Up-
   Projection auf Basis der vorhandenen `sgemm`. **Ehrliche
   Offenlegung**: die Projektionsmatrizen sind nur zufallsinitialisiert
   (kein trainiertes Gewicht) — die Kompression ist verlustbehaftet;
   dies belegt nur, dass der Rechenpfad korrekt verdrahtet ist, nicht
   dass die Generierungsqualität erhalten bleibt.
10. **Repetition-Penalty (CTRL-Stil)**: `logit>0` → `/penalty`,
    `logit<=0` → `*penalty`, angewendet auf bereits gesehene Tokens
    vor dem Argmax. `penalty=1.0` ist ein Kurzschluss-Rückweg, der
    das bestehende `generate()`-Verhalten exakt erhält (keine
    Regression). Empirischer Standardwert `1.3` — muss je nach
    Prompt-Struktur neu kalibriert werden.

**Aktueller Stand**: Cargo-Workspace aus `opencuda-core`/
`opencuda-cpu`/`opencuda-vulkan`/`opencuda-directx`/`opencuda-blas`/
`open-cuda-bert`/`open-cuda-llm`. `opencuda-directx` bis Phase 2
implementiert (vector_add/matmul/ChaCha20/Poly1305 auf echter
Hardware verifiziert). RAID6 P-/Q-Parity-Kernel in `opencuda-vulkan`
hinzugefügt, auf echter Hardware verifiziert. Details siehe die
HANDOFF-Einträge in [CLAUDE.md](CLAUDE.md).

## 21.11.2026 Automatische Genauigkeitswahl und geringer Speicherbedarf (open-cuda, aruaru-llm)
- **Genauigkeit der Gewichte**: Auf einem Rechner mit 32 GB und nur CPU sind INT8-Gewichte die praktische Wahl (etwa viermal schneller als FP8; ein 7B-Modell braucht ungefähr 2 Sekunden pro Token). INT4 nur in den mittleren Schichten ist möglich, war bei anderem Text aber etwa 17 % schlechter. FP8 ist langsam und daher nicht der Standard.
- **Automatische Wahl**: `PrecisionTable::auto_for_memory` nimmt in der Reihenfolge f32, alles INT8, INT4 in den mittleren Schichten die erste Tabelle, die in 70 % des freien Arbeitsspeichers passt. In `aruaru-llm` steuert man das über die Umgebungsvariable `ARUARU_LLM_PRECISION` (nicht gesetzt = f32, `auto` oder eine Tabelle wie `int8,1:f32`).
- **INT8-KV-Cache**: `QWEN_KV_INT8=1` simuliert nur die Rundung; die Perplexität wurde etwa 1 % schlechter. Der Speicherbedarf sinkt dadurch noch nicht.
- **Offen**: den KV-Cache tatsächlich als INT8 speichern und `auto` auf einem echten Server vollständig durchlaufen lassen. Die Übersetzungen stammen von einer KI und wurden noch nicht von Muttersprachlern geprüft.

## 21.11.2026 Nachtrag: KV-Cache wird wirklich als INT8 gespeichert
- Mit `QWEN_KV_INT8=1` wird der KV-Cache jetzt **tatsächlich** als INT8 mit einem Skalierungswert pro Kopf und Token gespeichert (`open-cuda` 2ba4b5f). Bei Qwen2.5-1.5B stieg die Perplexität von 7,079 auf 7,159 (etwa +1 %); bei anderem Text sank sie von 12,1 auf 11,8 (im Rauschen). Der KV-Speicher beträgt etwa ein Viertel von f32 (ein Test mit head_dim 64 zeigt mehr als dreifache Verkleinerung).
- Offen: ein vollständiger Lauf von `ARUARU_LLM_PRECISION=auto` auf einem echten Server. Die Übersetzungen stammen von einer KI und wurden noch nicht von Muttersprachlern geprüft.
