# open-cuda — Guida alla portabilità (versione condensata)

> **Nota**: questa è una traduzione condensata degli schemi
> attualmente riutilizzabili. Il registro storico dettagliato degli
> HANDOFF resta disponibile solo in giapponese in
> [PORTING.md](PORTING.md) — consultarlo prima di adottare davvero
> uno schema.

Riepilogo degli schemi di progettazione riutilizzabili di questo
progetto, utile se portati in un altro progetto.

1. **Trait `GpuDevice` (design indipendente dal backend)**: contratto
   minimo simile alla CUDA Runtime API (`alloc`/`free`/`memcpy_h2d`/
   `memcpy_d2h`/`memcpy_d2d`/`launch_kernel`/`synchronize`) più flag
   di capacità (`supports_spirv`/`supports_dxil`, default `false`).
   I nuovi backend hardware implementano questo contratto ed estendono
   l'enum `KernelSource` in modo non distruttivo (es. `Dxil(Vec<u8>)`).
2. **Schema a due fasi "mock → hardware reale"**: Fase 1 (dispositivo
   mock, funziona senza hardware, verifica i contratti anche in CI
   senza GPU), Fase 1,5–2 (implementazione reale dietro una feature
   Cargo, disattivata di default, si salta onestamente se non c'è
   hardware reale).
3. **Trappola del padding degli array in cbuffer HLSL**: gli array
   scalari in `cbuffer` vengono allineati a 16 byte — costanti
   Rust/C++ compattate non corrispondono al layout HLSL (causò un
   bug reale "l'output resta in chiaro" nel kernel ChaCha20).
   Soluzione: campi scalari individuali (`key0`…`key7`) invece di
   dichiarazioni ad array.
4. **Root signature incorporata in HLSL**: l'attributo
   `[RootSignature(...)]` scritto direttamente nello shader fa sì che
   `dxc` incorpori la root signature nel DXIL — evita di costruire
   manualmente il descrittore della root signature in Rust/C++.
   Bind diretto dei descrittori UAV root invece di usare descriptor
   heap.
5. **Enumerazione adattatori DXGI per il rilevamento del vendor**:
   `D3D12CreateDevice(None, ...)` non fornisce informazioni su
   vendor/VRAM. `IDXGIFactory1::EnumAdapters1(0)` →
   `DXGI_ADAPTER_DESC1` fornisce il vendor ID PCIe (NVIDIA/AMD/Intel).
   In caso di errore, ripiegare in modo sicuro su `None`.
6. **Avviso onesto su compressione/cifratura GPU**: con payload
   piccoli (dimensione MTU di rete) l'overhead di trasferimento
   Host↔Device può annullare il vantaggio computazionale della GPU —
   eseguire benchmark reali sulla dimensione del payload target prima
   dell'integrazione.
7. **Schema kernel di parità RAID6**: numero variabile di dischi dati
   come un unico buffer concatenato (invece di binding separati per
   disco). Moltiplicazione GF(2^8) per Q-parity (Reed-Solomon) come
   funzione `gf_mul` autonoma (moltiplicazione "Russian peasant",
   polinomio irriducibile `0x11D`) nello shader.
8. **Implementazione GPU indipendente da interi a 64 bit**: gli
   interi a 64 bit sono una feature hardware opzionale in DXIL SM6.0.
   Poly1305 implementa moltiplicazione/addizione/shift a coppie
   32×32→64 bit usando solo operazioni a 32 bit — schema utile per
   portare crittografia/big number su target a 64 bit non garantiti.
9. **Compressione della KV-cache in stile MLA di DeepSeek-V3**:
   down-/up-projection basate sulla `sgemm` esistente. **Divulgazione
   onesta**: le matrici di proiezione sono solo inizializzate
   casualmente (nessun peso addestrato) — la compressione è
   lossy; questo dimostra solo che il percorso di calcolo è cablato
   correttamente, non che la qualità della generazione sia
   preservata.
10. **Repetition penalty (stile CTRL)**: `logit>0` → `/penalty`,
    `logit<=0` → `*penalty`, applicato ai token già visti prima
    dell'argmax. `penalty=1.0` è una scorciatoia che preserva
    esattamente il comportamento esistente di `generate()` (nessuna
    regressione). Valore predefinito empirico `1.3` — va ricalibrato
    in base alla struttura del prompt.

**Stato attuale**: workspace Cargo composto da `opencuda-core`/
`opencuda-cpu`/`opencuda-vulkan`/`opencuda-directx`/`opencuda-blas`/
`open-cuda-bert`/`open-cuda-llm`. `opencuda-directx` implementato
fino alla Fase 2 (vector_add/matmul/ChaCha20/Poly1305 verificati su
hardware reale). Kernel di parità RAID6 P/Q aggiunti a
`opencuda-vulkan`, verificati su hardware reale. Dettagli negli
HANDOFF di [CLAUDE.md](CLAUDE.md).
