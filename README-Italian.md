# open-cuda

*English*: [README-English.md](README-English.md) ·
*Altre lingue*: [Deutsch](README-German.md) · [Italiano](README-Italian.md) ·
[Français](README-French.md) · [Русский](README-Russian.md) ·
[Українська](README-Ukrainian.md) · [עברית](README-Hebrew.md) ·
[فارسی](README-Persian.md) · [العربية](README-Arabic.md)

> **Aggiornamento recente (2026-08-20)**: Implementata la
> quantizzazione INT6 in stile FlexQ
> (`quantize_int6`/`dequantize_int6`/`QuantizedInt6Tensor` in
> `opencuda-blas`, che impacchetta 4 valori a 6 bit in 3 byte).
> PuzzleMoE è stato scartato dopo aver verificato i suoi prerequisiti
> — richiede un'architettura MoE già esistente, assente nei modelli
> densi GPT-2/BERT di questo repository. Corretta anche una nota
> precedente imprecisa che affermava un "collegamento in fase di
> progettazione" tra `open-directx` e `open-cuda` — in realtà si
> tratta di due progetti omonimi non correlati: il crate interno
> `opencuda-directx` e il repository indipendente
> `aon-co-jp/open-directx`. Inoltre è stata valutata e respinta
> l'aggiunta di un meccanismo di auto-aggiornamento (sullo stile di
> `self_update.rs` di `open-english`) — questo repository non ha un
> servizio residente (10 crate di libreria + 12 binari di esempio
> usa-e-getta), ed è stato confermato che i veri DirectX/CUDA di
> Microsoft/NVIDIA sono essi stessi architetture a libreria runtime
> anziché servizi residenti in background — il design attuale non è
> quindi un difetto. Vedi le voci HANDOFF del 2026-08-19/2026-08-20 in
> [CLAUDE.md](CLAUDE.md).

> **Aggiornamento recente (2026-08-10)**: Aggiunto `generate_with_repetition_penalty`
> (penalità di ripetizione in stile CTRL — penalty>1.0 indebolisce i logit
> dei token già comparsi) a `open-cuda-llm::GptModel`. L'esistente
> `generate()` è ora un thin wrapper che lo chiama con `penalty=1.0`
> (comportamento identico byte per byte, nessuna regressione per test/
> chiamanti esistenti). Questo risolve direttamente un bug segnalato in
> `aruaru-llm` in cui il GPT-2 base (senza fine-tuning conversazionale)
> con decodifica greedy ripete all'infinito la stessa stringa (es.
> "Student: Hello"). Aggiunto un nuovo test su pesi reali GPT-2 124M,
> `repetition_penalty_reduces_degenerate_loop_on_real_gpt2_weights`, che
> conferma che il loop si riproduce effettivamente senza penalità e che
> effettivamente si interrompe (producendo testo grammaticalmente
> naturale) con `penalty=1.3`. `/v1/generate` di `aruaru-llm` ora chiama
> questa nuova API con `penalty=1.3` come default (sovrascrivibile via
> `ARUARU_LLM_REPETITION_PENALTY`). Vedi la voce HANDOFF del 2026-08-10 in
> [CLAUDE.md](CLAUDE.md).

> **Aggiornamento recente (2026-08-08)**: Riverificata su hardware reale
> la nota "MLA implementata" del 2026-08-06
> (`cargo test -p opencuda-blas mla -- --nocapture` → `1 passed; 0
> failed`, ha percorso il vero path Vulkan della GT730). Sono stati
> studiati FP8 mixed-precision e DeepSeekMoE ma **si è deciso di non
> implementare nessuno dei due**: FP8 non ha supporto hardware reale
> sull'unica GPU di questa macchina (GT730, Kepler, CC 3.5 — nessun
> Tensor Core FP8, sarebbe solo emulazione software); DeepSeekMoE non ha
> un punto di integrazione genuino poiché il `DecoderLayer` di
> `open-cuda-llm` ha solo un singolo FFN denso (nessuna struttura
> esperti/router) e non esiste alcun checkpoint MoE reale. Vedi la voce
> HANDOFF del 2026-08-08 in [CLAUDE.md](CLAUDE.md) per i dettagli.

> **Aggiornamento recente (2026-08-07)**: Collegato il kernel fuso di
> flash-attention (`flash_attention_with_spirv`) nel `DecoderLayer` di
> `open-cuda-llm`, tramite `GptModel::set_flash_attention_spirv()` —
> torna al path esistente a 3 dispatch quando non impostato (pienamente
> retrocompatibile). Verificato su hardware reale NVIDIA GT 730: le
> sequenze di token generate via Vulkan sono identiche byte per byte al
> path CPU. Vedi HANDOFF in [CLAUDE.md](CLAUDE.md) per i dettagli.

> **Implementato il 2026-08-06**: compressione low-rank della cache KV
> ispirata alla Multi-Head Latent Attention (MLA) di DeepSeek-V3 —
> `opencuda-blas::mla_compress_kv`/`mla_decompress_kv`. Dopo aver
> studiato il report tecnico
> ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437)) e blog di
> implementazione in giapponese e inglese, è stato costruito il
> meccanismo di proiezione low-rank (il design dietro la riduzione del
> 93,3% della cache KV riportata da DeepSeek) sopra il backend `sgemm`
> esistente, già verificato su hardware reale. Verificato su hardware
> reale (GT730) che i path CPU e Vulkan corrispondano numericamente. Non
> trasporta pesi addestrati (questo dimostra il meccanismo, non la
> qualità di compressione addestrata — vedi la sezione HANDOFF di
> `CLAUDE.md` per la divulgazione onesta). L'applicazione delle tecniche
> Toshiba SBM / DeepSeek agli altri 7 repository è ancora in fase di
> valutazione.

> **Aggiornato il 2026-07-25**: L'intestazione del file di policy di
> sviluppo (`CLAUDE.md`) è stata rinominata da "Politica di sviluppo &
> regole dell'ambiente di sviluppo" a "Filosofia di design & Politica di
> sviluppo & regole dell'ambiente di sviluppo", per separare più
> chiaramente la filosofia di design del progetto (ciò che apprezziamo),
> la politica di sviluppo (come lavoriamo) e le regole dell'ambiente di
> sviluppo (convenzioni operative concrete). Vedi `CLAUDE.md` per i
> dettagli.

**Inizio dello sviluppo: 2026-06-26** (data di creazione GitHub di
questo repository)

"Il secondo CUDA" — una base di astrazione/calcolo GPU (il design
`OmniGPU`) che mira alla compatibilità Windows/macOS/Linux e
Intel/AMD/NVIDIA. Abbinato ("SET") con `aruaru-llm`, che è il
consumatore implementativo della pipeline di esecuzione GPU/CPU.

## Cos'è questo

- **`opencuda-core`**: Il trait `GpuDevice` condiviso da tutti i
  backend (equivalente della CUDA Runtime API: `alloc`/`memcpy`/
  `launch_kernel`).
- **`opencuda-cpu`**: Il backend CPU (parallelismo dati via `rayon`).
- **`opencuda-vulkan`**: Il backend Vulkan Compute (cross-platform,
  nativo su Windows/Linux/Android, macOS/iOS via MoltenVK). GEMM/
  Attention/quantizzazione INT4·INT8 verificati su esecuzione Vulkan
  reale.
- **`opencuda-directx`** (aggiunto il 2026-07-23): Un backend DirectX 12
  Compute (solo Windows, backend opt-in che coesiste con Vulkan).
  Dispatch GPU di `vector_add`/`matmul`/`ChaCha20` verificato su
  hardware reale (NVIDIA GT 730) — l'output corrisponde esattamente
  alle implementazioni di riferimento CPU (es. il crate RustCrypto
  `chacha20`) nei test. È implementato anche il recupero reale di
  nome vendor/capacità VRAM tramite enumerazione degli adattatori DXGI.
- **`opencuda-blas`**: Equivalente di NumPy (GEMM/Attention/
  quantizzazione).
- **`open-cuda-bert`**: Forward pass per encoder della famiglia BERT
  (supporta multilingual-e5-small).
- **`open-cuda-llm`**: Equivalente di vLLM (decodifica greedy con cache
  KV). Implementa `GptModel::load`, che carica GPT-2 (Hugging Face
  `openai-community/gpt2`) `safetensors` (2026-07-25, stesso design di
  `open-cuda-bert::BertModel::load`). Verificato su hardware reale:
  scaricare e caricare pesi reali GPT-2 124M produce inglese decodificato
  greedy chiaramente più fluente rispetto all'inizializzazione casuale
  (output privo di significato) — es. "The quick brown fox" → "es are a
  great way to get a little bit of a". Vedi HANDOFF di `CLAUDE.md` per i
  dettagli.
- **`open-cuda-whisper`** (aggiunto il 2026-07-31): Equivalente di
  Whisper (riconoscimento vocale, #6 nella roadmap di ricerca di
  mercato). Estrazione log-mel-spettrogramma + encoder (stesso design
  Multi-Head Attention di `open-cuda-bert`) + decoder con cache KV
  (stesso design di `open-cuda-llm`) + cross-attention. **Attualmente
  solo un MVP con inizializzazione casuale** (nessun caricatore di pesi
  Whisper addestrati ancora — vedi HANDOFF di `CLAUDE.md` per i
  dettagli).

## Perché abbiamo sia DirectX che Vulkan (decisione tecnica del 2026-07-23)

Inizialmente, si pensava che questo progetto fosse "in sviluppo come
plugin DirectX". Dopo aver verificato con ricerche web in
giapponese/inglese, è emerso che DXVK/vkd3d-proton (la tecnologia che
Proton di Valve usa realmente) convertono entrambi nella direzione
"DirectX (API solo Windows) → Vulkan (API cross-platform)" — non sono
stati trovati esempi della direzione inversa. **Per l'obiettivo del
supporto cross-platform, l'approccio Vulkan Compute esistente è
tecnicamente la via più diretta.** Sulla base di ciò, è stata adottata
la politica "mantenere Vulkan, e aggiungere DirectX per Windows in
coesistenza" ed è stato implementato `opencuda-directx`.

## Divulgazione onesta

- **Il supporto cross-platform è un lavoro in corso**: Vulkan Compute è
  progettato per supporto nativo su Windows/Linux/Android e supporto
  basato su MoltenVK su macOS/iOS, ma la verifica su hardware reale è
  stata eseguita solo su questa macchina (Windows, NVIDIA GT 730).
- **Il dispatch dei kernel di `opencuda-directx` copre solo
  parzialmente la Fase 2**: sono implementati `vector_add`, `matmul` e
  `ChaCha20` (solo cifratura — non include il calcolo del tag di
  autenticazione Poly1305). È implementato anche il rilevamento del
  vendor tramite enumerazione degli adattatori DXGI (`GpuVendor::Nvidia`
  ecc.), ma informazioni dettagliate come `compute_capability` restano
  un placeholder poiché non ottenibili da DXGI.
- **Il beneficio reale della compressione/cifratura GPU è non
  verificato**: per un payload piccolo come un singolo frame di tunnel,
  c'è una preoccupazione tecnica che l'overhead di trasferimento
  host↔device possa annullare il vantaggio computazionale della GPU (il
  benchmarking reale resta un compito futuro).

## Relazioni all'interno di questo ecosistema

- [aruaru-llm](https://github.com/aon-co-jp/aruaru-llm) — un'implementazione
  di riferimento di questo repo (consumatore della pipeline di
  esecuzione GPU/CPU).
- [RS-LinkFusion](https://github.com/aon-co-jp/RS-LinkFusion) — sta
  valutando l'uso dell'accelerazione di compressione/cifratura GPU (il
  kernel ChaCha20 in `opencuda-directx`).
- [open-raid-z](https://github.com/aon-co-jp/open-raid-z) — la fonte
  canonica delle regole di politica di sviluppo.

## Build & test

```bash
cargo build --workspace
cargo test --workspace

# Test hardware reale DirectX 12 (solo Windows, feature real-dx12)
cargo test -p opencuda-directx --features real-dx12
```

### Provalo sulla tua GPU (aggiunto il 2026-07-27)

Se vuoi semplicemente eseguire prima una cosa per verificare che
funzioni, ogni sub-crate sotto `examples/` (un membro del workspace) può
essere eseguito con `cargo run -p <nome>`. `vulkan_info` in particolare
è un esempio minimale che si limita a enumerare e stampare i dispositivi
fisici Vulkan reali (nome vendor GPU, capacità VRAM) sulla tua macchina,
rendendolo il primo comando migliore per verificare se una GPU viene
rilevata nel tuo ambiente:

```bash
cargo run -p vulkan_info
```

Altri esempi (`matmul`, `matmul_vulkan_real`, `vector_add`,
`vector_add_vulkan`, `vector_add_vulkan_real`, `vector_add_omniir`)
possono essere eseguiti allo stesso modo con `cargo run -p <nome>`. Vedi
`OmniGPU-Design.md` §8.5 per la matrice di stato del supporto per
vendor (Intel/AMD/NVIDIA ecc.).

## Licenza

Apache-2.0
