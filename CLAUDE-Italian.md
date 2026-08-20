# Filosofia di design & Politica di sviluppo (open-cuda)

> **Nota**: Questa è una traduzione condensata dello stato attuale. Il
> log dei cambiamenti storici HANDOFF completo (decine di voci dal
> 2026-06-26) resta disponibile solo in giapponese in
> [CLAUDE.md](CLAUDE.md), per brevità — consultalo per i dettagli di
> ogni sessione.

Drive di lavoro: `F:\runo`. Repository GitHub:
[aon-co-jp/open-cuda](https://github.com/aon-co-jp/open-cuda). Inizio
sviluppo: 2026-06-26.

## Ruolo di questo progetto

"Il secondo CUDA" — una base di astrazione/calcolo GPU (il design
`OmniGPU`, vedi `OmniGPU-Design.md`) che mira alla compatibilità
Windows/macOS/Linux e Intel/AMD/NVIDIA. Forma un "SET" con
`aruaru-llm`, che è il consumatore reale della pipeline di esecuzione
GPU/CPU.

## Architettura dei crate

- **`opencuda-core`**: trait `GpuDevice` condiviso (equivalente della
  CUDA Runtime API).
- **`opencuda-cpu`**: backend CPU (parallelismo dati via `rayon`).
- **`opencuda-vulkan`**: backend Vulkan Compute, cross-platform
  (Windows/Linux/Android nativo, macOS/iOS via MoltenVK). GEMM/
  Attention/quantizzazione INT4·INT8 verificati su hardware reale.
- **`opencuda-directx`**: backend DirectX 12 Compute (solo Windows,
  coesiste con Vulkan). Dispatch dei kernel vector_add/matmul/
  ChaCha20/Poly1305 verificato su hardware reale (GT 730); enumerazione
  adattatori DXGI per rilevamento vendor/VRAM implementata.
- **`opencuda-blas`**: equivalente di NumPy (GEMM/Attention/
  quantizzazione/Flash Attention/compressione KV MLA).
- **`open-cuda-bert`**: forward pass per encoder BERT
  (multilingual-e5-small).
- **`open-cuda-llm`**: equivalente di vLLM — decoder GPT-2 con cache
  KV, penalità di ripetizione, compressione MLA, flash attention e
  (funzionalità più recente) decodifica speculativa.
- **`open-cuda-whisper`**: equivalente di Whisper (log-mel-spettrogramma
  + encoder + decoder con cache KV + cross-attention), attualmente solo
  un MVP a inizializzazione casuale.

## Stato onesto della divulgazione

- **cuBLAS/rocBLAS/oneMKL restano stub non verificati** — questa
  macchina non ha toolchain CUDA/ROCm/oneAPI per la verifica.
- **FP8 rifiutato**: l'unica GPU di questa macchina (GT730, Kepler,
  CC 3.5) non ha Tensor Core FP8 — un'implementazione sarebbe pura
  emulazione software senza beneficio reale.
- **DeepSeekMoE rifiutato**: `DecoderLayer` ha solo un singolo FFN
  denso (nessuna struttura esperti/router) e non esiste alcun
  checkpoint MoE reale.
- **Compressione/cifratura GPU**: il beneficio reale per payload
  piccoli è non verificato (l'overhead host↔device potrebbe annullare
  il vantaggio GPU).
- **Scoperta su dispositivo Android reale (2026-08-15)**: su un
  dispositivo Android (moto g53y 5G, Adreno 619), la GPU del telefono
  ha superato sia la CPU sia il confronto CPU/GPU della GT730 desktop a
  dimensioni di matrice più grandi (fino a ~6× a 512×512), confermando
  l'ipotesi che "le GPU dei telefoni potrebbero essere sorprendentemente
  veloci". Avvertenza: i calcoli piccoli singoli restano limitati
  dall'overhead di inizializzazione GPU (58–63ms).

## Matrice di supporto vendor

Tre livelli: l'integrazione Vulkan funzionante / l'enum `GpuVendor`
come livello di sola segnalazione (incl. Qualcomm/ARM/ImaginationPowerVr
non verificati) / il livello stub delle librerie vendor. Dettagli in
`OmniGPU-Design.md` §8.5.

## Voci HANDOFF recenti rilevanti

- **2026-08-10**: `generate_with_repetition_penalty` (penalità di
  ripetizione in stile CTRL) risolve un bug reale di loop infinito
  nella decodifica greedy GPT-2 di `aruaru-llm`. Default `penalty=1.3`
  via `ARUARU_LLM_REPETITION_PENALTY`.
- **Lavoro più recente (commit `0c43ba3`)**: Nuovo
  `GptModel::generate_speculative` — decodifica speculativa senza
  perdita in stile DSpark/Leviathan (il modello draft propone token,
  il modello target verifica tramite prefill batch). Verificato come
  identico bit per bit a `generate()` su fixture sintetiche e pesi
  reali. **Divulgazione onesta**: sul path CPU di questa macchina è
  più lento del semplice `generate()` (draft_k=4: plain 4,63s vs
  speculativo 7,65s), perché il GEMM naive su CPU ha poco overhead di
  dispatch da ammortizzare. Il caso target reale — velocità su Vulkan
  reale — resta non misurato.
- **2026-08-19 — Aggiornamento automatico**: indagine sull'introduzione
  di un meccanismo di auto-update (sul modello di `self_update.rs` di
  `open-english`) — scartata: questo repo contiene solo crate di
  libreria e binari di esempio usa-e-getta, nessun servizio residente.
  **Confutazione di un'obiezione**: l'assenza di un servizio residente
  non è un difetto — il vero DirectX/CUDA di Microsoft/NVIDIA funziona
  esso stesso come librerie runtime linkate da ogni processo, non come
  servizi in background (`nvidia-persistenced` è un'eccezione limitata
  alla cache dello stato di inizializzazione della GPU, non un arbitro
  tra processi).
- **2026-08-20 — Correzione**: la nota precedente su un "collegamento
  in fase di progettazione" tra `open-directx` e `open-cuda` era
  imprecisa. In realtà si tratta di due progetti omonimi non correlati:
  (1) il crate interno `opencuda-directx` di questo repo, e (2) il
  repository indipendente `aon-co-jp/open-directx`.
- **2026-08-20 — Quantizzazione INT6 in stile FlexQ**: aggiunti
  `quantize_int6`/`dequantize_int6`/`QuantizedInt6Tensor` in
  `opencuda-blas`, che impacchettano 4 valori a 6 bit in 3 byte.
  PuzzleMoE è stato scartato perché nei modelli di questo repo non
  esiste un'architettura MoE (solo un singolo FFN denso).
