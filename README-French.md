# open-cuda

*English*: [README-English.md](README-English.md) ·
*Autres langues*: [Deutsch](README-German.md) · [Italiano](README-Italian.md) ·
[Français](README-French.md) · [Русский](README-Russian.md) ·
[Українська](README-Ukrainian.md) · [עברית](README-Hebrew.md) ·
[فارسی](README-Persian.md) · [العربية](README-Arabic.md)

> **Mise à jour récente (2026-08-10)** : Ajout de
> `generate_with_repetition_penalty` (pénalité de répétition façon CTRL
> — penalty>1.0 affaiblit les logits des tokens déjà apparus) à
> `open-cuda-llm::GptModel`. Le `generate()` existant est désormais un
> wrapper léger qui l'appelle avec `penalty=1.0` (comportement identique
> octet par octet, aucune régression pour les tests/appelants
> existants). Cela corrige directement un bug signalé dans `aruaru-llm`
> où le décodage glouton de GPT-2 de base (sans fine-tuning
> conversationnel) boucle indéfiniment sur la même chaîne (par ex.
> "Student: Hello"). Un nouveau test sur des poids réels GPT-2 124M,
> `repetition_penalty_reduces_degenerate_loop_on_real_gpt2_weights`,
> confirme que la boucle se reproduit effectivement sans pénalité et
> s'arrête effectivement (produisant un texte grammaticalement naturel)
> avec `penalty=1.3`. `/v1/generate` d'`aruaru-llm` appelle désormais
> cette nouvelle API avec `penalty=1.3` par défaut (modifiable via
> `ARUARU_LLM_REPETITION_PENALTY`). Voir l'entrée HANDOFF du 2026-08-10
> dans [CLAUDE.md](CLAUDE.md).

> **Mise à jour récente (2026-08-08)** : Revérifié sur du matériel réel
> la note "MLA implémenté" du 2026-08-06
> (`cargo test -p opencuda-blas mla -- --nocapture` → `1 passed; 0
> failed`, a emprunté le vrai chemin Vulkan de la GT730). FP8
> mixed-precision et DeepSeekMoE ont été étudiés mais **il a été décidé
> de n'implémenter ni l'un ni l'autre** : FP8 n'a aucun support matériel
> réel sur l'unique GPU de cette machine (GT730, Kepler, CC 3.5 — pas de
> Tensor Cores FP8, ce ne serait qu'une émulation logicielle) ;
> DeepSeekMoE n'a pas de point d'intégration réel puisque le
> `DecoderLayer` d'`open-cuda-llm` n'a qu'un seul FFN dense (aucune
> structure d'experts/routeur) et aucun checkpoint MoE réel n'existe.
> Voir l'entrée HANDOFF du 2026-08-08 dans [CLAUDE.md](CLAUDE.md) pour
> les détails.

> **Mise à jour récente (2026-08-07)** : Connexion du noyau flash-
> attention fusionné (`flash_attention_with_spirv`) dans le
> `DecoderLayer` d'`open-cuda-llm`, via
> `GptModel::set_flash_attention_spirv()` — repli sur le chemin existant
> à 3 dispatchs si non défini (entièrement rétrocompatible). Vérifié sur
> du matériel réel NVIDIA GT 730 : les séquences de tokens générées via
> Vulkan sont identiques octet par octet au chemin CPU. Voir HANDOFF
> dans [CLAUDE.md](CLAUDE.md) pour les détails.

> **Implémenté le 2026-08-06** : compression low-rank du cache KV
> inspirée de la Multi-Head Latent Attention (MLA) de DeepSeek-V3 —
> `opencuda-blas::mla_compress_kv`/`mla_decompress_kv`. Après avoir
> étudié le rapport technique
> ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437)) et des blogs
> d'implémentation en japonais et en anglais, le mécanisme de projection
> low-rank (la conception derrière la réduction de 93,3 % du cache KV
> rapportée par DeepSeek) a été construit sur le backend `sgemm`
> existant déjà vérifié sur matériel réel. Vérifié sur matériel réel
> (GT730) que les chemins CPU et Vulkan correspondent numériquement. Ne
> transporte pas de poids entraînés (ceci démontre le mécanisme, pas la
> qualité de compression entraînée — voir la section HANDOFF de
> `CLAUDE.md` pour la divulgation honnête). L'application des techniques
> Toshiba SBM / DeepSeek aux 7 autres dépôts est encore à l'étude.

> **Mis à jour le 2026-07-25** : L'en-tête du fichier de politique de
> développement (`CLAUDE.md`) a été renommé de « Politique de
> développement & règles d'environnement de développement » à
> « Philosophie de conception & Politique de développement & règles
> d'environnement de développement », pour mieux séparer la philosophie
> de conception du projet (ce que nous valorisons), la politique de
> développement (comment nous travaillons) et les règles d'environnement
> de développement (conventions opérationnelles concrètes). Voir
> `CLAUDE.md` pour les détails.

**Début du développement : 2026-06-26** (date de création GitHub de ce
dépôt)

« Le second CUDA » — une fondation d'abstraction/calcul GPU (la
conception `OmniGPU`) visant la compatibilité Windows/macOS/Linux et
Intel/AMD/NVIDIA. Associé (« SET ») avec `aruaru-llm`, qui est le
consommateur d'implémentation du pipeline d'exécution GPU/CPU.

## Ce que c'est

- **`opencuda-core`** : Le trait `GpuDevice` partagé par tous les
  backends (équivalent de l'API CUDA Runtime : `alloc`/`memcpy`/
  `launch_kernel`).
- **`opencuda-cpu`** : Le backend CPU (parallélisme de données via
  `rayon`).
- **`opencuda-vulkan`** : Le backend Vulkan Compute (multiplateforme,
  natif sur Windows/Linux/Android, macOS/iOS via MoltenVK). GEMM/
  Attention/quantification INT4·INT8 vérifiés sur exécution Vulkan
  réelle.
- **`opencuda-directx`** (ajouté le 2026-07-23) : Un backend DirectX 12
  Compute (Windows uniquement, backend opt-in coexistant avec Vulkan).
  Dispatch GPU de `vector_add`/`matmul`/`ChaCha20` vérifié sur matériel
  réel (NVIDIA GT 730) — la sortie correspond exactement aux
  implémentations de référence CPU (par ex. le crate RustCrypto
  `chacha20`) dans les tests. La récupération réelle du nom du
  fournisseur/capacité VRAM via l'énumération des adaptateurs DXGI est
  également implémentée.
- **`opencuda-blas`** : Équivalent de NumPy (GEMM/Attention/
  quantification).
- **`open-cuda-bert`** : Passe avant pour les encodeurs de la famille
  BERT (prend en charge multilingual-e5-small).
- **`open-cuda-llm`** : Équivalent de vLLM (décodage glouton avec cache
  KV). Implémente `GptModel::load`, qui charge les `safetensors` de
  GPT-2 (Hugging Face `openai-community/gpt2`) (2026-07-25, même
  conception que `open-cuda-bert::BertModel::load`). Vérifié sur
  matériel réel : télécharger et charger les poids réels GPT-2 124M
  produit un anglais décodé glouton nettement plus fluide que
  l'initialisation aléatoire (sortie dénuée de sens) — par ex. "The
  quick brown fox" → "es are a great way to get a little bit of a".
  Voir le HANDOFF de `CLAUDE.md` pour les détails.
- **`open-cuda-whisper`** (ajouté le 2026-07-31) : Équivalent de
  Whisper (reconnaissance vocale, #6 sur la feuille de route d'étude de
  marché). Extraction du log-mel-spectrogramme + encodeur (même
  conception Multi-Head Attention qu'`open-cuda-bert`) + décodeur avec
  cache KV (même conception qu'`open-cuda-llm`) + cross-attention.
  **Actuellement uniquement un MVP à initialisation aléatoire** (pas
  encore de chargeur de poids Whisper entraînés — voir le HANDOFF de
  `CLAUDE.md` pour les détails).

## Pourquoi nous avons à la fois DirectX et Vulkan (décision technique du 2026-07-23)

Au départ, on pensait que ce projet était « en développement en tant que
plugin DirectX ». Après vérification par des recherches web en
japonais/anglais, il s'est avéré que DXVK/vkd3d-proton (la technologie
que Proton de Valve utilise réellement) convertissent tous deux dans le
sens « DirectX (API Windows uniquement) → Vulkan (API multiplateforme) »
— aucun exemple du sens inverse n'a été trouvé. **Pour l'objectif de
support multiplateforme, l'approche Vulkan Compute existante est
techniquement le chemin le plus direct.** Sur cette base, la politique
« garder Vulkan, et ajouter DirectX pour Windows en coexistence » a été
adoptée et `opencuda-directx` a été implémenté.

## Divulgation honnête

- **Le support multiplateforme est un travail en cours** : Vulkan
  Compute est conçu pour un support natif sur Windows/Linux/Android et
  un support basé sur MoltenVK sur macOS/iOS, mais la vérification sur
  matériel réel n'a été effectuée que sur cette machine (Windows, NVIDIA
  GT 730).
- **Le dispatch de noyaux d'`opencuda-directx` ne couvre que
  partiellement la Phase 2** : `vector_add`, `matmul` et `ChaCha20`
  (chiffrement uniquement — n'inclut pas le calcul du tag
  d'authentification Poly1305) sont implémentés. La détection du
  fournisseur via l'énumération des adaptateurs DXGI
  (`GpuVendor::Nvidia` etc.) est également implémentée, mais des
  informations détaillées comme `compute_capability` restent un
  placeholder car non disponibles via DXGI.
- **Le bénéfice réel de la compression/chiffrement GPU n'est pas
  vérifié** : pour une charge utile aussi petite qu'une seule trame de
  tunnel, il existe une préoccupation technique selon laquelle le
  surcoût de transfert hôte↔périphérique pourrait annuler l'avantage
  computationnel du GPU (le benchmarking réel reste une tâche future).

## Relations au sein de cet écosystème

- [aruaru-llm](https://github.com/aon-co-jp/aruaru-llm) — une
  implémentation de référence de ce dépôt (consommateur du pipeline
  d'exécution GPU/CPU).
- [RS-LinkFusion](https://github.com/aon-co-jp/RS-LinkFusion) — évalue
  l'utilisation de l'accélération de compression/chiffrement GPU (le
  noyau ChaCha20 dans `opencuda-directx`).
- [open-raid-z](https://github.com/aon-co-jp/open-raid-z) — la source
  canonique des règles de politique de développement.

## Build & test

```bash
cargo build --workspace
cargo test --workspace

# Tests matériel réel DirectX 12 (Windows uniquement, feature real-dx12)
cargo test -p opencuda-directx --features real-dx12
```

### Essayez-le sur votre propre GPU (ajouté le 2026-07-27)

Si vous voulez simplement exécuter une chose d'abord pour vérifier que
ça fonctionne, chaque sous-crate sous `examples/` (membre du workspace)
peut être exécuté avec `cargo run -p <nom>`. `vulkan_info` en
particulier est un exemple minimal qui énumère et affiche simplement les
vrais périphériques physiques Vulkan (nom du fournisseur GPU, capacité
VRAM) sur votre machine, ce qui en fait la meilleure première commande
pour vérifier si un GPU est détecté dans votre environnement :

```bash
cargo run -p vulkan_info
```

D'autres exemples (`matmul`, `matmul_vulkan_real`, `vector_add`,
`vector_add_vulkan`, `vector_add_vulkan_real`, `vector_add_omniir`)
peuvent de même être exécutés avec `cargo run -p <nom>`. Voir
`OmniGPU-Design.md` §8.5 pour la matrice d'état de support par
fournisseur (Intel/AMD/NVIDIA etc.).

## Licence

Apache-2.0
