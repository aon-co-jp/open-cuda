# Philosophie de conception & Politique de développement (open-cuda)

> **Remarque** : Ceci est une traduction condensée de l'état actuel. Le
> journal historique complet des changements HANDOFF (des dizaines
> d'entrées depuis le 2026-06-26) reste disponible uniquement en
> japonais dans [CLAUDE.md](CLAUDE.md), par souci de concision —
> consultez-le pour les détails de chaque session.

Disque de travail : `F:\runo`. Dépôt GitHub :
[aon-co-jp/open-cuda](https://github.com/aon-co-jp/open-cuda). Début du
développement : 2026-06-26.

## Rôle de ce projet

« Le second CUDA » — une fondation d'abstraction/calcul GPU (la
conception `OmniGPU`, voir `OmniGPU-Design.md`) visant la compatibilité
Windows/macOS/Linux et Intel/AMD/NVIDIA. Forme un « SET » avec
`aruaru-llm`, qui est le véritable consommateur du pipeline d'exécution
GPU/CPU.

## Architecture des crates

- **`opencuda-core`** : trait `GpuDevice` partagé (équivalent de l'API
  CUDA Runtime).
- **`opencuda-cpu`** : backend CPU (parallélisme de données via
  `rayon`).
- **`opencuda-vulkan`** : backend Vulkan Compute, multiplateforme
  (Windows/Linux/Android natif, macOS/iOS via MoltenVK). GEMM/
  Attention/quantification INT4·INT8 vérifiés sur matériel réel.
- **`opencuda-directx`** : backend DirectX 12 Compute (Windows
  uniquement, coexiste avec Vulkan). Dispatch de noyaux vector_add/
  matmul/ChaCha20/Poly1305 vérifié sur matériel réel (GT 730) ;
  énumération des adaptateurs DXGI pour la détection
  fournisseur/VRAM implémentée.
- **`opencuda-blas`** : équivalent de NumPy (GEMM/Attention/
  quantification/Flash Attention/compression KV MLA).
- **`open-cuda-bert`** : passe avant pour encodeurs BERT
  (multilingual-e5-small).
- **`open-cuda-llm`** : équivalent de vLLM — décodeur GPT-2 avec cache
  KV, pénalité de répétition, compression MLA, flash attention et
  (fonctionnalité la plus récente) décodage spéculatif.
- **`open-cuda-whisper`** : équivalent de Whisper (log-mel-spectrogramme
  + encodeur + décodeur avec cache KV + cross-attention), actuellement
  seulement un MVP à initialisation aléatoire.

## État honnête de la divulgation

- **cuBLAS/rocBLAS/oneMKL restent des stubs non vérifiés** — cette
  machine n'a pas de toolchain CUDA/ROCm/oneAPI pour la vérification.
- **FP8 rejeté** : le seul GPU de cette machine (GT730, Kepler, CC 3.5)
  n'a pas de Tensor Cores FP8 — une implémentation ne serait qu'une
  émulation logicielle sans bénéfice réel.
- **DeepSeekMoE rejeté** : `DecoderLayer` n'a qu'un seul FFN dense
  (aucune structure d'experts/routeur) et aucun checkpoint MoE réel
  n'existe.
- **Compression/chiffrement GPU** : le bénéfice réel pour les petites
  charges utiles n'est pas vérifié (le surcoût hôte↔périphérique
  pourrait annuler l'avantage GPU).
- **Découverte sur appareil Android réel (2026-08-15)** : sur un
  appareil Android (moto g53y 5G, Adreno 619), le GPU du téléphone a
  surpassé à la fois le CPU et la comparaison CPU/GPU de la GT730
  bureau à des tailles de matrice plus grandes (jusqu'à ~6× à
  512×512), confirmant l'hypothèse que « les GPU de téléphones
  pourraient être étonnamment rapides ». Réserve : les calculs uniques
  et petits restent limités par le surcoût d'initialisation GPU
  (58–63ms).

## Matrice de support des fournisseurs

Trois couches : l'intégration Vulkan fonctionnelle / l'énumération
`GpuVendor` en tant que couche de rapport uniquement (incl. Qualcomm/
ARM/ImaginationPowerVr non vérifiés) / la couche stub des bibliothèques
fournisseurs. Détails dans `OmniGPU-Design.md` §8.5.

## Entrées HANDOFF récentes pertinentes

- **2026-08-10** : `generate_with_repetition_penalty` (pénalité de
  répétition façon CTRL) corrige un vrai bug de boucle infinie dans le
  décodage glouton GPT-2 d'`aruaru-llm`. Par défaut `penalty=1.3` via
  `ARUARU_LLM_REPETITION_PENALTY`.
- **Travail le plus récent (commit `0c43ba3`)** : Nouveau
  `GptModel::generate_speculative` — décodage spéculatif sans perte à
  la DSpark/Leviathan (le modèle brouillon propose des tokens, le
  modèle cible vérifie via un préremplissage par lot). Vérifié comme
  identique bit à bit à `generate()` sur des fixtures synthétiques et
  des poids réels. **Divulgation honnête** : sur le chemin CPU de cette
  machine, plus lent que le simple `generate()` (draft_k=4 : plain
  4,63s vs spéculatif 7,65s), car le GEMM naïf sur CPU a peu de
  surcoût de dispatch à amortir. Le cas cible réel — la vitesse sur
  Vulkan réel — reste non mesuré.
