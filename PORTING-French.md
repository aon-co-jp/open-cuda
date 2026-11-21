# open-cuda — Guide de portage (version condensée)

> **Remarque** : ceci est une traduction condensée des schémas
> actuellement réutilisables. Le journal historique détaillé des
> HANDOFF reste disponible uniquement en japonais dans
> [PORTING.md](PORTING.md) — le consulter avant d'adopter réellement
> un schéma.

Résumé des schémas de conception réutilisables de ce projet, utile
en cas de portage vers un autre projet.

1. **Trait `GpuDevice` (conception indépendante du backend)** :
   contrat minimal équivalent à l'API CUDA Runtime (`alloc`/`free`/
   `memcpy_h2d`/`memcpy_d2h`/`memcpy_d2d`/`launch_kernel`/
   `synchronize`) plus des drapeaux de capacité
   (`supports_spirv`/`supports_dxil`, par défaut `false`). Les
   nouveaux backends matériels implémentent ce contrat et étendent
   l'enum `KernelSource` de façon non destructive (ex.
   `Dxil(Vec<u8>)`).
2. **Schéma en deux étapes « mock → matériel réel »** : Phase 1
   (dispositif simulé, fonctionne sans matériel, vérifie les
   contrats même en CI sans GPU), Phase 1,5–2 (implémentation réelle
   derrière une feature Cargo, désactivée par défaut, s'auto-ignore
   honnêtement en l'absence de matériel réel).
3. **Piège du padding des tableaux dans les cbuffer HLSL** : les
   tableaux scalaires dans `cbuffer` sont alignés sur des frontières
   de 16 octets — des constantes Rust/C++ compactées ne correspondent
   plus à la disposition HLSL (a réellement causé un bug « la sortie
   reste en clair » dans le kernel ChaCha20). Solution : champs
   scalaires individuels (`key0`…`key7`) plutôt que des déclarations
   de tableau.
4. **Root signature intégrée dans HLSL** : l'attribut
   `[RootSignature(...)]` écrit directement dans le shader permet à
   `dxc` d'intégrer la root signature dans le DXIL — évite de
   construire manuellement le descripteur de root signature côté
   Rust/C++. Lier directement les descripteurs UAV root plutôt que
   passer par un descriptor heap.
5. **Énumération des adaptateurs DXGI pour détecter le fabricant** :
   `D3D12CreateDevice(None, ...)` ne fournit aucune information sur
   le fabricant ni la VRAM. `IDXGIFactory1::EnumAdapters1(0)` →
   `DXGI_ADAPTER_DESC1` fournit l'ID fabricant PCIe (NVIDIA/AMD/
   Intel). En cas d'échec, revenir en sécurité sur `None`.
6. **Avertissement honnête sur la compression/le chiffrement GPU** :
   pour de petites charges utiles (taille MTU réseau), la surcharge
   de transfert Host↔Device peut annuler l'avantage de calcul du GPU
   — faire de vrais benchmarks sur la taille de charge utile visée
   avant intégration.
7. **Schéma des kernels de parité RAID6** : un nombre variable de
   disques de données regroupés en un seul buffer concaténé (plutôt
   que des bindings séparés par disque). Multiplication GF(2^8) pour
   la Q-parity (Reed-Solomon) sous forme de fonction `gf_mul`
   autonome (multiplication « à la russe », polynôme irréductible
   `0x11D`) dans le shader.
8. **Implémentation GPU indépendante des entiers 64 bits** : les
   entiers 64 bits sont une fonctionnalité matérielle optionnelle en
   DXIL SM6.0. Poly1305 implémente la multiplication/addition/
   décalage par paires 32×32→64 bits en n'utilisant que des
   opérations 32 bits — schéma utile pour porter de la crypto/des
   grands nombres vers des cibles 64 bits non garanties.
9. **Compression du cache KV façon MLA de DeepSeek-V3** : projection
   descendante/ascendante bâtie sur le `sgemm` existant.
   **Divulgation honnête** : les matrices de projection sont
   uniquement initialisées aléatoirement (aucun poids entraîné) — la
   compression est avec perte ; cela prouve seulement que le chemin
   de calcul est correctement câblé, pas que la qualité de
   génération est préservée.
10. **Pénalité de répétition (style CTRL)** : `logit>0` → `/penalty`,
    `logit<=0` → `*penalty`, appliquée aux tokens déjà vus avant
    l'argmax. `penalty=1.0` est un raccourci qui préserve exactement
    le comportement existant de `generate()` (aucune régression).
    Valeur par défaut empirique `1,3` — à recalibrer selon la
    structure du prompt.

**État actuel** : workspace Cargo composé de `opencuda-core`/
`opencuda-cpu`/`opencuda-vulkan`/`opencuda-directx`/`opencuda-blas`/
`open-cuda-bert`/`open-cuda-llm`. `opencuda-directx` implémenté
jusqu'à la Phase 2 (vector_add/matmul/ChaCha20/Poly1305 vérifiés sur
matériel réel). Kernels de parité RAID6 P/Q ajoutés à
`opencuda-vulkan`, vérifiés sur matériel réel. Détails dans les
entrées HANDOFF de [CLAUDE.md](CLAUDE.md).

## 21/11/2026 Choix automatique de la précision et faible mémoire (open-cuda, aruaru-llm)
- **Précision des poids** : sur une machine de 32 Go sans GPU, garder les poids en INT8 est le choix pratique (environ 4 fois plus rapide que le FP8 ; un modèle 7B produit un jeton en 2 secondes environ). On peut aussi passer seulement les couches centrales en INT4, mais la qualité a baissé d'environ 17 % sur un autre texte. Le FP8 est lent, donc il n'est pas le choix par défaut.
- **Choix automatique** : `PrecisionTable::auto_for_memory` retient, dans l'ordre f32, tout en INT8, INT4 sur les couches centrales, la première table qui tient dans 70 % de la RAM libre. Dans `aruaru-llm`, on l'indique avec la variable d'environnement `ARUARU_LLM_PRECISION` (non définie = f32, `auto`, ou une table comme `int8,1:f32`).
- **Cache KV en INT8** : `QWEN_KV_INT8=1` ne simule que l'effet de l'arrondi ; la perplexité s'est dégradée d'environ 1 %. La mémoire n'est pas encore réduite.
- **À faire** : stocker réellement le cache KV en INT8, et exécuter `auto` de bout en bout sur un vrai serveur. Les traductions sont faites par une IA et n'ont pas encore été relues par des locuteurs natifs.

## 21/11/2026 Mise à jour : le cache KV est vraiment stocké en INT8
- Avec `QWEN_KV_INT8=1`, le cache KV est désormais **réellement stocké** en INT8, avec un facteur d'échelle par tête et par jeton (`open-cuda` 2ba4b5f). Sur Qwen2.5-1.5B, la perplexité passe de 7,079 à 7,159 (environ +1 %) ; sur un autre texte, de 12,1 à 11,8 (dans le bruit). La mémoire du cache KV représente environ le quart de celle du f32 (un test avec head_dim 64 montre une réduction de plus de 3 fois).
- À faire : un essai complet de `ARUARU_LLM_PRECISION=auto` sur un vrai serveur. Les traductions sont faites par une IA et n'ont pas encore été relues par des locuteurs natifs.
