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
