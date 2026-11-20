# POLICY (open-cuda)

## 日本語
- 重みの精度は `Precision`(F32/FP8/INT8/INT4)で層ごとに選ぶ。FP64以上は試験の基準値としてだけ使い、本番の推論には使わない。
- 量子化の効果は必ず実測(迷い度・別の文での迷い度・生成秒・メモリ)で確かめ、悪い結果も正直に記録する。
- 新しいRESTは作らない(Connect/gRPCのみ)。JSONはRS-JSONで読み書きする。秘密は書かない。
- 関連: `aruaru-llm/docs/LOW-VRAM-KNOWHOW.md`。姉妹リポジトリ(aruaru-llm・open-cpu・open-directx・pl2pl)は同時に更新する。

## English
- Weight precision is chosen per layer with `Precision` (F32/FP8/INT8/INT4). FP64 and wider are only test oracles, never used for production inference.
- Always verify the effect of quantization by measurement (perplexity, held-out perplexity, generation seconds, memory) and record bad results honestly.
- No new REST (Connect/gRPC only). JSON is read and written with RS-JSON. Never write secrets.
- See `aruaru-llm/docs/LOW-VRAM-KNOWHOW.md`. Sister repositories (aruaru-llm, open-cpu, open-directx, pl2pl) are updated together.


## 2026-11-20 現状と完成予定(10か国語) / Status and plan in 10 languages
(AI翻訳、ネイティブ確認前 / AI translation, not yet checked by native speakers)

### 日本語
**現状**: 重みの精度を層ごとに選べます(`QWEN_PRECISION="int8,1:f32,24:fp8"`、F32 / FP8 / INT8 / INT4)。CPUだけで実測済み。1.5B: f32 迷い度7.08、FP8 7.27、INT8 7.37(27秒)、INT4 10.55(29秒)。3B: INT8 8.05(40秒)、INT4 10.76(46秒)。重みメモリはINT8で約1/4、INT4で約1/8。速さはf32と同程度。
**完成予定**: ①層の感度から精度表を自動で決める ②KVキャッシュの量子化 ③FP8/INT8/INT4のSIMD高速化 ④起動時にRAMを調べてモデルと精度表を自動選択 ⑤7BのINT4測定 ⑥`pl2pl`の新版公開。

### English
**Status**: Weight precision is selectable per layer (`QWEN_PRECISION="int8,1:f32,24:fp8"`; F32 / FP8 / INT8 / INT4), measured on CPU only. 1.5B: f32 perplexity 7.08, FP8 7.27, INT8 7.37 (27 s), INT4 10.55 (29 s). 3B: INT8 8.05 (40 s), INT4 10.76 (46 s). Weight memory is about 1/4 (INT8) and 1/8 (INT4); speed is about the same as f32.
**Planned**: (1) choose the precision table automatically from layer sensitivity; (2) KV-cache quantization; (3) SIMD speed-up for FP8/INT8/INT4; (4) detect RAM at startup and pick model and table; (5) measure 7B INT4; (6) release a new `pl2pl`.

### 简体中文
**现状**:权重精度可按层选择(`QWEN_PRECISION="int8,1:f32,24:fp8"`,F32 / FP8 / INT8 / INT4),仅在CPU上实测。1.5B:f32困惑度7.08,FP8 7.27,INT8 7.37(27秒),INT4 10.55(29秒)。3B:INT8 8.05(40秒),INT4 10.76(46秒)。权重内存约为INT8的1/4、INT4的1/8,速度与f32相当。
**计划**:①按层敏感度自动决定精度表 ②KV缓存量化 ③FP8/INT8/INT4的SIMD加速 ④启动时检测内存并自动选择模型与精度表 ⑤测量7B的INT4 ⑥发布新版`pl2pl`。

### 繁體中文
**現況**:權重精度可依層選擇(`QWEN_PRECISION="int8,1:f32,24:fp8"`,F32 / FP8 / INT8 / INT4),僅在CPU上實測。1.5B:f32困惑度7.08,FP8 7.27,INT8 7.37(27秒),INT4 10.55(29秒)。3B:INT8 8.05(40秒),INT4 10.76(46秒)。權重記憶體約為INT8的1/4、INT4的1/8,速度與f32相當。
**計畫**:①依層敏感度自動決定精度表 ②KV快取量化 ③FP8/INT8/INT4的SIMD加速 ④啟動時偵測記憶體並自動選擇模型與精度表 ⑤測量7B的INT4 ⑥發布新版`pl2pl`。

### 한국어
**현황**: 가중치 정밀도를 층마다 고를 수 있습니다(`QWEN_PRECISION="int8,1:f32,24:fp8"`, F32 / FP8 / INT8 / INT4). CPU만으로 실측했습니다. 1.5B: f32 혼란도 7.08, FP8 7.27, INT8 7.37(27초), INT4 10.55(29초). 3B: INT8 8.05(40초), INT4 10.76(46초). 가중치 메모리는 INT8에서 약 1/4, INT4에서 약 1/8이며 속도는 f32와 비슷합니다.
**예정**: ①층별 민감도로 정밀도 표 자동 결정 ②KV 캐시 양자화 ③FP8/INT8/INT4 SIMD 가속 ④시작 시 RAM을 확인해 모델과 정밀도 표 자동 선택 ⑤7B INT4 측정 ⑥`pl2pl` 새 버전 공개.

### Deutsch
**Stand**: Die Gewichtspräzision ist pro Schicht wählbar (`QWEN_PRECISION="int8,1:f32,24:fp8"`; F32 / FP8 / INT8 / INT4), nur auf der CPU gemessen. 1.5B: f32-Perplexität 7,08, FP8 7,27, INT8 7,37 (27 s), INT4 10,55 (29 s). 3B: INT8 8,05 (40 s), INT4 10,76 (46 s). Der Gewichtsspeicher sinkt auf etwa 1/4 (INT8) bzw. 1/8 (INT4); die Geschwindigkeit entspricht etwa f32.
**Geplant**: ①Präzisionstabelle automatisch aus der Schichtempfindlichkeit bestimmen ②KV-Cache-Quantisierung ③SIMD-Beschleunigung für FP8/INT8/INT4 ④beim Start RAM prüfen und Modell/Tabelle wählen ⑤7B mit INT4 messen ⑥neue `pl2pl`-Version veröffentlichen.

### Français
**État** : la précision des poids se choisit par couche (`QWEN_PRECISION="int8,1:f32,24:fp8"` ; F32 / FP8 / INT8 / INT4), mesurée sur CPU seulement. 1.5B : perplexité f32 7,08, FP8 7,27, INT8 7,37 (27 s), INT4 10,55 (29 s). 3B : INT8 8,05 (40 s), INT4 10,76 (46 s). La mémoire des poids tombe à environ 1/4 (INT8) et 1/8 (INT4) ; la vitesse est proche de f32.
**Prévu** : ①choisir automatiquement la table de précision selon la sensibilité des couches ②quantification du cache KV ③accélération SIMD de FP8/INT8/INT4 ④détecter la RAM au démarrage et choisir modèle et table ⑤mesurer le 7B en INT4 ⑥publier une nouvelle version de `pl2pl`.

### Español
**Estado**: la precisión de los pesos se elige por capa (`QWEN_PRECISION="int8,1:f32,24:fp8"`; F32 / FP8 / INT8 / INT4), medida solo en CPU. 1.5B: perplejidad f32 7,08, FP8 7,27, INT8 7,37 (27 s), INT4 10,55 (29 s). 3B: INT8 8,05 (40 s), INT4 10,76 (46 s). La memoria de los pesos baja a cerca de 1/4 (INT8) y 1/8 (INT4); la velocidad es parecida a f32.
**Previsto**: ①decidir la tabla de precisión automáticamente según la sensibilidad de las capas ②cuantización de la caché KV ③aceleración SIMD de FP8/INT8/INT4 ④detectar la RAM al iniciar y elegir modelo y tabla ⑤medir 7B en INT4 ⑥publicar una nueva versión de `pl2pl`.

### Português
**Estado**: a precisão dos pesos pode ser escolhida por camada (`QWEN_PRECISION="int8,1:f32,24:fp8"`; F32 / FP8 / INT8 / INT4), medida apenas na CPU. 1.5B: perplexidade f32 7,08, FP8 7,27, INT8 7,37 (27 s), INT4 10,55 (29 s). 3B: INT8 8,05 (40 s), INT4 10,76 (46 s). A memória dos pesos cai para cerca de 1/4 (INT8) e 1/8 (INT4); a velocidade é parecida com a do f32.
**Planejado**: ①definir a tabela de precisão automaticamente pela sensibilidade das camadas ②quantização do cache KV ③aceleração SIMD de FP8/INT8/INT4 ④detectar a RAM na inicialização e escolher modelo e tabela ⑤medir o 7B em INT4 ⑥publicar uma nova versão do `pl2pl`.

### Русский
**Состояние**: точность весов выбирается по слоям (`QWEN_PRECISION="int8,1:f32,24:fp8"`; F32 / FP8 / INT8 / INT4), измерено только на CPU. 1.5B: перплексия f32 7,08, FP8 7,27, INT8 7,37 (27 с), INT4 10,55 (29 с). 3B: INT8 8,05 (40 с), INT4 10,76 (46 с). Память весов примерно 1/4 (INT8) и 1/8 (INT4); скорость близка к f32.
**Планы**: ①автоматически определять таблицу точности по чувствительности слоёв ②квантование KV-кэша ③SIMD-ускорение FP8/INT8/INT4 ④при запуске проверять ОЗУ и выбирать модель и таблицу ⑤измерить 7B в INT4 ⑥выпустить новую версию `pl2pl`.

## 2026-11-21 方針 / Policy: low-memory serving
日本語: 小さなPCでは、品質を守れる範囲で重みをINT8(必要なら中間層のみINT4)にする。FP8は遅いので既定にしない。未測定の効果を成果として書かない。
English: On small machines, keep weights at INT8 (INT4 only on middle layers if needed) while quality holds. FP8 is slow, so it is not the default. Never report unmeasured gains as results.
