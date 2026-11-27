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

## 2026-11-21 精度の自動選択と省メモリ(open-cuda・aruaru-llm)
- **重みの精度**: CPUだけの32GB機では、重みをINT8で持つのが実用的です(FP8より約4倍速く、7Bが1トークン約2秒)。中間層だけINT4にもできますが、別の文では約17%悪化しました。FP8は遅いので既定にしません。
- **自動選択**: `PrecisionTable::auto_for_memory` が、f32、全層INT8、中間層INT4の順に、空きRAMの7割に収まる最初の表を選びます。`aruaru-llm` では環境変数 `ARUARU_LLM_PRECISION`(未設定=f32、`auto`、または `int8,1:f32` のような表)で指定します。
- **KVのINT8**: `QWEN_KV_INT8=1` は丸めの影響だけを測る試験で、迷い度は約1%の悪化でした。メモリはまだ減りません。
- **未了**: KVを実際にINT8で保持する実装、実サーバーでの `auto` の通し確認。翻訳はAIによるもので、母語話者の確認はまだです。

## 2026-11-21 Automatic precision selection and low memory (open-cuda, aruaru-llm)
- **Weight precision**: on a CPU-only 32 GB machine, INT8 weights are the practical choice (about 4x faster than FP8; 7B runs at roughly 2 seconds per token). INT4 on the middle layers is possible but was about 17% worse on held-out text. FP8 is slow, so it is not the default.
- **Automatic choice**: `PrecisionTable::auto_for_memory` picks the first table that fits 70% of free RAM, in the order f32, all-INT8, INT4 on middle layers. In `aruaru-llm`, set the environment variable `ARUARU_LLM_PRECISION` (unset = f32, `auto`, or a table such as `int8,1:f32`).
- **INT8 KV cache**: `QWEN_KV_INT8=1` only simulates the rounding; perplexity got about 1% worse. Memory is not reduced yet.
- **Open**: actually storing the KV cache as INT8, and an end-to-end run of `auto` on a real server. Translations are AI-made and not yet checked by native speakers.

## 2026-11-21 精度自动选择与节省内存(open-cuda、aruaru-llm)
- **权重精度**:在只有CPU的32GB机器上,用INT8保存权重最实用(比FP8快约4倍,7B模型每个词元约2秒)。也可以只把中间层改为INT4,但在另一段文字上质量下降约17%。FP8较慢,因此不作为默认。
- **自动选择**:`PrecisionTable::auto_for_memory` 按"f32、全部INT8、中间层INT4"的顺序,选出第一个能放进可用内存70%的方案。在 `aruaru-llm` 中用环境变量 `ARUARU_LLM_PRECISION` 指定(未设置=f32、`auto`,或形如 `int8,1:f32` 的表)。
- **KV缓存INT8**:`QWEN_KV_INT8=1` 只是模拟舍入的影响,困惑度约变差1%,内存尚未减少。
- **未完成**:真正以INT8保存KV缓存,以及在真实服务器上完整测试 `auto`。翻译由AI完成,尚未经母语者确认。

## 2026-11-21 精度自動選擇與節省記憶體(open-cuda、aruaru-llm)
- **權重精度**:在只有CPU的32GB機器上,以INT8保存權重最實用(比FP8快約4倍,7B模型每個詞元約2秒)。也可以只把中間層改為INT4,但在另一段文字上品質下降約17%。FP8較慢,因此不作為預設。
- **自動選擇**:`PrecisionTable::auto_for_memory` 依「f32、全部INT8、中間層INT4」的順序,選出第一個能放進可用記憶體70%的方案。在 `aruaru-llm` 中以環境變數 `ARUARU_LLM_PRECISION` 指定(未設定=f32、`auto`,或形如 `int8,1:f32` 的表)。
- **KV快取INT8**:`QWEN_KV_INT8=1` 只是模擬捨入的影響,困惑度約變差1%,記憶體尚未減少。
- **未完成**:真正以INT8保存KV快取,以及在真實伺服器上完整測試 `auto`。翻譯由AI完成,尚未經母語人士確認。

## 2026-11-21 정밀도 자동 선택과 메모리 절약 (open-cuda, aruaru-llm)
- **가중치 정밀도**: CPU만 있는 32GB 컴퓨터에서는 가중치를 INT8로 두는 것이 실용적입니다(FP8보다 약 4배 빠르고, 7B 모델이 토큰당 약 2초). 중간 층만 INT4로 바꿀 수도 있지만, 다른 문장에서는 품질이 약 17% 나빠졌습니다. FP8은 느려서 기본값으로 쓰지 않습니다.
- **자동 선택**: `PrecisionTable::auto_for_memory`가 f32, 전 층 INT8, 중간 층 INT4 순으로, 남은 RAM의 70%에 들어가는 첫 번째 구성을 고릅니다. `aruaru-llm`에서는 환경 변수 `ARUARU_LLM_PRECISION`으로 지정합니다(미설정=f32, `auto`, 또는 `int8,1:f32` 같은 표).
- **KV 캐시 INT8**: `QWEN_KV_INT8=1`은 반올림의 영향만 재는 시험이며, 혼란도(perplexity)는 약 1% 나빠졌습니다. 메모리는 아직 줄지 않았습니다.
- **미완료**: KV 캐시를 실제로 INT8로 저장하는 구현, 실제 서버에서 `auto`를 끝까지 돌려 보는 시험. 번역은 AI가 했으며 원어민 확인은 아직입니다.

## 21.11.2026 Automatische Genauigkeitswahl und geringer Speicherbedarf (open-cuda, aruaru-llm)
- **Genauigkeit der Gewichte**: Auf einem Rechner mit 32 GB und nur CPU sind INT8-Gewichte die praktische Wahl (etwa viermal schneller als FP8; ein 7B-Modell braucht ungefähr 2 Sekunden pro Token). INT4 nur in den mittleren Schichten ist möglich, war bei anderem Text aber etwa 17 % schlechter. FP8 ist langsam und daher nicht der Standard.
- **Automatische Wahl**: `PrecisionTable::auto_for_memory` nimmt in der Reihenfolge f32, alles INT8, INT4 in den mittleren Schichten die erste Tabelle, die in 70 % des freien Arbeitsspeichers passt. In `aruaru-llm` steuert man das über die Umgebungsvariable `ARUARU_LLM_PRECISION` (nicht gesetzt = f32, `auto` oder eine Tabelle wie `int8,1:f32`).
- **INT8-KV-Cache**: `QWEN_KV_INT8=1` simuliert nur die Rundung; die Perplexität wurde etwa 1 % schlechter. Der Speicherbedarf sinkt dadurch noch nicht.
- **Offen**: den KV-Cache tatsächlich als INT8 speichern und `auto` auf einem echten Server vollständig durchlaufen lassen. Die Übersetzungen stammen von einer KI und wurden noch nicht von Muttersprachlern geprüft.

## 21/11/2026 Choix automatique de la précision et faible mémoire (open-cuda, aruaru-llm)
- **Précision des poids** : sur une machine de 32 Go sans GPU, garder les poids en INT8 est le choix pratique (environ 4 fois plus rapide que le FP8 ; un modèle 7B produit un jeton en 2 secondes environ). On peut aussi passer seulement les couches centrales en INT4, mais la qualité a baissé d'environ 17 % sur un autre texte. Le FP8 est lent, donc il n'est pas le choix par défaut.
- **Choix automatique** : `PrecisionTable::auto_for_memory` retient, dans l'ordre f32, tout en INT8, INT4 sur les couches centrales, la première table qui tient dans 70 % de la RAM libre. Dans `aruaru-llm`, on l'indique avec la variable d'environnement `ARUARU_LLM_PRECISION` (non définie = f32, `auto`, ou une table comme `int8,1:f32`).
- **Cache KV en INT8** : `QWEN_KV_INT8=1` ne simule que l'effet de l'arrondi ; la perplexité s'est dégradée d'environ 1 %. La mémoire n'est pas encore réduite.
- **À faire** : stocker réellement le cache KV en INT8, et exécuter `auto` de bout en bout sur un vrai serveur. Les traductions sont faites par une IA et n'ont pas encore été relues par des locuteurs natifs.

## 21-11-2026 Elección automática de precisión y poca memoria (open-cuda, aruaru-llm)
- **Precisión de los pesos**: en un equipo de 32 GB sin GPU, lo práctico es guardar los pesos en INT8 (unas 4 veces más rápido que FP8; un modelo 7B tarda unos 2 segundos por token). También se puede usar INT4 solo en las capas centrales, pero con otro texto la calidad empeoró cerca de un 17 %. FP8 es lento, así que no es el valor por defecto.
- **Elección automática**: `PrecisionTable::auto_for_memory` elige, en el orden f32, todo INT8, INT4 en las capas centrales, la primera tabla que cabe en el 70 % de la RAM libre. En `aruaru-llm` se indica con la variable de entorno `ARUARU_LLM_PRECISION` (sin definir = f32, `auto`, o una tabla como `int8,1:f32`).
- **Caché KV en INT8**: `QWEN_KV_INT8=1` solo simula el efecto del redondeo; la perplejidad empeoró cerca de un 1 %. La memoria aún no se reduce.
- **Pendiente**: guardar de verdad la caché KV en INT8 y ejecutar `auto` de principio a fin en un servidor real. Las traducciones las hizo una IA y aún no las han revisado hablantes nativos.

## 21/11/2026 Escolha automática de precisão e pouca memória (open-cuda, aruaru-llm)
- **Precisão dos pesos**: num computador de 32 GB sem GPU, guardar os pesos em INT8 é a escolha prática (cerca de 4 vezes mais rápido que FP8; um modelo 7B leva uns 2 segundos por token). Também é possível usar INT4 só nas camadas do meio, mas com outro texto a qualidade piorou cerca de 17 %. O FP8 é lento, por isso não é o padrão.
- **Escolha automática**: `PrecisionTable::auto_for_memory` escolhe, na ordem f32, tudo em INT8, INT4 nas camadas do meio, a primeira tabela que cabe em 70 % da RAM livre. No `aruaru-llm`, indica-se com a variável de ambiente `ARUARU_LLM_PRECISION` (não definida = f32, `auto`, ou uma tabela como `int8,1:f32`).
- **Cache KV em INT8**: `QWEN_KV_INT8=1` apenas simula o efeito do arredondamento; a perplexidade piorou cerca de 1 %. A memória ainda não diminui.
- **Pendente**: guardar de fato o cache KV em INT8 e executar o `auto` de ponta a ponta num servidor real. As traduções foram feitas por uma IA e ainda não foram revistas por falantes nativos.

## 21/11/2026 Scelta automatica della precisione e poca memoria (open-cuda, aruaru-llm)
- **Precisione dei pesi**: su un computer da 32 GB senza GPU, tenere i pesi in INT8 è la scelta pratica (circa 4 volte più veloce di FP8; un modello 7B impiega circa 2 secondi per token). Si può usare INT4 solo nei livelli centrali, ma su un altro testo la qualità è peggiorata di circa il 17 %. FP8 è lento, quindi non è l'impostazione predefinita.
- **Scelta automatica**: `PrecisionTable::auto_for_memory` sceglie, nell'ordine f32, tutto INT8, INT4 nei livelli centrali, la prima tabella che sta nel 70 % della RAM libera. In `aruaru-llm` si indica con la variabile d'ambiente `ARUARU_LLM_PRECISION` (non impostata = f32, `auto`, oppure una tabella come `int8,1:f32`).
- **Cache KV in INT8**: `QWEN_KV_INT8=1` simula soltanto l'effetto dell'arrotondamento; la perplessità è peggiorata di circa l'1 %. La memoria non è ancora ridotta.
- **Da fare**: salvare davvero la cache KV in INT8 ed eseguire `auto` da capo a fondo su un server reale. Le traduzioni sono state fatte da un'IA e non sono ancora state verificate da madrelingua.

## 21.11.2026 Автоматический выбор точности и экономия памяти (open-cuda, aruaru-llm)
- **Точность весов**: на компьютере с 32 ГБ без видеокарты практичнее хранить веса в INT8 (примерно в 4 раза быстрее FP8; модель 7B выдаёт токен примерно за 2 секунды). Можно перевести в INT4 только средние слои, но на другом тексте качество ухудшилось примерно на 17 %. FP8 работает медленно, поэтому по умолчанию не используется.
- **Автоматический выбор**: `PrecisionTable::auto_for_memory` берёт первую таблицу, которая помещается в 70 % свободной оперативной памяти, в порядке: f32, всё в INT8, INT4 в средних слоях. В `aruaru-llm` это задаётся переменной окружения `ARUARU_LLM_PRECISION` (не задана = f32, `auto` или таблица вроде `int8,1:f32`).
- **KV-кэш в INT8**: `QWEN_KV_INT8=1` лишь имитирует влияние округления; перплексия ухудшилась примерно на 1 %. Памяти пока не стало меньше.
- **Не сделано**: по-настоящему хранить KV-кэш в INT8 и пройти `auto` от начала до конца на реальном сервере. Переводы сделаны ИИ и ещё не проверены носителями языка.

## 21.11.2026 Автоматичний вибір точності та економія пам'яті (open-cuda, aruaru-llm)
- **Точність ваг**: на комп'ютері з 32 ГБ без відеокарти практичніше зберігати ваги в INT8 (приблизно у 4 рази швидше за FP8; модель 7B видає токен приблизно за 2 секунди). Можна перевести в INT4 лише середні шари, але на іншому тексті якість погіршилася приблизно на 17 %. FP8 працює повільно, тому типово не використовується.
- **Автоматичний вибір**: `PrecisionTable::auto_for_memory` бере першу таблицю, що вміщується у 70 % вільної оперативної пам'яті, у порядку: f32, усе в INT8, INT4 у середніх шарах. В `aruaru-llm` це задається змінною середовища `ARUARU_LLM_PRECISION` (не задана = f32, `auto` або таблиця на кшталт `int8,1:f32`).
- **KV-кеш в INT8**: `QWEN_KV_INT8=1` лише імітує вплив округлення; перплексія погіршилася приблизно на 1 %. Пам'яті поки що не побільшало.
- **Не зроблено**: справді зберігати KV-кеш в INT8 і пройти `auto` від початку до кінця на реальному сервері. Переклади зробив ШІ, їх ще не перевіряли носії мови.

## 21.11.2026 בחירה אוטומטית של דיוק וחיסכון בזיכרון (open-cuda, aruaru-llm)
- **דיוק המשקלים**: במחשב עם 32 ג״ב וללא כרטיס מסך, הפתרון המעשי הוא לשמור את המשקלים ב-INT8 (מהיר פי 4 בערך מ-FP8; מודל 7B מפיק אסימון אחד תוך כ-2 שניות). אפשר גם להעביר ל-INT4 רק את השכבות האמצעיות, אך בטקסט אחר האיכות ירדה בכ-17%. FP8 איטי, ולכן הוא אינו ברירת המחדל.
- **בחירה אוטומטית**: `PrecisionTable::auto_for_memory` בוחרת, לפי הסדר f32, הכול ב-INT8, INT4 בשכבות האמצעיות, את הטבלה הראשונה שנכנסת ל-70% מהזיכרון הפנוי. ב-`aruaru-llm` מגדירים זאת במשתנה הסביבה `ARUARU_LLM_PRECISION` (לא מוגדר = f32, `auto`, או טבלה כמו `int8,1:f32`).
- **מטמון KV ב-INT8**: `QWEN_KV_INT8=1` רק מדמה את השפעת העיגול; הפרפלקסיה החמירה בכ-1%. הזיכרון עדיין לא קטן.
- **טרם הושלם**: שמירה אמיתית של מטמון ה-KV ב-INT8, והרצה מלאה של `auto` על שרת אמיתי. התרגומים נעשו בידי בינה מלאכותית וטרם נבדקו על ידי דוברי שפת אם.

## ٢١‏/١١‏/٢٠٢٦ الاختيار التلقائي للدقة وتوفير الذاكرة (open-cuda وaruaru-llm)
- **دقة الأوزان**: على جهاز بذاكرة ٣٢ غيغابايت وبلا بطاقة رسوميات، يكون الأنسب حفظ الأوزان بصيغة INT8 (أسرع بنحو ٤ مرات من FP8، ونموذج 7B يُنتج الرمز الواحد في نحو ثانيتين). يمكن أيضاً استخدام INT4 في الطبقات الوسطى فقط، لكن الجودة تراجعت بنحو ١٧٪ على نص آخر. صيغة FP8 بطيئة، لذلك ليست الخيار الافتراضي.
- **الاختيار التلقائي**: تختار `PrecisionTable::auto_for_memory` أول جدول يتسع في ٧٠٪ من الذاكرة الحرة، بهذا الترتيب: f32، ثم كل الطبقات INT8، ثم INT4 في الطبقات الوسطى. في `aruaru-llm` تُحدَّد عبر متغير البيئة `ARUARU_LLM_PRECISION` (غير معيَّن = f32، أو `auto`، أو جدول مثل `int8,1:f32`).
- **ذاكرة KV بصيغة INT8**: الخيار `QWEN_KV_INT8=1` يحاكي أثر التقريب فقط؛ ازداد مقياس الحيرة (perplexity) سوءاً بنحو ١٪. ولم تنخفض الذاكرة بعد.
- **لم يكتمل بعد**: حفظ ذاكرة KV فعلياً بصيغة INT8، وتشغيل `auto` من البداية إلى النهاية على خادم حقيقي. الترجمات أنجزها الذكاء الاصطناعي ولم يراجعها ناطقون أصليون بعد.

## ۲۱‏/۱۱‏/۲۰۲۶ انتخاب خودکار دقت و صرفه‌جویی در حافظه (open-cuda و aruaru-llm)
- **دقت وزن‌ها**: روی رایانه‌ای با ۳۲ گیگابایت حافظه و بدون کارت گرافیک، نگه‌داشتن وزن‌ها با INT8 عملی‌ترین انتخاب است (حدود ۴ برابر سریع‌تر از FP8؛ مدل 7B هر توکن را در حدود ۲ ثانیه تولید می‌کند). می‌توان فقط لایه‌های میانی را INT4 کرد، اما روی متنی دیگر کیفیت حدود ۱۷٪ بدتر شد. FP8 کند است و به همین دلیل پیش‌فرض نیست.
- **انتخاب خودکار**: `PrecisionTable::auto_for_memory` به ترتیب f32، همه‌چیز INT8 و INT4 در لایه‌های میانی، نخستین جدولی را برمی‌گزیند که در ۷۰٪ حافظهٔ آزاد جا شود. در `aruaru-llm` آن را با متغیر محیطی `ARUARU_LLM_PRECISION` تعیین می‌کنید (تنظیم‌نشده = f32، `auto`، یا جدولی مانند `int8,1:f32`).
- **حافظهٔ نهان KV با INT8**: گزینهٔ `QWEN_KV_INT8=1` فقط اثر گرد کردن را شبیه‌سازی می‌کند؛ سرگشتگی (perplexity) حدود ۱٪ بدتر شد. هنوز حافظه کم نشده است.
- **انجام‌نشده**: ذخیرهٔ واقعی حافظهٔ نهان KV با INT8 و اجرای کامل `auto` روی یک سرور واقعی. ترجمه‌ها را هوش مصنوعی انجام داده و هنوز فارسی‌زبانان بومی آن‌ها را بررسی نکرده‌اند.

## 2026-11-21 追記: KVキャッシュを実際にINT8で保持
- `QWEN_KV_INT8=1` で、KVキャッシュを1ヘッド1トークンごとのスケール付きINT8として**実際に保持**するようにした(`open-cuda` 2ba4b5f)。Qwen2.5-1.5Bで迷い度は7.079→7.159(約+1%)、別の文では12.1→11.8(誤差の範囲)。KVのメモリはf32の約1/4(head_dim 64の試験で3倍以上の縮小を確認)。
- 未了: `ARUARU_LLM_PRECISION=auto` を実サーバーで通して確認すること。翻訳はAIが作成し、ネイティブ話者による確認は未了。

## Versionless API の方針 / Versionless API policy (2026-11-25)
**日本語**: このリポジトリのAPIは **Versionless**(URL・パッケージ名に `v1` 等を付けず、互換性を壊さず足すだけで進化させる)。欄は足すだけ・足す欄は省略可能・知らない欄は無視・名前変更は新名を足して旧を非推奨に(期限は既定で6か月以上、使用0を確認してから削除)・enumは未知の値を許容・安全上の緊急修正は例外。既存の `.v1` は呼び出し側を壊さないよう当面残し、版なしの別名を追加して段階的に畳む。詳細・出典は [`runo/API-POLICY.md`](https://github.com/aon-co-jp/runo/blob/main/API-POLICY.md) 末尾。適用はこれから順に行う。
**English**: This repository's APIs are **Versionless**: no `v1` in URLs or package names; evolve by adding only, without breaking compatibility. Only add fields, new fields are optional, ignore unknown fields, rename by adding the new name and deprecating the old (default deadline 6 months or more, remove only after confirming zero use), tolerate unknown enum values, emergency security fixes are the exception. Existing `.v1` names stay for now so callers do not break; a versionless alias is added and `.v1` is retired in stages. Details and sources: the end of [`runo/API-POLICY.md`](https://github.com/aon-co-jp/runo/blob/main/API-POLICY.md). Rollout is step by step.

## API 方針: REST 全廃・gRPC / Connect へ
**日本語**: このリポジトリでは **REST API を新規に作らない**。ブラウザ・`curl` から呼ぶ API は **Connect の単純呼び出し**(`POST /<パッケージ>.<Service>/<Method>`、`content-type: application/json`)、サービス間は **gRPC / gRPC-web**。GraphQL は読み取り専用。既存の REST は触るときに移し、移したら旧経路は消す。JSON は RS-JSON(`from_str_secure` / `parse_secure`)で読む。例外は相手の規格が形を決めるもの(ACME、OAuth/OIDC のコールバック、第三者の Webhook、ヘルスチェック、静的ファイル、OTLP)のみ。迷ったら Connect。正本は [`runo/API-POLICY.md`](https://github.com/aon-co-jp/runo/blob/main/API-POLICY.md)。
**English**: This repository **does not add new REST APIs**. APIs called from browsers or `curl` use **Connect unary calls** (`POST /<package>.<Service>/<Method>`, `content-type: application/json`); service-to-service calls use **gRPC / gRPC-web**. GraphQL is read-only. Existing REST is migrated when touched and the old route is removed. Read JSON with RS-JSON (`from_str_secure` / `parse_secure`). Exceptions are only shapes dictated by someone else's standard (ACME, OAuth/OIDC callbacks, third-party webhooks, health checks, static files, OTLP). When in doubt, use Connect. Source of truth: the link above.

## 完成まで作り切る方針 / Finish everything policy (2026-11-27)
**日本語**: 全リポジトリで、**未確認は自動で確認し、未実施は実施し、未了は完成させる**(確認を求めず進める)。使用頻度が低いと判断される機能や、**1から作り直す大規模開発になる内容でも、やめずに最後まで開発・TEST・DEBUGする**(別の方式で済ませない、後回しにしない)。進め方: 調査(英語・日本語の検索とGitHub) → 設計 → 実装 → 実機/実データで確認 → 失敗は原因を直して再試験 → README・CLAUDE・PORTING・POLICY を日英で更新 → push。「できた」と書くのは実測した範囲だけ、確認できなかった所は「未確認」と正直に書く。例外(確認を取る): 元に戻せない削除、秘密の公開、所有者の認証・決済・実機操作が要るもの、利用者のシステム設定の変更。
**English**: In every repository, **verify what is unverified, do what is not yet done, and finish what is unfinished** automatically, without asking. Even features that look rarely used, or work that means **a large rewrite from scratch, are developed, tested and debugged to the end** (no substituting another approach, no postponing). Method: research (English and Japanese search plus GitHub) -> design -> implement -> verify on real devices or data -> fix causes and retest on failure -> update README, CLAUDE, PORTING and POLICY in Japanese and English -> push. Write "done" only for what was measured, and say "unverified" honestly for the rest. Exceptions that still need confirmation: irreversible deletions, publishing secrets, anything needing the owner's authentication, payment or real-device actions, and changes to the user's system settings.