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
