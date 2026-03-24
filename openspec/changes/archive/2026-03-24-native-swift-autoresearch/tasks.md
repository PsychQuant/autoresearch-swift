## 1. 專案骨架與 Swift Package 結構

- [x] 1.1 建立 Swift Package Manager 專案（Package.swift），設定 mlx-swift 依賴和模組結構（對應 design: Swift Package 結構）
- [x] 1.2 建立 Sources/ 下所有模組目錄（AutoResearch、Model、Optimizer、Config、Data、Training）
- [x] 1.3 建立 main.swift 入口，串接 config 載入 → 模型建構 → 訓練 → 評估流程
- [x] 1.4 確認 `swift build` 可編譯通過（空實作 placeholder）

## 2. Config 系統與硬體偵測

- [x] 2.1 實作 ExperimentConfig（Codable struct），支援 experiment config loading from JSON 和 config schema validation（對應 config-driven-experiment spec）
- [x] 2.2 實作 Component registry：activation function registry、optimizer 類型、attention variant 的字串→實作映射（對應 design: Config-Driven 架構搜索）
- [x] 2.3 實作 HardwareProfile：hardware detection at startup，透過 sysctl/IOKit/Metal API 讀取晶片型號、GPU 核心數、統一記憶體（對應 hardware-auto-detect spec）
- [x] 2.4 實作 chip profile database，內建 M1/M2/M3/M4 系列的預設參數映射
- [x] 2.5 實作 optional micro-benchmark（硬體自動偵測 + Micro-benchmark，~5 秒 tiny model 測 tok/sec）和 available memory check
- [x] 2.6 實作 three-layer config resolution（硬體→profile→agent override）和 resolved config output 列印

## 3. 資料管線

- [x] 3.1 實作 BPE tokenizer compatible with reference，從 `~/.cache/autoresearch/tokenizer/` 載入詞彙表和 merge rules（對應 swift-data-pipeline spec，design: Tokenizer 策略）
- [x] 3.2 實作 Parquet data shard reading，讀取 `~/.cache/autoresearch/data/` 下的 shard 檔案（對應 design: Parquet 資料讀取）
- [x] 3.3 實作 streaming DataLoader，產生 (input_tokens, target_tokens) MLXArray batches，含 epoch tracking
- [x] 3.4 實作 validation evaluation (val_bpb) 計算，使用最後一個 shard 作為驗證資料

## 4. GPT 模型

- [x] 4.1 實作 RoPE（Rotary Position Embeddings）for MLX-Swift
- [x] 4.2 實作 causal self-attention with sliding window，支援 window_pattern 配置
- [x] 4.3 實作 MLP 層，含 activation selection（relu_squared、silu、gelu）
- [x] 4.4 實作 Block（attention + MLP with pre-norm residual connections）
- [x] 4.5 實作完整 GPT model definition：embedding、blocks、lm_head、resid_lambdas、x0_lambdas、value embedding with gated residual（對應 swift-gpt-model spec）
- [x] 4.6 實作 forward pass produces loss or logits，含 logit softcapping
- [x] 4.7 實作 default model initialization（weight init scheme）

## 5. Optimizer

- [x] 5.1 實作 AdamW optimizer with float32 state、bias correction、learning rate scaling by model dimension（對應 muon-adamw-optimizer spec，design: MLX-Swift 作為唯一 ML 後端）
- [x] 5.2 實作 Muon optimizer for matrix parameters：Polar Express 正交化、NorMuon variance reduction、Nesterov momentum warmup
- [x] 5.3 實作 cautious weight decay for Muon
- [x] 5.4 實作 combined MuonAdamW optimizer，含 optimizer routing（按參數角色分配 Muon 或 AdamW）
- [x] 5.5 驗證 AdamW-only 模式可獨立運作（optimizer="adamw" config）

## 6. 訓練迴圈

- [x] 6.1 實作 fixed time-budget training 主迴圈，含 warmup exclusion（對應 training-loop spec）
- [x] 6.2 實作 gradient accumulation
- [x] 6.3 實作 learning rate schedule（warmup → constant → cooldown）
- [x] 6.4 實作 loss monitoring and early abort（loss > 100 或 NaN）
- [x] 6.5 實作 training progress logging（step、loss、lr、dt、tok/sec、epoch、remaining）
- [x] 6.6 實作 final summary output（val_bpb、training_seconds 等，匹配 reference 格式）
- [x] 6.7 實作 GC management（disable after first step, periodic collection）

## 7. Benchmark 比較機制

- [x] 7.1 實作 benchmark output format compatibility：輸出格式匹配原版 `---` key-value 格式，支援 `grep "^val_bpb:"` 解析
- [x] 7.2 實作 startup overhead measurement：分離量測啟動時間（binary launch → 第一個 training step），報告 startup_seconds
- [x] 7.3 實作 throughput comparison metrics：報告 tok/sec 和 experiments_per_hour
- [x] 7.4 實作 benchmark results tracking：`benchmark_results.tsv` 記錄 swift/python-mlx 的並排比較資料
- [x] 7.5 確保 experiment loop compatibility with program.md：Swift 版 program.md 遵循原版結構，agent 可用相同 LOOP FOREVER 流程
- [x] 7.6 確保 results.tsv format 相容原版（tab-separated: commit, val_bpb, memory_gb, status, description）

## 8. 整合與驗證

- [x] 8.1 端到端測試：完整跑一輪 5 分鐘訓練，確認產出 val_bpb
- [x] 8.2 驗證 config-driven 流程：修改 experiment.json → 重跑 → 確認參數生效
- [x] 8.3 驗證 hardware auto-detect 在 M4 Max 上正確偵測並設定預設參數
- [x] 8.4 Benchmark 對比：用 Python MLX 版和 MPS 版分別跑基準，再用 Swift 版跑相同 config，比較 tok/sec 和 startup_seconds（對應 design: Benchmark 比較機制）
- [x] 8.5 確認 startup overhead < 2 秒（排除 micro-benchmark 時間）
- [x] 8.6 撰寫 program.md（agent 指令），相容 Claude Code / Codex 工作流程
- [x] 8.7 產生 Benchmark 視覺化呈現（學習 Surya 風格）：bar chart PNG 比較各版本的 tok/sec、startup time、experiments/hour
- [x] 8.8 撰寫 README.md，Benchmark 區塊包含：圖表 → 數據表格 → 方法論 → Running your own benchmarks 教學（參考 datalab-to/surya 呈現風格）
