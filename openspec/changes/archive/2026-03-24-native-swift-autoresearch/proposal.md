## Why

Karpathy 的 autoresearch 以及現有的 macOS fork（autoresearch-mlx、autoresearch-macos）全部依賴 Python runtime，帶來啟動延遲（~15-20 秒）、Python 環境管理複雜度、以及無法直接利用 Apple 原生框架（如 CoreML、Neural Engine）的限制。我們需要一個完全原生的 Swift 版本，消除 Python 依賴，透過 config-driven 架構搜索降低每輪迭代 overhead，並加入硬體自動偵測以適配不同 Apple Silicon 機型。

**核心目標**：以 `references/autoresearch/`（原版 karpathy/autoresearch）作為 benchmark 基準，在相同 5 分鐘時間預算下，Swift 版的訓練吞吐量（tok/sec）和每小時可完成實驗數必須超越 Python MLX 版。輸出格式與 `program.md` 的實驗迴圈機制必須相容原版，使 benchmark 結果可直接比較。

## What Changes

- 以 Swift Package Manager 建立全新的 CLI 專案，使用 MLX-Swift 作為核心 ML 框架
- 實作 GPT 模型（Attention、MLP、Block、Embedding）以 MLX-Swift 原生 API
- 實作 Muon + AdamW 混合 optimizer（補上 MLX Python 版缺失的 Muon）
- Agent 透過修改 `experiment.json` 來驅動架構搜索，不需修改或重新編譯 Swift 程式碼
- 啟動時自動偵測硬體規格（晶片型號、GPU 核心數、統一記憶體、頻寬），搭配可選的 micro-benchmark 來動態決定最佳預設參數
- 三層 config 覆蓋機制：硬體自動偵測（基底）→ 晶片 profile 預設 → agent 覆寫
- 純 Swift 資料管線：BPE tokenizer、Parquet 資料載入、DataLoader
- 保留 autoresearch 核心機制：固定 5 分鐘時間預算、val_bpb 指標、keep/revert 實驗迴圈

## Capabilities

### New Capabilities

- `swift-gpt-model`: 以 MLX-Swift 實作完整 GPT 模型，包含 RoPE、Sliding Window Attention、Value Embedding、RMS Norm、Softcap Logits
- `muon-adamw-optimizer`: 原生 Swift 實作的 Muon + AdamW 混合 optimizer，含 Polar Express 正交化和 NorMuon variance reduction
- `config-driven-experiment`: JSON config 驅動的實驗系統，agent 修改 `experiment.json` 即可改變模型架構、超參數、optimizer 設定，零編譯 overhead
- `hardware-auto-detect`: 啟動時自動偵測 Apple Silicon 硬體規格並產生最佳預設參數，含可選 micro-benchmark
- `swift-data-pipeline`: 純 Swift 實作的 BPE tokenizer、Parquet reader、DataLoader 和 evaluation 工具
- `training-loop`: 固定時間預算的訓練迴圈，含 LR schedule、gradient accumulation、loss monitoring 和 val_bpb 評估
- `benchmark-comparison`: 以原版 autoresearch 為基準的效能比較系統，量測 tok/sec、experiments/hour、startup overhead，產出可比較的 benchmark 報告

### Modified Capabilities

（無 — 全新專案）

## Impact

- **新增檔案**: 完整 Swift Package 結構（`Package.swift`、`Sources/` 下所有模組）
- **Dependencies**: mlx-swift（Apple 官方）、Swift Parquet reader、可能需要 Swift-Rust FFI（for rustbpe tokenizer）
- **相容性**: macOS 14+、Apple Silicon（M1/M2/M3/M4 全系列）
- **Agent 整合**: `program.md` 和 `experiment.json` 作為 agent 介面，相容現有 Claude Code / Codex 工作流程
- **資料相容**: 使用與原版相同的 `~/.cache/autoresearch/` 資料格式，可共享已下載的 data shards 和 tokenizer
