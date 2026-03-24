## Context

autoresearch 是 Karpathy 設計的自主 AI 研究框架：agent 修改訓練程式碼 → 跑 5 分鐘 → 評估 val_bpb → keep/revert → 重複。現有三個 Python 實作：原版（PyTorch/CUDA）、MLX port、MPS port。我們要建立一個零 Python 依賴的原生 Swift 版本。

`references/` 下已有三個 repo 作為參考實作：
- `autoresearch/` — 原版，PyTorch/CUDA/H100，Muon+AdamW
- `autoresearch-mlx/` — MLX port，AdamW only，M4 Max 最佳 1.294 val_bpb
- `autoresearch-macos/` — MPS port，PyTorch/MPS

目標硬體：Apple Silicon M1–M4 全系列，主要開發機為 M4 Max。

## Goals / Non-Goals

**Goals:**

- 完全消除 Python 依賴，單一 `swift build` 產生可執行檔
- 每輪迭代 overhead < 2 秒（vs Python 版 15-20 秒）
- 支援 Muon + AdamW optimizer（超越 MLX Python 版的 AdamW-only）
- 自動適配所有 Apple Silicon 機型
- 相容原版 `~/.cache/autoresearch/` 資料格式
- **訓練速度超越原版**：以 `references/autoresearch/` 為 benchmark，在同等 5 分鐘預算下達到更高的 tok/sec 和更多 experiments/hour

**Non-Goals:**

- 不支援 NVIDIA GPU 或非 Apple 平台
- 不實作 GUI — 純 CLI 工具
- 不在訓練迴圈中使用 CoreML（留給未來部署階段）
- 不支援分散式多機訓練

## Decisions

### Swift Package 結構

採用 Swift Package Manager，將程式碼分為多個模組以利測試和維護：

```
Package.swift
Sources/
├── AutoResearch/          ← 主執行檔
│   └── main.swift
├── Model/                 ← GPT 模型
│   ├── GPT.swift
│   ├── Attention.swift
│   ├── MLP.swift
│   └── Block.swift
├── Optimizer/             ← Muon + AdamW
│   ├── AdamW.swift
│   └── Muon.swift
├── Config/                ← 實驗配置 + 硬體偵測
│   ├── ExperimentConfig.swift
│   ├── HardwareProfile.swift
│   └── ComponentRegistry.swift
├── Data/                  ← 資料管線
│   ├── Tokenizer.swift
│   ├── ParquetReader.swift
│   ├── DataLoader.swift
│   └── Evaluator.swift
└── Training/              ← 訓練迴圈
    └── TrainingLoop.swift
```

**理由**: 模組化設計讓各元件可以獨立測試，也讓未來 agent 的搜索空間清楚劃分。

### MLX-Swift 作為唯一 ML 後端

使用 Apple 官方 `ml-explore/mlx-swift` 作為所有張量運算、自動微分、GPU 加速的唯一後端。

**替代方案考量**:
- CoreML：缺少訓練時的自動微分支援，主要設計給推論
- Metal + MPS：過低階，需自行實作大量基礎設施
- PyTorch/LibTorch C++：違反零 Python 依賴的目標

**理由**: MLX-Swift 是 Apple 官方的 Swift binding，原生支援 lazy evaluation、自動微分、unified memory，且 API 與 Python MLX 高度相似，方便參照 reference 實作。

### Config-Driven 架構搜索

Agent 透過修改 `experiment.json` 來控制實驗，不修改 Swift source code：

```json
{
  "changes": {
    "depth": 4,
    "activation": "silu",
    "optimizer": "adamw",
    "mlp_ratio": 3
  }
}
```

元件透過 Swift Protocol + ComponentRegistry 實現執行時組裝：

| Config 欄位 | 型別 | 可選值 |
|-------------|------|--------|
| `depth` | Int | 2–16 |
| `mlp_ratio` | Int | 2–6 |
| `activation` | String | `relu_squared`, `silu`, `gelu` |
| `optimizer` | String | `adamw`, `muon`, `muon_adamw` |
| `window_pattern` | String | `L`, `SL`, `SSSL` 等組合 |
| `logit_cap` | Float? | `null` 或正數 |
| `batch_size_log2` | Int | 12–20 |
| `total_batch_log2` | Int | 14–20 |

**替代方案考量**:
- Agent 改 Swift code + 重編譯：每次 20-40 秒 overhead，不划算
- DSL/scripting：增加複雜度，config JSON 已足夠

**理由**: 涵蓋 agent 實際會調整的所有旋鈕（從 MLX 版實驗記錄分析得出），零編譯 overhead。

### 硬體自動偵測 + Micro-benchmark

啟動流程：
1. 透過 `sysctl` / `IOKit` / `Metal API` 讀取硬體規格
2. 依晶片型號查詢內建 profile（預設 batch size、depth 等）
3. 可選：跑 100 steps micro-benchmark（~5 秒）量測實際 tok/sec
4. 根據實測或 profile 資料設定預設參數
5. Agent 的 `experiment.json` 覆蓋特定欄位

**理由**: 同一個 binary 在不同機型上自動最佳化，不需人工調參。Micro-benchmark 還能因應當前系統負載動態降級。

### Tokenizer 策略

優先嘗試純 Swift BPE 實作（參照 rustbpe 的演算法）。如果效能不足，退而使用 Swift-Rust FFI 直接呼叫 rustbpe。

**理由**: 純 Swift 實作可消除額外的編譯依賴。BPE tokenizer 的演算法不複雜，Swift 實作的效能足夠用於 tokenize 階段（相對於 5 分鐘訓練，tokenize 時間可忽略）。

### Parquet 資料讀取

使用 Swift 原生 Parquet reader（如 SwiftParquet 或直接讀 Arrow IPC format）讀取 `~/.cache/autoresearch/data/` 下的 shard 檔案。

**退路**: 如果沒有成熟的 Swift Parquet library，可以先用一次性的 Python 腳本將 Parquet 轉為 binary token file，Swift 端直接讀 binary。

### Benchmark 比較機制

以 `references/autoresearch/` 的原版為 benchmark 基準。比較維度：

| 指標 | 原版 (H100) | MLX Python (M4 Max) | Swift 目標 (M4 Max) |
|------|-------------|---------------------|---------------------|
| startup overhead | ~5-10s | ~15-20s | < 2s |
| experiments/hour | ~12 | ~9-10 | ~12+ |
| tok/sec | 因 GPU 而異 | 因 config 而異 | > MLX Python |
| 輸出格式 | program.md 標準 | 相容 | 完全相容 |
| 實驗迴圈 | git keep/revert | git keep/revert | git keep/revert |

**勝出策略**:
1. **消除 Python overhead**：啟動時間從 15-20s 降到 < 2s，每輪省 ~15s
2. **Muon optimizer**：MLX Python 版只有 AdamW，加入 Muon 可在相同步數內達到更低 loss
3. **Config-driven zero-recompile**：不需要 Python import/compile，不需要 MLX JIT compile
4. **精確記憶體管理**：Swift ARC vs Python GC，減少記憶體碎片和 GC stall

**Benchmark 流程**:
1. 先用 Python MLX 版跑一輪基準（相同硬體、相同資料）
2. 用 Swift 版跑相同 config，比較 tok/sec、val_bpb、total overhead
3. 結果寫入 `benchmark_results.tsv`，格式相容原版 `results.tsv`

**輸出格式相容性**：Swift 版的 `program.md` 和輸出格式必須匹配原版的 `---` 分隔 key-value 格式，使 benchmark 結果可以直接用 `grep "^val_bpb:"` 解析。

### Benchmark 視覺化呈現（學習 Surya 風格）

README 的 Benchmark 區塊採用 [datalab-to/surya](https://github.com/datalab-to/surya) 的呈現模式：

```markdown
# Benchmarks

## Training Throughput

![Benchmark chart](static/images/benchmark_throughput.png)   ← 1. 視覺化圖表（一眼看出勝負）

| Implementation | tok/sec (⬆) | startup (s) (⬇) | experiments/hour (⬆) | val_bpb (⬇) |
|...|...|...|...|...|                                          ← 2. 精確數據表格

**Methodology**                                               ← 3. 方法論（公平性）
All benchmarks on M4 Max 128GB, same dataset, same config...

## Running your own benchmarks                                ← 4. 可重現性教學
swift build -c release
./benchmark.sh
```

圖表使用 Python matplotlib 或 Swift Charts 產生 PNG，存放於 `static/images/`。圖表風格：
- Bar chart，每個 bar 代表一個實作（Swift / MLX Python / MPS Python）
- 用顏色區分：Swift 用醒目色（勝出者）、Python 版用灰色調
- 三組 chart：tok/sec、startup overhead、experiments/hour

## Risks / Trade-offs

- **[MLX-Swift 成熟度]** → MLX-Swift 的 API surface 可能不完整，某些 Python MLX 操作可能缺少 Swift binding。**緩解**: 開發初期先確認所需的所有 API 都存在，必要時直接呼叫底層 Metal。

- **[Muon Optimizer 複雜度]** → Polar Express 正交化和 NorMuon variance reduction 的 Swift 實作需要仔細驗證數值正確性。**緩解**: 先實作 AdamW-only 路徑確保端到端可用，再加入 Muon，以 Python 版結果作為 numerical baseline 比對。

- **[Parquet Library]** → Swift 生態的 Parquet 支援不如 Python 成熟。**緩解**: 準備 binary format fallback。

- **[Config 空間有限]** → Config-driven 方式無法表達全新的模型架構（如新的 attention 機制）。**緩解**: ComponentRegistry 設計為可擴展，新增元件只需加一個 Protocol conformance + 註冊，不需改 config schema。

- **[硬體 Profile 覆蓋不全]** → 可能有未列入 profile 的 Apple Silicon 型號。**緩解**: 永遠有 micro-benchmark fallback，不依賴查表。
