# Benchmark Replication

How to reproduce the benchmark results from the README.

## Prerequisites

Data preparation requires Python (one-time only — not needed for training):

```bash
pip install tiktoken rustbpe pyarrow numpy requests
```

## 1. Prepare Data

```bash
# Download shards + train tokenizer + convert to binary tokens
python3 scripts/prepare_tokens.py --num-shards 10

# Quick test with fewer shards
python3 scripts/prepare_tokens.py --num-shards 3
```

Data is cached at `~/.cache/autoresearch/`. Only needs to run once.

## 2. Run Swift Version

```bash
swift build -c release
.build/release/AutoResearch 2>&1 | tee swift_run.log
```

## 3. Run Python MLX Version (for comparison)

```bash
git clone https://github.com/trevin-creator/autoresearch-mlx references/autoresearch-mlx
cd references/autoresearch-mlx
python3 train.py 2>&1 | tee ../../python_mlx_run.log
cd ../..
```

## 4. Parse Results

```bash
# Swift
grep "^val_bpb:" swift_run.log
grep "^tok_per_sec:" swift_run.log
grep "^startup_seconds:" swift_run.log

# Python MLX
grep "^val_bpb:" python_mlx_run.log
```

## 5. Generate Chart

```bash
pip install matplotlib numpy
python3 scripts/benchmark_chart.py
```

Output: `benchmark_chart.png`

## Methodology

- **Hardware**: Apple M4 Max, 128GB unified memory
- **Config**: `depth=4, total_batch=2^16, seq_len=2048` (set via `experiment.json`)
- **Budget**: Fixed 5-minute training time (300s wall clock, excluding warmup)
- **Metric**: val_bpb (bits per byte) — lower is better, vocab-size independent
- **Throughput**: tok/sec = totalBatchSize / step_time, averaged over steady-state
- **Startup**: Time from binary launch to first training step
