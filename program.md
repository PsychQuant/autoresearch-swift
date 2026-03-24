# AutoResearch-Swift: Agent-Driven Architecture Search

You are an AI researcher using automated experimentation to find better LLM training recipes on Apple Silicon. You drive the loop below. The training code is in Swift using MLX-Swift — no Python dependency.

## LOOP FOREVER

### 1. Read the current state

```bash
cat results.tsv
```

Look at what has been tried. Identify the best val_bpb so far and which changes led to improvements.

### 2. Form a hypothesis

Based on the results so far, decide what to try next. Ideas:
- Change model depth (2-16)
- Change activation function (relu_squared, silu, gelu)
- Change optimizer (adamw, muon_adamw)
- Adjust learning rates (matrix_lr, embedding_lr, weight_decay)
- Adjust batch size (total_batch_log2: 12-20)
- Change MLP ratio (mlp_ratio: 2-8)
- Change window pattern (e.g., "SSSL", "SSLL", "LLLL")
- Disable logit cap (logit_cap: null)

### 3. Create an experiment branch

```bash
git checkout -b exp-<short-description>
```

### 4. Write the experiment config

Edit `experiment.json` with your changes:

```json
{
  "changes": {
    "depth": 6,
    "activation": "silu"
  }
}
```

Only include fields you want to override. Unspecified fields use hardware-detected defaults.

### 5. Run the experiment

```bash
swift build -c release && .build/release/AutoResearch 2>&1 | tee run.log
```

### 6. Parse the results

```bash
val_bpb=$(grep "^val_bpb:" run.log | awk '{print $2}')
memory_gb=$(echo "scale=2; $(grep "^peak_memory_mb:" run.log | awk '{print $2}') / 1024" | bc)
tok_sec=$(grep "^tok_per_sec:" run.log | awk '{print $2}')
startup=$(grep "^startup_seconds:" run.log | awk '{print $2}')
total=$(grep "^total_seconds:" run.log | awk '{print $2}')
```

### 7. Decide: keep or revert

If val_bpb improved (lower is better):

```bash
git add -A && git commit -m "exp: <description> val_bpb=$val_bpb"
echo -e "$(git rev-parse --short HEAD)\t$val_bpb\t$memory_gb\tkeep\t<description>" >> results.tsv
git checkout main && git merge exp-<short-description>
```

If val_bpb did not improve:

```bash
echo -e "none\t$val_bpb\t$memory_gb\trevert\t<description>" >> results.tsv
git checkout main && git branch -D exp-<short-description>
```

### 8. Log benchmark comparison (optional)

```bash
echo -e "swift\t$tok_sec\t$startup\t$val_bpb\t$total\t$(echo "scale=1; 3600 / $total" | bc)" >> benchmark_results.tsv
```

### 9. Go to step 1

## Notes

- Default time budget is 300 seconds (5 minutes). Override with `"time_budget": N` in experiment.json changes.
- Hardware is auto-detected. The system reads your chip model and memory to set optimal defaults.
- All optimizer state is float32 for numerical stability.
- The Muon optimizer (muon_adamw) includes Polar Express orthogonalization — this is unique to the Swift version among MLX implementations.
