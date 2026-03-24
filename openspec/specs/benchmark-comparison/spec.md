# benchmark-comparison Specification

## Purpose

TBD - created by archiving change 'native-swift-autoresearch'. Update Purpose after archive.

## Requirements

### Requirement: Benchmark output format compatibility

The system SHALL produce output in the exact same format as the reference autoresearch implementation: a `---` delimited block with key-value pairs including val_bpb, training_seconds, total_seconds, peak_memory_mb (peak_vram_mb in reference), mfu_percent, total_tokens_M, num_steps, num_params_M, and depth. The output SHALL be parseable by `grep "^val_bpb:"` and equivalent commands used in the reference `program.md`.

#### Scenario: Output format matching

- **WHEN** a training run completes
- **THEN** the output contains a `---` line followed by key-value lines in the same order and naming as the reference, differing only in peak_vram_mb being renamed to peak_memory_mb for Apple Silicon

#### Scenario: Grep compatibility

- **WHEN** `grep "^val_bpb:" run.log` is executed on Swift version output
- **THEN** it returns the val_bpb line in the same format as the reference

---
### Requirement: Experiment loop compatibility with program.md

The system SHALL support the same experiment loop workflow as the reference `program.md`: git branch creation, experiment run, result parsing, keep/revert via git, and results.tsv logging. The Swift version's `program.md` SHALL follow the same structure so that AI agents (Claude Code, Codex) can drive the loop identically.

#### Scenario: Agent-driven experiment loop

- **WHEN** an AI agent reads the Swift version's program.md
- **THEN** it can execute the same LOOP FOREVER workflow: modify experiment.json → run → parse results → keep/revert → log to results.tsv

#### Scenario: Results.tsv format

- **WHEN** an experiment completes
- **THEN** the result is logged to results.tsv with columns: commit, val_bpb, memory_gb, status, description (tab-separated, matching reference format)

---
### Requirement: Startup overhead measurement

The system SHALL measure and report startup overhead separately from training time. Startup overhead includes: binary launch, config loading, hardware detection, model construction, and initial data loading. The system SHALL report this as `startup_seconds` in the final summary.

#### Scenario: Startup time reporting

- **WHEN** a training run completes
- **THEN** the summary includes `startup_seconds` showing the time from process start to first training step

#### Scenario: Startup under 2 seconds

- **WHEN** the system runs on Apple Silicon with pre-cached data
- **THEN** startup_seconds SHALL be less than 2.0 (excluding optional micro-benchmark time)

---
### Requirement: Throughput comparison metrics

The system SHALL report tok/sec (tokens per second) and experiments_per_hour (estimated based on total run time) to enable direct comparison with the reference implementation.

#### Scenario: Throughput reporting

- **WHEN** a training run completes
- **THEN** the summary includes average tok/sec across all training steps and estimated experiments_per_hour

---
### Requirement: Benchmark results tracking

The system SHALL support a `benchmark_results.tsv` file that records side-by-side comparison data: implementation (swift/python-mlx/python-cuda), tok_per_sec, startup_seconds, val_bpb, total_seconds, and experiments_per_hour.

#### Scenario: Benchmark result logging

- **WHEN** a benchmark comparison run completes
- **THEN** the result is appended to benchmark_results.tsv with implementation identifier and all comparison metrics
