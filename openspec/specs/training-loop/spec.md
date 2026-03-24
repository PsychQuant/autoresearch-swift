# training-loop Specification

## Purpose

TBD - created by archiving change 'native-swift-autoresearch'. Update Purpose after archive.

## Requirements

### Requirement: Fixed time-budget training

The system SHALL train for exactly TIME_BUDGET seconds (default 300s) of wall-clock training time, excluding startup, compilation, and evaluation. Training SHALL stop after the first step that exceeds the time budget, with a warmup exclusion period for the first N steps (configurable, default 10) to avoid counting compilation time.

#### Scenario: 5-minute training budget

- **WHEN** training starts with TIME_BUDGET=300
- **THEN** training stops after total_training_time >= 300 seconds, where total_training_time excludes the first 10 warmup steps

#### Scenario: Warmup exclusion

- **WHEN** step 5 takes 30 seconds (due to MLX compilation)
- **THEN** that 30 seconds is NOT counted toward total_training_time

---
### Requirement: Gradient accumulation

The system SHALL support gradient accumulation when TOTAL_BATCH_SIZE exceeds DEVICE_BATCH_SIZE * MAX_SEQ_LEN. The number of accumulation steps SHALL be computed as TOTAL_BATCH_SIZE / (DEVICE_BATCH_SIZE * MAX_SEQ_LEN). TOTAL_BATCH_SIZE SHALL be evenly divisible by the per-device token count.

#### Scenario: Accumulation computation

- **WHEN** TOTAL_BATCH_SIZE=2^18, DEVICE_BATCH_SIZE=32, MAX_SEQ_LEN=2048
- **THEN** grad_accum_steps = 262144 / (32 * 2048) = 4

---
### Requirement: Learning rate schedule

The system SHALL implement a learning rate schedule based on training progress (training_time / TIME_BUDGET) with three phases: warmup (linear ramp from 0), constant, and cooldown (linear decay to FINAL_LR_FRAC). Phase boundaries are controlled by WARMUP_RATIO and WARMDOWN_RATIO.

#### Scenario: Cooldown phase

- **WHEN** progress is 0.8 and WARMDOWN_RATIO=0.5 and FINAL_LR_FRAC=0.0
- **THEN** lr_multiplier = (1.0 - 0.8) / 0.5 = 0.4

---
### Requirement: Loss monitoring and early abort

The system SHALL monitor training loss and abort if loss exceeds 100 or becomes NaN. Abort SHALL print "FAIL" and exit with code 1.

#### Scenario: Loss explosion

- **WHEN** training loss reaches 150.0
- **THEN** the system prints "FAIL" and exits with code 1

#### Scenario: NaN loss

- **WHEN** training loss becomes NaN
- **THEN** the system prints "FAIL" and exits with code 1

---
### Requirement: Training progress logging

The system SHALL log training progress after each step, including: step number, progress percentage, smoothed training loss (EMA with beta=0.9), lr multiplier, step duration (ms), tokens/second, epoch number, and remaining time.

#### Scenario: Progress line format

- **WHEN** step 100 completes
- **THEN** the system prints a single-line status with all required metrics, using carriage return for in-place updates

---
### Requirement: Final summary output

The system SHALL print a final summary after training and evaluation, including: val_bpb, training_seconds, total_seconds, peak_memory_mb, total_tokens_M, num_steps, num_params_M, and depth. The output format SHALL match the reference implementation's `---` delimited key-value format.

#### Scenario: Summary output

- **WHEN** training and evaluation complete
- **THEN** the system prints a summary block starting with "---" followed by key: value lines matching the reference format

---
### Requirement: GC management

The system SHALL disable Python-style GC stalls by calling appropriate Swift/MLX memory management after the first step, and periodically (every 5000 steps) thereafter.

#### Scenario: GC freeze after warmup

- **WHEN** step 0 completes
- **THEN** the system performs a GC collection and freezes the GC to prevent stalls during training
