## ADDED Requirements

### Requirement: Hardware detection at startup

The system SHALL detect Apple Silicon hardware specifications at startup using system APIs (sysctl, IOKit, Metal). Detected properties SHALL include: chip model name, GPU core count, total unified memory (bytes), memory bandwidth estimate, and Neural Engine core count.

#### Scenario: M4 Max detection

- **WHEN** the system starts on an M4 Max with 128GB RAM
- **THEN** hardware detection reports chip="M4 Max", gpu_cores=40, memory_gb=128, and estimated bandwidth

#### Scenario: Unknown chip fallback

- **WHEN** the system cannot identify the specific chip model
- **THEN** it falls back to reading GPU core count and memory directly from Metal API, and uses conservative defaults

### Requirement: Chip profile database

The system SHALL contain built-in profiles for known Apple Silicon chips mapping chip model to recommended defaults: device_batch_size, total_batch_log2, depth, sequence_length, eval_tokens, and eval_batch_size.

#### Scenario: Known chip lookup

- **WHEN** hardware detects "M4 Max" with 128GB
- **THEN** the profile provides optimized defaults (e.g., device_batch_size=64, depth=8, total_batch_log2=18)

#### Scenario: M1 with 16GB

- **WHEN** hardware detects "M1" with 16GB
- **THEN** the profile provides conservative defaults (e.g., device_batch_size=8, depth=4, seq_len=512, total_batch_log2=14)

### Requirement: Optional micro-benchmark

The system SHALL support an optional micro-benchmark at startup (~5 seconds) that runs a small number of training steps on a tiny model to measure actual tokens/second throughput. The benchmark result SHALL be used to fine-tune batch_size if the measured throughput differs significantly from the profile estimate.

#### Scenario: Benchmark adjusts batch size

- **WHEN** micro-benchmark measures 50% lower throughput than profile expects (e.g., due to other running processes)
- **THEN** the system reduces device_batch_size to prevent OOM and improve iteration time

#### Scenario: Benchmark skipped

- **WHEN** the `--no-benchmark` flag is passed
- **THEN** the system uses profile defaults without running the micro-benchmark

### Requirement: Available memory check

The system SHALL check current available memory at startup and warn if it is significantly less than total memory (indicating other processes are using memory). If available memory is below 50% of total, the system SHALL automatically reduce batch sizes.

#### Scenario: Low memory warning

- **WHEN** total memory is 128GB but available memory is 40GB
- **THEN** the system prints a warning and reduces device_batch_size proportionally
