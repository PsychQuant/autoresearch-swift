# swift-data-pipeline Specification

## Purpose

TBD - created by archiving change 'native-swift-autoresearch'. Update Purpose after archive.

## Requirements

### Requirement: BPE tokenizer compatible with reference

The system SHALL implement a BPE tokenizer that produces identical token sequences as the reference rustbpe tokenizer. The tokenizer SHALL load vocabulary and merge rules from `~/.cache/autoresearch/tokenizer/`. Vocabulary size SHALL match the reference (8192 tokens + 4 special tokens).

#### Scenario: Tokenizer loading

- **WHEN** the tokenizer directory exists at `~/.cache/autoresearch/tokenizer/`
- **THEN** the tokenizer loads successfully and reports vocab_size matching the reference

#### Scenario: Token compatibility

- **WHEN** the same text is tokenized by both the Swift tokenizer and the reference rustbpe tokenizer
- **THEN** the resulting token sequences are identical

---
### Requirement: Parquet data shard reading

The system SHALL read Parquet files from `~/.cache/autoresearch/data/` containing training data shards. The reader SHALL extract the text column from each shard file.

#### Scenario: Read training shard

- **WHEN** shard file `shard_00001.parquet` exists in the data directory
- **THEN** the system reads all text records from the shard

#### Scenario: Missing data directory

- **WHEN** the data directory does not exist
- **THEN** the system prints instructions to run data preparation (referencing the original `prepare.py` or a Swift equivalent) and exits with code 1

---
### Requirement: Streaming DataLoader

The system SHALL provide a DataLoader that yields batches of (input_tokens, target_tokens) as MLXArrays. The DataLoader SHALL: tokenize text on-the-fly or read pre-tokenized data, pack sequences to the configured sequence length, track epoch boundaries, and support configurable batch size.

#### Scenario: Batch generation

- **WHEN** DataLoader is configured with batch_size=32 and seq_len=2048
- **THEN** each call to next() returns input tensor of shape [32, 2048] and target tensor of shape [32, 2048] (shifted by 1)

#### Scenario: Epoch tracking

- **WHEN** the DataLoader exhausts all data shards
- **THEN** it increments the epoch counter and wraps around to the first shard

---
### Requirement: Validation evaluation (val_bpb)

The system SHALL compute validation bits-per-byte (val_bpb) on a fixed validation shard. The evaluation SHALL: use the last shard as validation data, process a configurable number of eval tokens, and report the result in the same units as the reference implementation.

#### Scenario: Evaluation computation

- **WHEN** evaluation is triggered after training
- **THEN** the system computes val_bpb on the validation shard and prints the result with 6 decimal places

#### Scenario: Eval result comparability

- **WHEN** the same model weights are evaluated by both the Swift and Python MLX implementations
- **THEN** val_bpb values differ by less than 0.001 (numerical precision tolerance)
