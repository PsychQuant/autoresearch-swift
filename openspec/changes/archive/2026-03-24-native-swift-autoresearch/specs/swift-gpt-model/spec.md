## ADDED Requirements

### Requirement: GPT model definition

The system SHALL provide a GPT model implemented in MLX-Swift with configurable depth, embedding dimension, number of heads, KV heads, MLP ratio, activation function, and window pattern. The model SHALL support RoPE (Rotary Position Embeddings), RMS normalization, Value Embeddings with gated residual connections, and configurable logit softcapping.

#### Scenario: Model construction from config

- **WHEN** a valid ExperimentConfig is provided with depth=4, activation="silu", mlp_ratio=3
- **THEN** the GPT model is constructed with 4 transformer blocks, SiLU activation in MLP layers, and 3x hidden dimension in MLP

#### Scenario: Default model initialization

- **WHEN** a GPT model is constructed with any valid config
- **THEN** all weight matrices SHALL be initialized following the reference scheme: embedding weights with normal(0, 1), projection weights with zeros, QKV weights with uniform(-s, s) where s = sqrt(3) * n_embd^(-0.5)

### Requirement: Causal self-attention with sliding window

The system SHALL implement causal self-attention supporting both full-context and sliding-window patterns. The window pattern SHALL be configurable via a string (e.g., "SSSL" where S=half-context, L=full-context). The last layer SHALL always use full context.

#### Scenario: Sliding window attention mask

- **WHEN** window_pattern is "SSSL" and sequence_len is 2048
- **THEN** layers at pattern positions S use window_size=1024, layers at position L and the final layer use window_size=2048

#### Scenario: Full context only

- **WHEN** window_pattern is "L"
- **THEN** all layers use full causal attention with window_size equal to sequence_len

### Requirement: Activation function registry

The system SHALL support multiple activation functions selectable at runtime via config: `relu_squared` (ReLU(x)^2), `silu` (SiLU/Swish), and `gelu` (GELU).

#### Scenario: Activation selection

- **WHEN** config specifies activation="silu"
- **THEN** all MLP layers use SiLU activation instead of the default relu_squared

### Requirement: Value Embedding with gated residual

The system SHALL implement Value Embeddings on alternating layers (same pattern as reference: layer_idx % 2 == (n_layer - 1) % 2). Each Value Embedding SHALL use a learned gate with sigmoid activation scaled by 2.

#### Scenario: Value embedding layers

- **WHEN** a model has depth=8
- **THEN** Value Embeddings are present on layers matching the alternating pattern, and absent on other layers

### Requirement: Forward pass produces loss or logits

The system SHALL compute cross-entropy loss when targets are provided, or return logits when targets are nil. Logits SHALL be cast to float32 before loss computation. When logit_cap is configured, logits SHALL be clamped via `cap * tanh(logits / cap)`.

#### Scenario: Training forward pass

- **WHEN** forward is called with input tokens and target tokens
- **THEN** the system returns a scalar cross-entropy loss value

#### Scenario: Inference forward pass

- **WHEN** forward is called with input tokens and no targets
- **THEN** the system returns logit tensor of shape [batch, seq_len, vocab_size]
