## ADDED Requirements

### Requirement: AdamW optimizer

The system SHALL implement AdamW optimization with configurable learning rate, beta1, beta2, epsilon, and weight decay. Bias correction SHALL be applied. All optimizer state SHALL be maintained in float32 precision.

#### Scenario: AdamW parameter update

- **WHEN** a gradient step is executed with AdamW
- **THEN** parameters are updated using bias-corrected first and second moment estimates with weight decay applied before the gradient step

### Requirement: Muon optimizer for matrix parameters

The system SHALL implement Muon optimization for 2D matrix parameters using Polar Express orthogonalization (5 Newton-Schulz iterations) and NorMuon variance reduction. Nesterov momentum SHALL be used with warmup from 0.85 to 0.95 over the first 300 steps.

#### Scenario: Muon orthogonalization

- **WHEN** a Muon step is executed on stacked gradient matrices
- **THEN** the Polar Express algorithm applies iterative orthogonalization using the precomputed polynomial coefficients, choosing row or column orientation based on matrix aspect ratio

#### Scenario: Muon momentum warmup

- **WHEN** training step is 150 (halfway through 300-step warmup)
- **THEN** Muon momentum value is 0.90 (linear interpolation between 0.85 and 0.95)

### Requirement: Combined MuonAdamW optimizer

The system SHALL provide a combined optimizer that routes parameter groups to the correct algorithm: Muon for 2D matrix parameters in transformer blocks, AdamW for embeddings, unembedding, value embeddings, and scalar parameters. Parameter groups SHALL be organized by parameter shape for Muon and by role for AdamW.

#### Scenario: Optimizer routing

- **WHEN** a model is configured with optimizer="muon_adamw"
- **THEN** transformer block linear weights use Muon, embedding/lm_head weights use AdamW, and scalar parameters (resid_lambdas, x0_lambdas) use AdamW with distinct learning rates

#### Scenario: AdamW-only mode

- **WHEN** config specifies optimizer="adamw"
- **THEN** all parameters use AdamW optimization, Muon is not applied

### Requirement: Learning rate scaling by model dimension

The system SHALL scale AdamW learning rates proportionally to 1/sqrt(d_model/768), matching the reference implementation's dimensional scaling.

#### Scenario: LR scaling for non-default dimension

- **WHEN** model dimension is 512 (depth=8, aspect_ratio=64)
- **THEN** AdamW learning rates are multiplied by (512/768)^(-0.5) ≈ 1.2247

### Requirement: Cautious weight decay for Muon

The system SHALL apply weight decay only when the gradient and parameter have the same sign (cautious update), preventing decay from fighting the gradient direction.

#### Scenario: Cautious decay masking

- **WHEN** a Muon step is executed with weight_decay > 0
- **THEN** weight decay is applied only to parameter elements where (gradient * parameter) >= 0
