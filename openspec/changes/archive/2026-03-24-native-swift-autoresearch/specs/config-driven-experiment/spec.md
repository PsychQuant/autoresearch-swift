## ADDED Requirements

### Requirement: Experiment config loading from JSON

The system SHALL load experiment configuration from `experiment.json` in the working directory. The config file SHALL support a `changes` object containing only the fields the agent wants to override. All unspecified fields SHALL use hardware-profile defaults.

#### Scenario: Partial override

- **WHEN** experiment.json contains `{"changes": {"depth": 4, "activation": "silu"}}`
- **THEN** depth is set to 4, activation is set to silu, and all other parameters use hardware-profile defaults

#### Scenario: Empty or missing config

- **WHEN** experiment.json is missing or contains `{}`
- **THEN** all parameters use hardware-profile defaults and training proceeds normally

#### Scenario: Invalid config value

- **WHEN** experiment.json contains an invalid value (e.g., `"activation": "invalid_name"`)
- **THEN** the system SHALL exit with a clear error message listing valid options, before any training begins

### Requirement: Config schema validation

The system SHALL validate all config values before training starts. Validation SHALL check: type correctness, value ranges (e.g., depth 2-16, batch_size_log2 12-20), and enum membership (activation, optimizer, window_pattern characters).

#### Scenario: Range validation

- **WHEN** experiment.json specifies depth=0
- **THEN** the system reports "depth must be between 2 and 16" and exits with code 1

### Requirement: Component registry

The system SHALL maintain a registry mapping config string values to Swift Protocol implementations. The registry SHALL support: activation functions, optimizer types, and attention variants.

#### Scenario: Registry lookup

- **WHEN** config specifies activation="silu"
- **THEN** the ComponentRegistry returns the SiLU activation implementation conforming to the ActivationFunction protocol

### Requirement: Three-layer config resolution

The system SHALL resolve configuration through three layers in order: (1) hardware auto-detect baseline, (2) chip-specific profile defaults, (3) agent overrides from experiment.json. Each layer only overrides fields it specifies.

#### Scenario: Layer precedence

- **WHEN** hardware detects M4 Max (profile sets batch_size_log2=18) and experiment.json sets batch_size_log2=16
- **THEN** batch_size_log2 is 16 (agent override wins)

#### Scenario: No agent override

- **WHEN** hardware detects M1 (profile sets depth=4) and experiment.json has no depth field
- **THEN** depth is 4 (profile default used)

### Requirement: Resolved config output

The system SHALL print the fully resolved configuration (after all three layers) at startup, so the agent can see what parameters are being used.

#### Scenario: Config printout

- **WHEN** training starts
- **THEN** the system prints all resolved parameters including their source (hardware/profile/override)
