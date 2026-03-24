import Foundation

struct ExperimentConfig {
    // Model architecture
    var depth: Int = 8
    var aspectRatio: Int = 64
    var headDim: Int = 128
    var windowPattern: String = "SSSL"
    var activation: String = "relu_squared"
    var logitCap: Float? = 15.0
    var mlpRatio: Int = 4

    // Optimization
    var optimizerType: String = "muon_adamw"
    var totalBatchLog2: Int = 19
    var deviceBatchSize: Int = 128
    var embeddingLR: Float = 0.6
    var unembeddingLR: Float = 0.004
    var matrixLR: Float = 0.04
    var scalarLR: Float = 0.5
    var weightDecay: Float = 0.2
    var adamBetas: (Float, Float) = (0.8, 0.95)
    var warmupRatio: Float = 0.0
    var warmdownRatio: Float = 0.5
    var finalLRFrac: Float = 0.0

    // Evaluation
    var evalBatchSize: Int = 256

    // Training
    var timeBudget: Int = 300
    var startupExcludeSteps: Int = 10

    // Derived
    var modelDim: Int {
        let baseDim = depth * aspectRatio
        return ((baseDim + headDim - 1) / headDim) * headDim
    }

    var numHeads: Int { modelDim / headDim }
    var sequenceLen: Int = 2048
    var totalBatchSize: Int { 1 << totalBatchLog2 }

    // Track which fields were overridden by agent
    var overriddenFields: Set<String> = []

    static func resolve() -> ExperimentConfig {
        var config = ExperimentConfig()

        // Layer 1+2: Hardware auto-detect + chip profile defaults
        let hw = HardwareProfile.detect()
        hw.applyDefaults(to: &config)

        // Layer 3: Agent overrides from experiment.json
        config.applyOverrides()

        // Validate before proceeding
        do {
            try config.validate()
        } catch {
            print(error)
            exit(1)
        }

        return config
    }

    mutating func applyOverrides() {
        let path = FileManager.default.currentDirectoryPath + "/experiment.json"
        guard FileManager.default.fileExists(atPath: path),
              let data = FileManager.default.contents(atPath: path),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let changes = json["changes"] as? [String: Any]
        else { return }

        func override<T>(_ key: String, _ target: inout T) {
            if let v = changes[key] as? T {
                target = v
                overriddenFields.insert(key)
            }
        }

        override("depth", &depth)
        override("activation", &activation)
        override("optimizer", &optimizerType)
        override("mlp_ratio", &mlpRatio)
        override("window_pattern", &windowPattern)
        if let v = changes["batch_size_log2"] as? Int {
            deviceBatchSize = 1 << v
            overriddenFields.insert("batch_size_log2")
        }
        override("total_batch_log2", &totalBatchLog2)
        if let v = changes["logit_cap"] as? Float {
            logitCap = v
            overriddenFields.insert("logit_cap")
        }
        if changes["logit_cap"] is NSNull {
            logitCap = nil
            overriddenFields.insert("logit_cap")
        }
        override("matrix_lr", &matrixLR)
        override("embedding_lr", &embeddingLR)
        override("weight_decay", &weightDecay)
    }

    func validate() throws {
        guard (2...16).contains(depth) else {
            throw ConfigError.invalid("depth must be between 2 and 16, got \(depth)")
        }
        guard (12...20).contains(totalBatchLog2) else {
            throw ConfigError.invalid("total_batch_log2 must be between 12 and 20, got \(totalBatchLog2)")
        }
        guard ["relu_squared", "silu", "gelu"].contains(activation) else {
            throw ConfigError.invalid("activation must be relu_squared, silu, or gelu, got \(activation)")
        }
        guard ["adamw", "muon", "muon_adamw"].contains(optimizerType) else {
            throw ConfigError.invalid("optimizer must be adamw, muon, or muon_adamw, got \(optimizerType)")
        }
        guard windowPattern.allSatisfy({ "SL".contains($0.uppercased()) }) else {
            throw ConfigError.invalid("window_pattern must contain only S and L characters, got \(windowPattern)")
        }
    }

    func printResolved() {
        func src(_ key: String) -> String {
            overriddenFields.contains(key) ? "[override]" : "[profile]"
        }
        print("=== Resolved Config ===")
        print("  depth:             \(depth) \(src("depth"))")
        print("  model_dim:         \(modelDim)")
        print("  num_heads:         \(numHeads)")
        print("  activation:        \(activation) \(src("activation"))")
        print("  optimizer:         \(optimizerType) \(src("optimizer"))")
        print("  window_pattern:    \(windowPattern) \(src("window_pattern"))")
        print("  device_batch_size: \(deviceBatchSize) \(src("batch_size_log2"))")
        print("  total_batch_size:  \(totalBatchSize) \(src("total_batch_log2"))")
        print("  logit_cap:         \(logitCap.map { String($0) } ?? "none") \(src("logit_cap"))")
        print("  mlp_ratio:         \(mlpRatio) \(src("mlp_ratio"))")
        print("  matrix_lr:         \(matrixLR) \(src("matrix_lr"))")
        print("  weight_decay:      \(weightDecay) \(src("weight_decay"))")
        print("  time_budget:       \(timeBudget)s")
        print("  seq_len:           \(sequenceLen)")
        print("=======================")
    }
}

enum ConfigError: Error, CustomStringConvertible {
    case invalid(String)

    var description: String {
        switch self {
        case .invalid(let msg): return "Config error: \(msg)"
        }
    }
}
