import MLX

// MARK: - Activation functions

enum ActivationFunction {
    case reluSquared
    case silu
    case gelu

    static func from(_ name: String) -> ActivationFunction {
        switch name {
        case "relu_squared": return .reluSquared
        case "silu": return .silu
        case "gelu": return .gelu
        default: fatalError("Unknown activation: \(name). Valid: relu_squared, silu, gelu")
        }
    }

    func apply(_ x: MLXArray) -> MLXArray {
        switch self {
        case .reluSquared:
            let r = maximum(x, 0)
            return r * r
        case .silu:
            return x * sigmoid(x)
        case .gelu:
            // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            let c = Float(0.7978845608) // sqrt(2/π)
            return x * (0.5 * (1 + tanh(c * (x + 0.044715 * x * x * x))))
        }
    }
}

// MARK: - Optimizer type

enum OptimizerType {
    case adamw
    case muon
    case muonAdamW

    static func from(_ name: String) -> OptimizerType {
        switch name {
        case "adamw": return .adamw
        case "muon": return .muon
        case "muon_adamw": return .muonAdamW
        default: fatalError("Unknown optimizer: \(name). Valid: adamw, muon, muon_adamw")
        }
    }
}
