import Foundation
import MLX
import MLXNN

// MARK: - Optimizer Protocol

protocol OptimizerProtocol {
    func update(model: GPT, grads: ModuleParameters)
    func setLRMultiplier(_ multiplier: Float)
    func setMuonMomentum(_ momentum: Float)
    func setMuonWeightDecay(_ decay: Float)
    var stateArrays: [MLXArray] { get }
}

// MARK: - Parameter Utilities

/// Apply parameter updates to a model using identity matching.
/// This bypasses Module.update(parameters:) which has issues with dictionary-of-modules
/// (e.g. [String: Embedding] like valueEmbeds).
///
/// Strategy: Snapshot current param references → compute new values → apply via identity map.
func applyParameterUpdates(to model: GPT, updates: [(String, MLXArray)]) {
    // Build identity map: ObjectIdentifier of current param → new value
    let currentParams = Dictionary(model.parameters().flattened(), uniquingKeysWith: { a, _ in a })
    var identityMap: [ObjectIdentifier: MLXArray] = [:]

    for (path, newValue) in updates {
        if let currentArray = currentParams[path] {
            identityMap[ObjectIdentifier(currentArray)] = newValue
        }
    }

    // Apply updates by matching array identity
    model.apply { array in
        identityMap[ObjectIdentifier(array)] ?? array
    }
}

// MARK: - AdamW Optimizer

/// Per-parameter AdamW with float32 state, bias correction, and LR scaling by model dimension.
/// Matches reference: per-path config for embedding, unembedding, matrix, scalar parameters.
class AdamWOptimizer: OptimizerProtocol {

    struct ParamConfig {
        var lr: Float
        let betas: (Float, Float)
        let eps: Float
        let weightDecay: Float
    }

    private struct AdamState {
        var m: MLXArray   // first moment (float32)
        var v: MLXArray   // second moment (float32)
        var t: Int = 0
    }

    private var paramConfigs: [String: ParamConfig] = [:]
    private var states: [String: AdamState] = [:]
    private var initialLRs: [String: Float] = [:]
    private let excludePaths: Set<String>

    /// - Parameter excludePaths: paths routed to another optimizer (e.g. Muon in MuonAdamW mode)
    init(model: GPT, config: ExperimentConfig, excludePaths: Set<String> = []) {
        self.excludePaths = excludePaths

        let modelDim = config.modelDim
        let dmodelScale = pow(Float(modelDim) / 768.0, -0.5)
        print(String(format: "Scaling AdamW LRs by 1/sqrt(%d/768) = %.6f", modelDim, dmodelScale))

        for (path, param) in model.parameters().flattened() {
            if excludePaths.contains(path) { continue }

            let cfg: ParamConfig
            if path.contains("veEmbed") {
                // Value embeddings: embedding LR, zero weight decay (matches reference)
                cfg = ParamConfig(lr: config.embeddingLR * dmodelScale, betas: config.adamBetas, eps: 1e-10, weightDecay: 0)
            } else if path.contains("blocks") && param.ndim == 2 {
                // Block matrix params (only reached in AdamW-only mode)
                cfg = ParamConfig(lr: config.matrixLR, betas: config.adamBetas, eps: 1e-10, weightDecay: config.weightDecay)
            } else if path.contains("wte") {
                cfg = ParamConfig(lr: config.embeddingLR * dmodelScale, betas: config.adamBetas, eps: 1e-10, weightDecay: 0)
            } else if path.contains("lmHead") {
                cfg = ParamConfig(lr: config.unembeddingLR * dmodelScale, betas: config.adamBetas, eps: 1e-10, weightDecay: 0)
            } else if path.contains("residLambdas") {
                cfg = ParamConfig(lr: config.scalarLR * 0.01, betas: config.adamBetas, eps: 1e-10, weightDecay: 0)
            } else if path.contains("x0Lambdas") {
                cfg = ParamConfig(lr: config.scalarLR, betas: (0.96, 0.95), eps: 1e-10, weightDecay: 0)
            } else {
                // Fallback: use unembedding LR
                cfg = ParamConfig(lr: config.unembeddingLR * dmodelScale, betas: config.adamBetas, eps: 1e-10, weightDecay: 0)
            }
            paramConfigs[path] = cfg
            initialLRs[path] = cfg.lr
        }
    }

    private func adamStep(path: String, grad: MLXArray, param: MLXArray) -> MLXArray {
        guard let config = paramConfigs[path] else { return param }

        let gradF32 = grad.asType(.float32)
        var paramF32 = param.asType(.float32)
        let (beta1, beta2) = config.betas

        if states[path] == nil {
            states[path] = AdamState(
                m: MLXArray.zeros(gradF32.shape),
                v: MLXArray.zeros(gradF32.shape)
            )
        }

        states[path]!.t += 1
        let t = states[path]!.t

        // Update moments
        states[path]!.m = beta1 * states[path]!.m + (1 - beta1) * gradF32
        states[path]!.v = beta2 * states[path]!.v + (1 - beta2) * (gradF32 * gradF32)

        // Bias correction
        let bias1 = 1 - pow(beta1, Float(t))
        let bias2 = 1 - pow(beta2, Float(t))
        let denom = sqrt(states[path]!.v / bias2) + config.eps
        let stepSize = config.lr / bias1

        // Weight decay (applied before gradient step, matching reference)
        paramF32 = paramF32 * (1 - config.lr * config.weightDecay)
        // Gradient step
        paramF32 = paramF32 - stepSize * (states[path]!.m / denom)

        return paramF32.asType(param.dtype)
    }

    /// Compute parameter updates without applying them (for MuonAdamW batching)
    func computeUpdates(model: GPT, grads: ModuleParameters) -> [(String, MLXArray)] {
        let flatGrads = grads.flattened()
        let flatParams = Dictionary(model.parameters().flattened(), uniquingKeysWith: { a, _ in a })

        var updates: [(String, MLXArray)] = []
        for (path, grad) in flatGrads {
            if excludePaths.contains(path) { continue }
            guard let param = flatParams[path], paramConfigs[path] != nil else { continue }
            let newParam = adamStep(path: path, grad: grad, param: param)
            updates.append((path, newParam))
        }
        return updates
    }

    func update(model: GPT, grads: ModuleParameters) {
        let updates = computeUpdates(model: model, grads: grads)
        if !updates.isEmpty {
            applyParameterUpdates(to: model, updates: updates)
        }
    }

    func setLRMultiplier(_ multiplier: Float) {
        for path in paramConfigs.keys {
            paramConfigs[path]!.lr = initialLRs[path]! * multiplier
        }
    }

    func setMuonMomentum(_ momentum: Float) {
        // No-op for AdamW-only
    }

    func setMuonWeightDecay(_ decay: Float) {
        // No-op for AdamW-only
    }

    var stateArrays: [MLXArray] {
        var arrays: [MLXArray] = []
        for state in states.values {
            arrays.append(state.m)
            arrays.append(state.v)
        }
        return arrays
    }
}
