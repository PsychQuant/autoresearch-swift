import Foundation
import MLX
import MLXNN

/// Combined optimizer: Muon for 2D block matrices, AdamW for everything else.
/// Matches reference's parameter routing: embeddings, unembedding, value embeds,
/// and scalars use AdamW; transformer block linear weights use Muon.
class MuonAdamWOptimizer: OptimizerProtocol {

    let adamw: AdamWOptimizer
    let muon: MuonOptimizer
    private let muonPaths: Set<String>

    init(model: GPT, config: ExperimentConfig) {
        // Classify params: block 2D matrices → Muon, rest → AdamW
        var muonPathSet: Set<String> = []
        var muonPathShapes: [(String, [Int])] = []

        for (path, param) in model.parameters().flattened() {
            // Block 2D matrices → Muon, EXCEPT veEmbed and veGate (those use AdamW)
            let isBlockMatrix = path.contains("blocks") && param.ndim == 2
            let isValueEmbed = path.contains("veEmbed")  // veGate stays with Muon (block param)
            if isBlockMatrix && !isValueEmbed {
                muonPathSet.insert(path)
                muonPathShapes.append((path, param.shape))
            }
        }

        self.muonPaths = muonPathSet
        self.adamw = AdamWOptimizer(model: model, config: config, excludePaths: muonPathSet)
        self.muon = MuonOptimizer(matrixPaths: muonPathShapes, config: config)

        let muonParamCount = muonPathShapes.reduce(0) { $0 + $1.1.reduce(1, *) }
        print("MuonAdamW: \(muonPathShapes.count) Muon groups, \(muonParamCount) Muon params")
    }

    func update(model: GPT, grads: ModuleParameters) {
        // Flatten once, share across both optimizers
        let flatGrads = Dictionary(grads.flattened(), uniquingKeysWith: { a, _ in a })
        let flatParams = Dictionary(model.parameters().flattened(), uniquingKeysWith: { a, _ in a })

        var allUpdates = adamw.computeUpdates(flatGrads: flatGrads, flatParams: flatParams)
        allUpdates += muon.computeUpdates(flatGrads: flatGrads, flatParams: flatParams)

        if !allUpdates.isEmpty {
            applyParameterUpdates(to: model, updates: allUpdates)
        }
    }

    func setLRMultiplier(_ multiplier: Float) {
        adamw.setLRMultiplier(multiplier)
        muon.setLRMultiplier(multiplier)
    }

    func setMuonMomentum(_ momentum: Float) {
        muon.momentum = momentum
    }

    func setMuonWeightDecay(_ decay: Float) {
        muon.weightDecay = decay
    }

    var stateArrays: [MLXArray] {
        adamw.stateArrays + muon.stateArrays
    }
}

// MARK: - Optimizer Factory

/// Build the appropriate optimizer based on config.
/// Returns a type-erased OptimizerProtocol for use in the training loop.
func buildOptimizer(model: GPT, config: ExperimentConfig) -> OptimizerProtocol {
    switch config.optimizerType {
    case "muon_adamw", "muon":
        return MuonAdamWOptimizer(model: model, config: config)
    case "adamw":
        return AdamWOptimizer(model: model, config: config)
    default:
        print("Unknown optimizer type '\(config.optimizerType)', falling back to AdamW")
        return AdamWOptimizer(model: model, config: config)
    }
}

// MARK: - Schedule Helpers

/// Muon momentum warmup: linear from 0.85 to 0.95 over first 300 steps
func getMuonMomentum(step: Int) -> Float {
    let frac = Swift.min(Float(step) / 300.0, 1.0)
    return (1 - frac) * 0.85 + frac * 0.95
}

/// Weight decay schedule: decays linearly with training progress
func getWeightDecay(baseDecay: Float, progress: Float) -> Float {
    return baseDecay * (1 - progress)
}

/// Learning rate schedule: warmup → constant → cooldown
func getLRMultiplier(progress: Float, warmupRatio: Float, warmdownRatio: Float, finalLRFrac: Float) -> Float {
    if progress < warmupRatio {
        return warmupRatio > 0 ? progress / warmupRatio : 1.0
    } else if progress < 1.0 - warmdownRatio {
        return 1.0
    } else {
        let cooldown = (1.0 - progress) / warmdownRatio
        return cooldown * 1.0 + (1 - cooldown) * finalLRFrac
    }
}
