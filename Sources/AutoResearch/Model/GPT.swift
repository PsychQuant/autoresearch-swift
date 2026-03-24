import Foundation
import MLX
import MLXRandom
import MLXNN

struct GPTConfig {
    let sequenceLen: Int
    let vocabSize: Int
    let nLayer: Int
    let nHead: Int
    let nKVHead: Int
    let nEmbd: Int
    let windowPattern: String
}

class GPT: Module {
    let config: GPTConfig
    let activationName: String
    let logitCap: Float?
    let mlpRatio: Int

    let wte: Embedding
    var blocks: [Block]
    let lmHead: Linear
    var residLambdas: MLXArray
    var x0Lambdas: MLXArray

    let windowSizes: [Int]
    private var maskCache: [String: MLXArray] = [:]

    init(config: GPTConfig, activationName: String = "relu_squared", logitCap: Float? = 15.0, mlpRatio: Int = 4) {
        self.config = config
        self.activationName = activationName
        self.logitCap = logitCap
        self.mlpRatio = mlpRatio

        self.wte = Embedding(embeddingCount: config.vocabSize, dimensions: config.nEmbd)
        self.blocks = (0..<config.nLayer).map { i in
            Block(config: config, layerIdx: i, activationName: activationName, mlpRatio: mlpRatio)
        }
        self.lmHead = Linear(config.nEmbd, config.vocabSize, bias: false)
        self.residLambdas = MLXArray.ones([config.nLayer])
        self.x0Lambdas = MLXArray.zeros([config.nLayer])
        self.windowSizes = computeWindowSizes(config: config)
    }

    static func hasVE(layerIdx: Int, nLayer: Int) -> Bool {
        layerIdx % 2 == (nLayer - 1) % 2
    }

    // MARK: - Weight initialization (matching reference)

    func initWeights() {
        let nEmbd = config.nEmbd
        let scale = Float(sqrt(3.0)) * pow(Float(nEmbd), -0.5)

        self.apply { array in
            if array.ndim == 2 {
                return MLXRandom.uniform(low: -scale, high: scale, array.shape).asType(.bfloat16)
            }
            return array
        }

        residLambdas = MLXArray.ones([config.nLayer])
        x0Lambdas = MLXArray.full([config.nLayer], values: MLXArray(Float(0.1)))
    }

    func parameterCount() -> Int {
        var count = 0
        for (_, param) in parameters().flattened() {
            count += param.size
        }
        return count
    }

    // MARK: - Forward pass

    func callAsFunction(_ idx: MLXArray, targets: MLXArray? = nil) -> MLXArray {
        return forward(idx, targets: targets, reduction: "mean")
    }

    func forward(_ idx: MLXArray, targets: MLXArray? = nil, reduction: String = "mean") -> MLXArray {
        let (_, seqLen) = (idx.dim(0), idx.dim(1))
        let masks = getMasks(seqLen: seqLen)

        var x = wte(idx)
        x = rmsNorm(x)
        let x0 = x

        // Transformer blocks with residual stream
        // Each block now owns its own VE (if any)
        for (i, block) in blocks.enumerated() {
            x = residLambdas[i] * x + x0Lambdas[i] * x0
            x = block(x, idx: idx, mask: masks[i])
        }

        x = rmsNorm(x)

        var logits = lmHead(x).asType(.float32)

        if let cap = logitCap {
            logits = cap * tanh(logits / cap)
        }

        guard let targets = targets else {
            return logits
        }

        if reduction == "none" {
            return crossEntropyPerToken(logits: logits, targets: targets)
        }

        return crossEntropy(logits: logits, targets: targets)
    }

    // MARK: - Mask cache

    private func getMasks(seqLen: Int) -> [MLXArray] {
        return windowSizes.map { windowSize in
            let key = "\(seqLen)_\(windowSize)"
            if let cached = maskCache[key] { return cached }
            let mask: MLXArray
            if windowSize >= seqLen {
                mask = createCausalMask(seqLen: seqLen)
            } else {
                mask = createSlidingWindowMask(seqLen: seqLen, windowSize: windowSize)
            }
            maskCache[key] = mask
            return mask
        }
    }
}

// MARK: - Loss functions

private func crossEntropyPerToken(logits: MLXArray, targets: MLXArray) -> MLXArray {
    let B = logits.dim(0)
    let T = logits.dim(1)
    let V = logits.dim(2)

    let logitsFlat = logits.reshaped(B * T, V)
    let targetsFlat = targets.reshaped(B * T)

    let maxLogits = logitsFlat.max(axis: -1, keepDims: true)
    let shifted = logitsFlat - maxLogits
    let logSumExp = log(sum(exp(shifted), axis: -1, keepDims: true))
    let logProbs = shifted - logSumExp

    let indices = MLX.expandedDimensions(targetsFlat, axis: -1)
    let targetLogProbs = takeAlong(logProbs, indices, axis: -1).squeezed(axis: -1)

    return -targetLogProbs.reshaped(B, T)
}

private func crossEntropy(logits: MLXArray, targets: MLXArray) -> MLXArray {
    let perToken = crossEntropyPerToken(logits: logits, targets: targets)
    let valid = targets .!= MLXArray(Int32(-1))
    let masked = perToken * valid.asType(.float32)
    let denom = maximum(sum(valid), MLXArray(1))
    return sum(masked) / denom.asType(.float32)
}
