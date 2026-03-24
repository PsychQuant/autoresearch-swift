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
        let s = Float(sqrt(3.0)) * pow(Float(nEmbd), -0.5)

        // Build per-parameter init values keyed by ObjectIdentifier
        var initMap: [ObjectIdentifier: MLXArray] = [:]

        for (path, param) in parameters().flattened() {
            let newVal: MLXArray
            if path.contains("wte") && path.hasSuffix("weight") {
                // Token embedding: normal(std=1.0) → bf16 (reference casts embeddings)
                newVal = MLXRandom.normal(param.shape).asType(.bfloat16)
            } else if path.contains("lmHead") && path.hasSuffix("weight") {
                // LM head: normal(std=0.001) → bf16
                newVal = (MLXRandom.normal(param.shape) * 0.001).asType(.bfloat16)
            } else if path.contains("cProj") && path.hasSuffix("weight") {
                // Attention c_proj and MLP c_proj: zeros → bf16
                newVal = MLXArray.zeros(param.shape, dtype: .bfloat16)
            } else if path.contains("veGate") && path.hasSuffix("weight") {
                // VE gate: zeros → bf16
                newVal = MLXArray.zeros(param.shape, dtype: .bfloat16)
            } else if path.contains("veEmbed") && path.hasSuffix("weight") {
                // Value embeddings: uniform → bf16
                newVal = MLXRandom.uniform(low: -s, high: s, param.shape).asType(.bfloat16)
            } else if param.ndim == 2 {
                // Q, K, V, c_fc: uniform → bf16
                newVal = MLXRandom.uniform(low: -s, high: s, param.shape).asType(.bfloat16)
            } else {
                continue
            }
            initMap[ObjectIdentifier(param)] = newVal
        }

        // Apply all at once
        self.apply { array in
            initMap[ObjectIdentifier(array)] ?? array
        }

        // Per-layer scalars
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
        let maskModes = getMaskModes(seqLen: seqLen)

        var x = wte(idx)
        x = rmsNorm(x)
        let x0 = x

        for (i, block) in blocks.enumerated() {
            x = residLambdas[i] * x + x0Lambdas[i] * x0
            x = block(x, idx: idx, maskMode: maskModes[i])
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

    // MARK: - Mask modes

    private func getMaskModes(seqLen: Int) -> [MLXFast.ScaledDotProductAttentionMaskMode] {
        return windowSizes.map { windowSize in
            if windowSize >= seqLen {
                // Full context → use native causal mode (no dense mask needed)
                return .causal
            } else {
                // Sliding window → need additive mask array
                let key = "\(seqLen)_\(windowSize)"
                if let cached = maskCache[key] {
                    return .array(cached)
                }
                let mask = createSlidingWindowMask(seqLen: seqLen, windowSize: windowSize)
                maskCache[key] = mask
                return .array(mask)
            }
        }
    }
}

// MARK: - Loss functions (using MLX built-in cross-entropy)

private func crossEntropyPerToken(logits: MLXArray, targets: MLXArray) -> MLXArray {
    // Replace -1 (padding) with 0 to avoid out-of-bounds, then mask later
    let valid = targets .>= MLXArray(Int32(0))
    let targetsSafe = MLX.where(valid, targets, MLXArray(Int32(0)))
    // MLX built-in cross-entropy: fused log-softmax + nll
    let ce = MLXNN.crossEntropy(logits: logits, targets: targetsSafe, reduction: .none)
    return ce
}

private func crossEntropy(logits: MLXArray, targets: MLXArray) -> MLXArray {
    let perToken = crossEntropyPerToken(logits: logits, targets: targets)
    let valid = targets .!= MLXArray(Int32(-1))
    let masked = perToken * valid.asType(.float32)
    let denom = maximum(sum(valid), MLXArray(1))
    return sum(masked) / denom.asType(.float32)
}
