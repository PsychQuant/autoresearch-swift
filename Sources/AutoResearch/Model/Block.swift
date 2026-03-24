import MLX
import MLXNN

class Block: Module {
    let attn: CausalSelfAttention
    let mlp: MLP
    let veEmbed: Embedding?

    init(config: GPTConfig, layerIdx: Int, activationName: String, mlpRatio: Int = 4) {
        self.attn = CausalSelfAttention(config: config, layerIdx: layerIdx)
        self.mlp = MLP(config: config, activationName: activationName, mlpRatio: mlpRatio)

        // Value embedding lives inside the block that uses it
        if GPT.hasVE(layerIdx: layerIdx, nLayer: config.nLayer) {
            let headDim = config.nEmbd / config.nHead
            let kvDim = config.nKVHead * headDim
            self.veEmbed = Embedding(embeddingCount: config.vocabSize, dimensions: kvDim)
        } else {
            self.veEmbed = nil
        }
    }

    func callAsFunction(_ x: MLXArray, idx: MLXArray, mask: MLXArray?) -> MLXArray {
        let ve: MLXArray? = veEmbed?(idx)
        var out = x + attn(rmsNorm(x), ve: ve, mask: mask)
        out = out + mlp(rmsNorm(out))
        return out
    }
}

/// RMS Normalization (without learnable weights — matches reference implementation)
func rmsNorm(_ x: MLXArray) -> MLXArray {
    let variance = mean(x * x, axis: -1, keepDims: true)
    return x * rsqrt(variance + 1e-5)
}
