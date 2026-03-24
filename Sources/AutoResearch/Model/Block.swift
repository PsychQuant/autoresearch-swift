import MLX
import MLXFast
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

    func callAsFunction(_ x: MLXArray, idx: MLXArray,
                         maskMode: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        let ve: MLXArray? = veEmbed?(idx)
        var out = x + attn(rmsNorm(x), ve: ve, maskMode: maskMode)
        out = out + mlp(rmsNorm(out))
        return out
    }
}

/// RMS Normalization in float32 (matches reference F.rms_norm behavior)
func rmsNorm(_ x: MLXArray) -> MLXArray {
    let xf = x.asType(.float32)
    let variance = mean(xf * xf, axis: -1, keepDims: true)
    return x * rsqrt(variance + 1e-5).asType(x.dtype)
}
