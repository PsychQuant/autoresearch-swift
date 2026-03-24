import Foundation
import MLX
import MLXFast
import MLXNN

class CausalSelfAttention: Module {
    let nHead: Int
    let nKVHead: Int
    let nEmbd: Int
    let headDim: Int
    let veGateChannels: Int = 32

    let cQ: Linear
    let cK: Linear
    let cV: Linear
    let cProj: Linear
    let veGate: Linear?
    let rope: RoPE

    init(config: GPTConfig, layerIdx: Int) {
        self.nHead = config.nHead
        self.nKVHead = config.nKVHead
        self.nEmbd = config.nEmbd
        self.headDim = nEmbd / nHead

        self.cQ = Linear(nEmbd, nHead * headDim, bias: false)
        self.cK = Linear(nEmbd, nKVHead * headDim, bias: false)
        self.cV = Linear(nEmbd, nKVHead * headDim, bias: false)
        self.cProj = Linear(nEmbd, nEmbd, bias: false)

        if GPT.hasVE(layerIdx: layerIdx, nLayer: config.nLayer) {
            self.veGate = Linear(veGateChannels, nKVHead, bias: false)
        } else {
            self.veGate = nil
        }

        // traditional=false: half-split rotary (matches reference's apply_rotary_emb)
        self.rope = RoPE(dimensions: headDim, traditional: false, base: 10000)
    }

    func callAsFunction(_ x: MLXArray, ve: MLXArray?,
                         maskMode: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        let (B, T, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q = cQ(x).reshaped(B, T, nHead, headDim)
        var k = cK(x).reshaped(B, T, nKVHead, headDim)
        var v = cV(x).reshaped(B, T, nKVHead, headDim)

        if let ve = ve, let veGate = veGate {
            let veReshaped = ve.reshaped(B, T, nKVHead, headDim)
            let gate = 2.0 * sigmoid(veGate(x[0..., 0..., 0..<veGateChannels]))
            v = v + MLX.expandedDimensions(gate, axis: -1) * veReshaped
        }

        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        q = rmsNorm(rope(q))
        k = rmsNorm(rope(k))

        let scale: Float = 1.0 / Foundation.sqrt(Float(headDim))
        var y = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: scale, mask: maskMode
        )

        y = y.transposed(0, 2, 1, 3).reshaped(B, T, -1)
        return cProj(y)
    }
}

// MARK: - Attention mask creation

func createCausalMask(seqLen: Int) -> MLXArray {
    let indices = MLXArray(0..<Int32(seqLen))
    let blocked = MLX.expandedDimensions(indices, axis: 0) .> MLX.expandedDimensions(indices, axis: 1)
    return MLX.where(blocked, MLXArray(-Float.infinity), MLXArray(Float(0)))
}

func createSlidingWindowMask(seqLen: Int, windowSize: Int) -> MLXArray {
    let indices = MLXArray(0..<Int32(seqLen))
    let row = MLX.expandedDimensions(indices, axis: 1)  // [T, 1]
    let col = MLX.expandedDimensions(indices, axis: 0)  // [1, T]
    let causal = col .> row
    let tooFar = (row - col) .>= MLXArray(Int32(windowSize))
    let blocked = logicalOr(causal, tooFar)
    return MLX.where(blocked, MLXArray(-Float.infinity), MLXArray(Float(0)))
}

func computeWindowSizes(config: GPTConfig) -> [Int] {
    let pattern = config.windowPattern.uppercased()
    let longWindow = config.sequenceLen
    let shortWindow = longWindow / 2
    let charToWindow: [Character: Int] = ["L": longWindow, "S": shortWindow]

    var windowSizes: [Int] = []
    for i in 0..<config.nLayer {
        let charIdx = i % pattern.count
        let char = pattern[pattern.index(pattern.startIndex, offsetBy: charIdx)]
        windowSizes.append(charToWindow[char] ?? longWindow)
    }
    // Last layer always uses full context
    windowSizes[windowSizes.count - 1] = longWindow
    return windowSizes
}
