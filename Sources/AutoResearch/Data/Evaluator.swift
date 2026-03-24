import Foundation
import MLX
import MLXNN

/// Evaluation constants matching reference implementation
private let maxSeqLen = 2048
private let defaultEvalTokens = 3 * 524288  // MLX version uses 3x, original uses 40x

/// Compute validation bits-per-byte (val_bpb).
///
/// Sums per-token cross-entropy (nats), sums target byte lengths,
/// then converts nats/byte to bits/byte. Special tokens (byte_length=0)
/// are excluded from both sums.
func evaluateBPB(model: GPT, tokenizer: BPETokenizer, batchSize: Int) -> Float {
    print("Starting final evaluation...")

    let tokenBytes = tokenizer.tokenBytes
    let valLoader = DataLoader(
        tokenizer: tokenizer,
        batchSize: batchSize,
        seqLen: maxSeqLen,
        split: "val"
    )

    let steps = defaultEvalTokens / (batchSize * maxSeqLen)
    var totalNats: Float = 0.0
    var totalBytes: Int = 0

    for step in 0..<steps {
        let (x, y, _) = valLoader.nextBatch()

        // Forward pass with per-token loss (reduction="none")
        let lossFlat = model.forward(x, targets: y, reduction: "none").reshaped(-1)
        let yFlat = y.reshaped(-1)

        // Look up byte count for each target token
        let nbytes = tokenBytes[yFlat]
        let mask = nbytes .> 0

        let stepNats = (lossFlat * mask.asType(.float32)).sum().item(Float.self)
        let stepBytes = nbytes.sum().item(Int.self)

        totalNats += stepNats
        totalBytes += stepBytes

        if (step + 1) % 10 == 0 {
            print(String(format: "\r  eval step %d/%d", step + 1, steps), terminator: "")
            fflush(stdout)
        }
    }

    print()

    if totalBytes == 0 { return Float.infinity }

    let log2 = Float(log(2.0))
    return totalNats / (log2 * Float(totalBytes))
}

/// Get peak GPU memory usage in MB
func getPeakMemoryMB() -> Float {
    return Float(Memory.peakMemory) / (1024 * 1024)
}
