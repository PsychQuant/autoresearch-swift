import Foundation
import MLX
import MLXRandom
import MLXNN

struct BenchmarkResult {
    let tokPerSec: Int
    let suggestedBatchSize: Int
}

/// Note: MLX.eval() here is the MLX framework's array evaluation function,
/// NOT JavaScript eval(). It forces lazy computation to execute on GPU.
func runMicroBenchmark(config: inout ExperimentConfig) {
    // Skip if --no-benchmark flag
    if CommandLine.arguments.contains("--no-benchmark") {
        print("Micro-benchmark: skipped (--no-benchmark)")
        return
    }

    print("Running micro-benchmark (~5s)...")
    let tBench = Date()

    // Tiny model for benchmarking
    let tinyConfig = GPTConfig(
        sequenceLen: min(256, config.sequenceLen),
        vocabSize: 256,
        nLayer: 2,
        nHead: 2,
        nKVHead: 2,
        nEmbd: 128,
        windowPattern: "L"
    )

    let tinyModel = GPT(config: tinyConfig, activationName: "relu_squared")
    tinyModel.initWeights()
    // Force MLX lazy arrays to materialize on GPU
    MLX.eval(tinyModel.parameters())

    let batchSize = min(16, config.deviceBatchSize)
    let seqLen = tinyConfig.sequenceLen
    let tokensPerStep = batchSize * seqLen

    // Warmup step
    let warmupX = MLXRandom.randInt(low: 0, high: 256, [batchSize, seqLen])
    let warmupY = MLXRandom.randInt(low: 0, high: 256, [batchSize, seqLen])
    let warmupLoss = tinyModel(warmupX, targets: warmupY)
    // Materialize warmup result
    MLX.eval(warmupLoss)

    // Timed steps
    var totalTokens = 0
    let deadline = Date().addingTimeInterval(4.0)
    var steps = 0

    while Date() < deadline {
        let x = MLXRandom.randInt(low: 0, high: 256, [batchSize, seqLen])
        let y = MLXRandom.randInt(low: 0, high: 256, [batchSize, seqLen])
        let loss = tinyModel(x, targets: y)
        // Materialize training step result
        MLX.eval(loss)
        totalTokens += tokensPerStep
        steps += 1
    }

    let elapsed = Date().timeIntervalSince(tBench)
    let measuredTokPerSec = Int(Double(totalTokens) / elapsed)

    print(String(format: "Micro-benchmark: %d tok/sec (%d steps in %.1fs)", measuredTokPerSec, steps, elapsed))
}
