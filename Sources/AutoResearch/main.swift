import Foundation
import MLX
import MLXRandom
import MLXNN

// MARK: - Startup timing

let tStart = Date()

// MARK: - Load config (three-layer resolution)

let config = ExperimentConfig.resolve()
config.printResolved()

// MARK: - Seed

MLXRandom.seed(42)

// MARK: - Data pipeline

let tokenizer = try BPETokenizer.fromDirectory()
let vocabSize = tokenizer.vocabSize
print("Vocab size: \(vocabSize)")

let trainLoader = DataLoader(
    tokenizer: tokenizer,
    batchSize: config.deviceBatchSize,
    seqLen: config.sequenceLen,
    split: "train"
)

// MARK: - Model

let modelConfig = GPTConfig(
    sequenceLen: config.sequenceLen,
    vocabSize: vocabSize,
    nLayer: config.depth,
    nHead: config.numHeads,
    nKVHead: config.numHeads,
    nEmbd: config.modelDim,
    windowPattern: config.windowPattern
)

let model = GPT(config: modelConfig, activationName: config.activation, logitCap: config.logitCap, mlpRatio: config.mlpRatio)
model.initWeights()
MLX.eval(model.parameters())

let numParams = model.parameterCount()
print("Parameters: \(numParams / 1_000_000)M")

// MARK: - Optimizer

let optimizer: OptimizerProtocol = buildOptimizer(model: model, config: config)

// MARK: - Training loop

let tStartTraining = Date()
let startupSeconds = tStartTraining.timeIntervalSince(tStart)
print(String(format: "Startup: %.1fs", startupSeconds))
print("Time budget: \(config.timeBudget)s")

let runner = TrainingLoop(
    model: model,
    optimizer: optimizer,
    trainLoader: trainLoader,
    config: config
)
let summary = runner.run()

// MARK: - Evaluation

let valBPB = evaluateBPB(
    model: model,
    tokenizer: tokenizer,
    batchSize: config.evalBatchSize
)

// MARK: - Final summary

print("---")
print(String(format: "val_bpb:          %.6f", valBPB))
print(String(format: "training_seconds: %.1f", summary.trainingSeconds))
print(String(format: "total_seconds:    %.1f", Date().timeIntervalSince(tStart)))
print(String(format: "peak_memory_mb:   %.1f", getPeakMemoryMB()))
print(String(format: "mfu_percent:      %.2f", summary.mfuPercent))
print(String(format: "total_tokens_M:   %.1f", Double(summary.totalTokens) / 1e6))
print(String(format: "num_steps:        %d", summary.numSteps))
print(String(format: "num_params_M:     %.1f", Double(numParams) / 1e6))
print(String(format: "depth:            %d", config.depth))
print(String(format: "startup_seconds:  %.1f", startupSeconds))
print(String(format: "tok_per_sec:      %d", summary.tokPerSec))
print(String(format: "experiments_per_hour: %.1f", 3600.0 / Date().timeIntervalSince(tStart)))
