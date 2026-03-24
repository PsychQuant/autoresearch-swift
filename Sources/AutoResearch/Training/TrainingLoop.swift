import Foundation
import MLX
import MLXNN

struct TrainingSummary {
    let trainingSeconds: Double
    let totalTokens: Int
    let numSteps: Int
    let mfuPercent: Float
    let tokPerSec: Int
}

class TrainingLoop {
    let model: GPT
    let optimizer: OptimizerProtocol
    var trainLoader: DataLoader
    let config: ExperimentConfig

    init(model: GPT, optimizer: OptimizerProtocol, trainLoader: DataLoader, config: ExperimentConfig) {
        self.model = model
        self.optimizer = optimizer
        self.trainLoader = trainLoader
        self.config = config
    }

    func run() -> TrainingSummary {
        let tokensPerFwdBwd = config.deviceBatchSize * config.sequenceLen
        precondition(
            config.totalBatchSize % tokensPerFwdBwd == 0,
            "totalBatchSize (\(config.totalBatchSize)) must be divisible by deviceBatchSize * seqLen (\(tokensPerFwdBwd))"
        )
        let gradAccumSteps = config.totalBatchSize / tokensPerFwdBwd
        print("Gradient accumulation steps: \(gradAccumSteps)")

        let lossGradFn = valueAndGrad(model: model) {
            (model: GPT, x: MLXArray, y: MLXArray) -> MLXArray in
            model(x, targets: y)
        }

        var smoothTrainLoss: Float = 0
        var totalTrainingTime: Double = 0
        var step = 0
        var epoch = 0

        var trainIter = trainLoader.makeIterator()
        guard var currentBatch = trainIter.next() else {
            print("ERROR: No training data available")
            return TrainingSummary(trainingSeconds: 0, totalTokens: 0, numSteps: 0, mfuPercent: 0, tokPerSec: 0)
        }
        epoch = trainLoader.epoch

        while true {
            let t0 = Date()
            var trainLoss = MLXArray(Float(0))
            var accumGrads: ModuleParameters? = nil

            // --- Gradient accumulation micro-steps ---
            for _ in 0..<gradAccumSteps {
                let (loss, grads) = lossGradFn(model, currentBatch.0, currentBatch.1)
                // MLX array materialization
                MLX.eval(loss, grads)
                trainLoss = trainLoss + loss

                if let existing = accumGrads {
                    // Add grads element-wise using NestedDictionary.mapValues
                    accumGrads = existing.mapValues(grads, transform: { a, b in
                        a + (b ?? MLXArray(Float(0)))
                    })
                } else {
                    accumGrads = grads
                }

                // Fetch next batch
                if let batch = trainIter.next() {
                    currentBatch = batch
                } else {
                    trainIter = trainLoader.makeIterator()
                    currentBatch = trainIter.next()!
                }
                epoch = trainLoader.epoch
            }

            // Average loss across micro-steps
            trainLoss = trainLoss / Float(gradAccumSteps)

            // Average accumulated gradients
            if gradAccumSteps > 1 {
                let scale = 1.0 / Float(gradAccumSteps)
                accumGrads = accumGrads!.mapValues(transform: { $0 * scale })
            }

            // --- Schedules ---
            let progress = Swift.min(Float(totalTrainingTime / Double(config.timeBudget)), 1.0)
            let lrm = getLRMultiplier(
                progress: progress,
                warmupRatio: config.warmupRatio,
                warmdownRatio: config.warmdownRatio,
                finalLRFrac: config.finalLRFrac
            )

            optimizer.setLRMultiplier(lrm)
            optimizer.setMuonMomentum(getMuonMomentum(step: step))
            optimizer.setMuonWeightDecay(getWeightDecay(baseDecay: config.weightDecay, progress: progress))

            // --- Optimizer step ---
            optimizer.update(model: model, grads: accumGrads!)
            MLX.eval(model.parameters(), optimizer.stateArrays)

            // --- Loss monitoring ---
            let trainLossF = trainLoss.item(Float.self)
            if trainLossF.isNaN || trainLossF > 100 {
                print("\nFAIL")
                exit(1)
            }

            let dt = Date().timeIntervalSince(t0)
            if step >= config.startupExcludeSteps {
                totalTrainingTime += dt
            }

            // --- Logging ---
            let emaBeta: Float = 0.9
            smoothTrainLoss = emaBeta * smoothTrainLoss + (1 - emaBeta) * trainLossF
            let debiasedLoss = smoothTrainLoss / (1 - pow(emaBeta, Float(step + 1)))
            let pctDone = 100 * progress
            let tokPerSec = dt > 0 ? Int(Double(config.totalBatchSize) / dt) : 0
            let remaining = Swift.max(0.0, Double(config.timeBudget) - totalTrainingTime)

            print(
                String(format: "\rstep %05d (%.1f%%) | loss: %.6f | lrm: %.2f | dt: %.0fms | tok/sec: %d | epoch: %d | remaining: %.0fs    ",
                       step, pctDone, debiasedLoss, lrm, dt * 1000, tokPerSec, epoch, remaining),
                terminator: ""
            )
            fflush(stdout)

            step += 1

            if step >= config.startupExcludeSteps && totalTrainingTime >= Double(config.timeBudget) {
                break
            }
        }

        print()

        let finalTokPerSec = totalTrainingTime > 0
            ? Int(Double(step * config.totalBatchSize) / totalTrainingTime)
            : 0

        return TrainingSummary(
            trainingSeconds: totalTrainingTime,
            totalTokens: step * config.totalBatchSize,
            numSteps: step,
            mfuPercent: 0,
            tokPerSec: finalTokPerSec
        )
    }
}
