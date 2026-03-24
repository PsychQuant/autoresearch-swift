import Foundation
import MLX
import MLXLinalg
import MLXNN

/// Muon optimizer for 2D matrix parameters.
/// Implements Polar Express orthogonalization (Newton-Schulz iterations),
/// NorMuon variance reduction, Nesterov momentum with warmup, and cautious weight decay.
class MuonOptimizer {

    // Precomputed polynomial coefficients for Newton-Schulz orthogonalization
    static let polarExpressCoeffs: [(Double, Double, Double)] = [
        (8.156554524902461, -22.48329292557795, 15.878769915207462),
        (4.042929935166739, -2.808917465908714, 0.5000178451051316),
        (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
        (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
        (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
    ]

    struct ShapeGroup {
        let paths: [String]
        let shape: [Int]  // [rows, cols] of each parameter
        var momentumBuffer: MLXArray?
        var secondMomentumBuffer: MLXArray?
    }

    var groups: [ShapeGroup] = []
    var lr: Float
    let initialLR: Float
    var momentum: Float = 0.95
    var beta2: Float = 0.95
    var weightDecay: Float
    let nsSteps: Int = 5

    init(matrixPaths: [(String, [Int])], config: ExperimentConfig) {
        self.lr = config.matrixLR
        self.initialLR = config.matrixLR
        self.weightDecay = config.weightDecay

        // Group parameters by shape (Muon processes same-shape params together)
        var shapeDict: [[Int]: [String]] = [:]
        for (path, shape) in matrixPaths {
            shapeDict[shape, default: []].append(path)
        }

        for (shape, paths) in shapeDict.sorted(by: { "\($0.key)" < "\($1.key)" }) {
            groups.append(ShapeGroup(paths: paths, shape: shape))
        }
    }

    /// Compute parameter updates without applying them (for MuonAdamW batching)
    func computeUpdates(flatGrads: [String: MLXArray], flatParams: [String: MLXArray]) -> [(String, MLXArray)] {

        var allUpdates: [(String, MLXArray)] = []

        for i in 0..<groups.count {
            let paths = groups[i].paths
            let shape = groups[i].shape
            let numParams = paths.count

            // Collect grads and params for this shape group
            var gradArrays: [MLXArray] = []
            var paramArrays: [MLXArray] = []
            for path in paths {
                guard let g = flatGrads[path], let p = flatParams[path] else { continue }
                gradArrays.append(g)
                paramArrays.append(p)
            }
            if gradArrays.isEmpty { continue }

            // Stack into [numParams, rows, cols]
            let stackedGrads = stacked(gradArrays)
            let stackedParams = stacked(paramArrays)

            // Initialize state buffers on first call
            if groups[i].momentumBuffer == nil {
                groups[i].momentumBuffer = MLXArray.zeros(
                    [numParams, shape[0], shape[1]],
                    dtype: stackedGrads.dtype
                )
            }
            if groups[i].secondMomentumBuffer == nil {
                let stateShape: [Int] = shape[0] >= shape[1]
                    ? [numParams, shape[0], 1]
                    : [numParams, 1, shape[1]]
                groups[i].secondMomentumBuffer = MLXArray.zeros(
                    stateShape, dtype: stackedGrads.dtype
                )
            }

            let redDim = shape[0] >= shape[1] ? -1 : -2

            // --- Nesterov momentum ---
            let mom = momentum
            groups[i].momentumBuffer = mom * groups[i].momentumBuffer! + (1 - mom) * stackedGrads
            let gNest = (1 - mom) * stackedGrads + mom * groups[i].momentumBuffer!

            // --- Polar Express orthogonalization ---
            var X = gNest.asType(.bfloat16)
            let normVal = sqrt(sum(X * X, axes: [-2, -1], keepDims: true))
            X = X / (normVal * 1.02 + 1e-6)

            if shape[0] > shape[1] {
                // Tall matrices: compute X^T @ X
                for j in 0..<nsSteps {
                    let (a, b, c) = Self.polarExpressCoeffs[j]
                    let aF = MLXArray(Float(a))
                    let bF = MLXArray(Float(b))
                    let cF = MLXArray(Float(c))
                    let A = matmul(X.transposed(0, 2, 1), X)
                    let B = bF * A + cF * matmul(A, A)
                    X = aF * X + matmul(X, B)
                }
            } else {
                // Wide or square matrices: compute X @ X^T
                for j in 0..<nsSteps {
                    let (a, b, c) = Self.polarExpressCoeffs[j]
                    let aF = MLXArray(Float(a))
                    let bF = MLXArray(Float(b))
                    let cF = MLXArray(Float(c))
                    let A = matmul(X, X.transposed(0, 2, 1))
                    let B = bF * A + cF * matmul(A, A)
                    X = aF * X + matmul(B, X)
                }
            }

            let gOrtho = X

            // --- NorMuon variance reduction ---
            let b2 = beta2
            let gF32 = gOrtho.asType(.float32)
            let vMean = mean(gF32 * gF32, axis: redDim, keepDims: true)
            let redDimSize = Float(redDim == -1 ? shape[1] : shape[0])
            let vNormSq = sum(vMean, axes: [-2, -1], keepDims: true) * redDimSize
            let vNorm = sqrt(vNormSq)

            // Update second momentum buffer
            let prevBuf = groups[i].secondMomentumBuffer!.asType(.float32)
            let newBuf = b2 * prevBuf + (1 - b2) * vMean
            groups[i].secondMomentumBuffer = newBuf.asType(stackedGrads.dtype)

            let stepSize = rsqrt(maximum(newBuf, MLXArray(Float(1e-10))))
            let scaledSqSum = vMean * redDimSize * (stepSize * stepSize)
            let vNormNew = sqrt(sum(scaledSqSum, axes: [-2, -1], keepDims: true))
            let finalScale = stepSize * (vNorm / maximum(vNormNew, MLXArray(Float(1e-10))))
            let gScaled = gOrtho * finalScale.asType(gOrtho.dtype)

            // --- LR scaling by aspect ratio ---
            let aspectScale = Float(Swift.max(1.0, sqrt(Double(shape[0]) / Double(shape[1]))))
            let scaledLR = lr * aspectScale

            // --- Cautious weight decay + parameter update ---
            // Decay only where gradient and parameter have same sign
            let mask = (gScaled.asType(.float32) * stackedParams.asType(.float32)) .>= MLXArray(Float(0))
            let decayTerm = scaledLR * weightDecay * stackedParams * mask.asType(stackedParams.dtype)
            let gradTerm = scaledLR * gScaled.asType(stackedParams.dtype)
            let newParams = stackedParams - gradTerm - decayTerm

            // Unstack results
            for (j, path) in paths.enumerated() {
                allUpdates.append((path, newParams[j]))
            }
        }

        return allUpdates
    }

    func update(model: GPT, grads: ModuleParameters) {
        let flatGrads = Dictionary(grads.flattened(), uniquingKeysWith: { a, _ in a })
        let flatParams = Dictionary(model.parameters().flattened(), uniquingKeysWith: { a, _ in a })
        let updates = computeUpdates(flatGrads: flatGrads, flatParams: flatParams)
        if !updates.isEmpty {
            applyParameterUpdates(to: model, updates: updates)
        }
    }

    func setLRMultiplier(_ multiplier: Float) {
        lr = initialLR * multiplier
    }

    var stateArrays: [MLXArray] {
        var arrays: [MLXArray] = []
        for group in groups {
            if let mb = group.momentumBuffer { arrays.append(mb) }
            if let smb = group.secondMomentumBuffer { arrays.append(smb) }
        }
        return arrays
    }
}
