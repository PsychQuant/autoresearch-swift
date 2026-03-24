import MLX
import MLXNN

class MLP: Module {
    let cFC: Linear
    let cProj: Linear
    let activation: ActivationFunction

    init(config: GPTConfig, activationName: String, mlpRatio: Int = 4) {
        self.cFC = Linear(config.nEmbd, config.nEmbd * mlpRatio, bias: false)
        self.cProj = Linear(config.nEmbd * mlpRatio, config.nEmbd, bias: false)
        self.activation = ActivationFunction.from(activationName)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = cFC(x)
        h = activation.apply(h)
        return cProj(h)
    }
}
