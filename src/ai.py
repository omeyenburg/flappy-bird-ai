import numpy


def randarray(n):
    return numpy.random.random(n) * 2 - 1


class ActivationFunction:
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + numpy.exp(-z))

    @staticmethod
    def tanh(z):
        return numpy.tanh(z)

    @staticmethod
    def relu(z):
        return numpy.maximum(0, z)


class Agent:
    def __init__(
        self, inputs: list[str], outputs: list[str], hidden: list[int], activation
    ):
        self.inputs = inputs
        self.outputs = outputs
        self.layers = (len(inputs), *hidden, len(outputs))

        self.values = []
        self.bias = []
        self.weights = []

        for i, layer in enumerate(self.layers):
            self.values.append(numpy.zeros(layer))
            if i + 1 == len(self.layers):
                continue
            self.bias.append(randarray(self.layers[i + 1]))
            self.weights.append(randarray((layer, self.layers[i + 1])))

        self.activation = activation

    def run(self, inputs: list[float]):
        self.values[0][:] = inputs

        for i in range(len(self.layers) - 1):
            self.values[i + 1] = self.activation(
                numpy.dot(self.values[i], self.weights[i]) + self.bias[i]
            )

        return self.values[-1]


class ReinforcementLearningModel:
    def __init__(
        self,
        agents,
        inputs: list[str],
        outputs: list[str],
        hidden: list[int],
        activation,
    ):
        self.agents = agents
