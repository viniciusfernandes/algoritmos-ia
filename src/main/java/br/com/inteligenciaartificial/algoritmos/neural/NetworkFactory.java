package br.com.inteligenciaartificial.algoritmos.neural;

public class NetworkFactory {
    private NetworkFactory() {}

    public static NeuralNetwork create(final ActivationType type) {
        if (ActivationType.SIGMOID == type) {
            return new NeuralNetwork(NetworkFactory::sigmoid, NetworkFactory::sigmoidDerivative);
        }
        return new NeuralNetwork(NetworkFactory::tanh, NetworkFactory::tanhDerivative);

    }

    private static double sigmoid(final double z) {
        return 1 / (1 + Math.exp(-z));
    }

    private static double sigmoidDerivative(final double z) {
        final double sig = sigmoid(z);
        return sig * (1 - sig);
    }

    public static double tanh(final double x) {
        return Math.tanh(x);
    }

    public static double tanhDerivative(final double z) {
        final double tan = tanh(z);
        return 1 - tan * tan;
    }
}
