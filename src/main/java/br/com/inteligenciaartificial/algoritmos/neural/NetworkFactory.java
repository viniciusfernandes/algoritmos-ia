package br.com.inteligenciaartificial.algoritmos.neural;

public class NetworkFactory {
	public static NeuralNetwork create(final ActivationType type) {
		if (ActivationType.SIGMOID == type) {
			return new NeuralNetwork(MathUtils::sigmoid, MathUtils::sigmoidDerivative);
		}
		return new NeuralNetwork(MathUtils::tanh, MathUtils::tanhDerivative);
	}

	private NetworkFactory() {
	}

}
