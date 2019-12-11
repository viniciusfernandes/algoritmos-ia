package br.com.inteligenciaartificial.algoritmos.neuralnetwork;

import java.util.function.UnaryOperator;

import br.com.inteligenciaartificial.algoritmos.math.Matrix;

public class NetworkFactory {
	public static NeuralNetwork create(final ActivationType type, final UnaryOperator<Matrix> outputFunction,
			final int... numerOfNeurons) {
		if (ActivationType.SIGMOID_GRADIENT_DESCENDENT == type) {
			return new SigmoidGradientDescentNetwork(outputFunction, numerOfNeurons);
		}
		return null;
	}

	private NetworkFactory() {
	}

}
