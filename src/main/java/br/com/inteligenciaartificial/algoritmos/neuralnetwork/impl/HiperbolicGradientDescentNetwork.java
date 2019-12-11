package br.com.inteligenciaartificial.algoritmos.neuralnetwork.impl;

import java.util.function.UnaryOperator;

import br.com.inteligenciaartificial.algoritmos.math.Column;
import br.com.inteligenciaartificial.algoritmos.math.Matrix;
import br.com.inteligenciaartificial.algoritmos.neuralnetwork.LinkedLayer;
import br.com.inteligenciaartificial.algoritmos.neuralnetwork.NeuralNetwork;
import br.com.inteligenciaartificial.algoritmos.utils.MathUtils;

class HiperbolicGradientDescentNetwork extends NeuralNetwork {

	private final UnaryOperator<Matrix> activationDerivativeFunction;

	public HiperbolicGradientDescentNetwork(final UnaryOperator<Matrix> outputFunction, final int[] layersNumber) {
		super(m -> m.apply(MathUtils::tanh), outputFunction, layersNumber);
		activationDerivativeFunction = m -> m.apply(MathUtils::tanhDerivative);
	}

	@Override
	public void propagateError(final Column expectedVal) {
		Matrix input = lastLayer.activate();
		Matrix derivative = input.apply(MathUtils::tanhDerivative);

		final Matrix outputVal = outputFunction.apply(input);

		final Matrix gradient = outputVal.sub(expectedVal);
		final Matrix outError = gradient.dot(derivative);

		lastLayer.addError(outError);

		Matrix error = null;
		Matrix outWeight = null;
		LinkedLayer layer = lastLayer.getPrevious();
		do {
			input = layer.activate();
			derivative = activationDerivativeFunction.apply(input);
			error = layer.getNext().getError();
			outWeight = layer.getNext().getWeight();
			error = outWeight.multiply(error).dot(derivative);
			layer.addError(error);

			layer = layer.getPrevious();
		} while (layer.getPrevious() != null);

	}

}
