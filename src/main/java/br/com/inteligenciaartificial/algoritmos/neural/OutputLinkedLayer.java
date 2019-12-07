package br.com.inteligenciaartificial.algoritmos.neural;

import java.util.function.UnaryOperator;

import br.com.inteligenciaartificial.algoritmos.math.Matrix;

public class OutputLinkedLayer extends LinkedLayer {
	private Matrix expectedValue;

	public OutputLinkedLayer(final int neuronNumber, final UnaryOperator<Matrix> activation) {
		super(neuronNumber, activation);
	}

	public OutputLinkedLayer expectedValue(final Matrix value) {
		expectedValue = value;
		return this;
	}

}
