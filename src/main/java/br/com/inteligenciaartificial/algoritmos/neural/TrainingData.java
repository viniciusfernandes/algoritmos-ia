package br.com.inteligenciaartificial.algoritmos.neural;

import java.util.List;

public class TrainingData<I, O> {
	private final O expectedValue;

	private final List<I> inputValues;

	public TrainingData(final List<I> inputValues, final O expectedValue) {
		this.inputValues = inputValues;
		this.expectedValue = expectedValue;
	}

}
