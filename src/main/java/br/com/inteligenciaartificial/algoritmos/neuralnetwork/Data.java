package br.com.inteligenciaartificial.algoritmos.neuralnetwork;

public class Data {
	private final double[] inputValues;

	public Data(final double[] inputValue) {
		inputValues = inputValue;
	}

	public Data(final double[] inputValue, final double[] expectedValue) {
		inputValues = inputValue;
	}

	public double getInput(final int i) {
		return inputValues[i];
	}

	public double[] getInputValues() {
		return inputValues;
	}

	public int size() {
		return inputValues != null ? inputValues.length : 0;
	}
}
