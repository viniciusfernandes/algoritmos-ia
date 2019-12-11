package br.com.inteligenciaartificial.algoritmos.neuralnetwork;

public class TrainingData extends Data {
	private double[] expectedValue;

	public TrainingData(final double[] inputValue) {
		super(inputValue);
	}

	public double[] getExpectedValue() {
		return expectedValue;
	}

	public void setExpectedValue(final double[] expectedValue) {
		this.expectedValue = expectedValue;
	}
}
