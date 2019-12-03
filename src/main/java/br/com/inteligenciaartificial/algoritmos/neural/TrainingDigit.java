package br.com.inteligenciaartificial.algoritmos.neural;

public class TrainingDigit extends Digit {
	public static final int OUTPUT_SIZE = 10;

	private final double[] expectedOutput;

	public TrainingDigit(final int[] inputValues, final int expectedValue) {
		super(inputValues, expectedValue);
		expectedOutput = new double[OUTPUT_SIZE];
		expectedOutput[expectedValue] = 1;
	}

	public double[] getExpectedOutput() {
		return expectedOutput;
	}

}
