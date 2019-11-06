package br.com.inteligenciaartificial.algoritmos.neural;

public class TrainingDigit extends Digit {
	public final static int OUTPUT = 10;
	public final double[] expectedOutput;

	public TrainingDigit(final double[] pixels, final double[] expectedOutput) {
		super(pixels);
		if (expectedOutput == null || expectedOutput.length != OUTPUT) {
			throw new IllegalArgumentException("Output of anyone digit must be " + OUTPUT);
		}
		this.expectedOutput = expectedOutput;
	}

}
