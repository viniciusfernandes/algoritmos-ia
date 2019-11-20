package br.com.inteligenciaartificial.algoritmos.neural;

public class TrainingDigit extends Digit {
	public final static int OUTPUT = 3;
	public final double[] expectedOutput;

	public TrainingDigit(final int[] pixels, final double[] expectedOutput) {
		super(pixels);
		this.expectedOutput = expectedOutput;
	}

}
