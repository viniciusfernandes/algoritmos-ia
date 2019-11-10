package br.com.inteligenciaartificial.algoritmos.neural;

public class TrainingDigit extends Digit {
	public final static int OUTPUT = 10;
	public final int expectedOutput;

	public TrainingDigit(final int[] pixels, final int expectedOutput) {
		super(pixels);
		this.expectedOutput = expectedOutput;
	}

}
