package br.com.inteligenciaartificial.algoritmos.neural;

public class Digit {
	public final static int DIGIT_PIXELS = 28 * 28;
	public double[] pixels;

	public Digit(final double[] pixels) {
		if (pixels == null || pixels.length != DIGIT_PIXELS) {
			throw new IllegalArgumentException("Pixels of anyone digit must be " + DIGIT_PIXELS);
		}
		this.pixels = pixels;
	}

}
