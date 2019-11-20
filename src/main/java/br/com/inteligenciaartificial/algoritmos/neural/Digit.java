package br.com.inteligenciaartificial.algoritmos.neural;

public class Digit {
	public final static int DIGITS_SIZE_SET = 3;
	public final static int PIXELS_PER_DIGIT = 3;
	public int[] pixels;

	public Digit(final int[] pixels) {
		if (pixels == null || pixels.length != PIXELS_PER_DIGIT) {
			throw new IllegalArgumentException("Pixels of anyone digit must be " + PIXELS_PER_DIGIT);
		}
		this.pixels = pixels;
	}

}
