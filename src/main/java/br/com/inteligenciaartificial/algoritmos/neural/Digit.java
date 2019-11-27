package br.com.inteligenciaartificial.algoritmos.neural;

public class Digit {
    public static final int DIGITS_SIZE_SET = 10;
    public static final int PIXELS_PER_DIGIT = 28 * 28;
    private final int[] pixels;

    public Digit(final int[] pixels) {
        if (pixels == null || pixels.length == 0) {
            throw new IllegalArgumentException("Pixels array of each one digit must not be null or empty");
        }

        if (pixels.length != PIXELS_PER_DIGIT) {
            throw new IllegalArgumentException(
                            String.format("Pixels array of each one digit must hasve the length equals to %d and it was sent %d.",
                                            PIXELS_PER_DIGIT, pixels.length));
        }
        this.pixels = pixels;
    }

    public int getPixel(final int index) {
        return pixels[index];
    }

    public int[] getPixels() {
        return pixels;
    }
}
