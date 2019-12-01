package br.com.inteligenciaartificial.algoritmos.neural;

public class Digit {
    public static final int DIGITS_SIZE_SET = 10;
    public static final int PIXELS_PER_DIGIT = 28 * 28;
    private final int[] inputValues;

    public Digit(final int[] inputValues) {
        if (inputValues == null || inputValues.length == 0) {
            throw new IllegalArgumentException("Pixels array of each one digit must not be null or empty");
        }

        if (inputValues.length != PIXELS_PER_DIGIT) {
            throw new IllegalArgumentException(
                            String.format("Pixels array of each one digit must hasve the length equals to %d and it was sent %d.",
                                            PIXELS_PER_DIGIT, inputValues.length));
        }
        this.inputValues = inputValues;
    }

    public int getInputValue(final int index) {
        return inputValues[index];
    }

    public int[] getInputValues() {
        return inputValues;
    }
}
