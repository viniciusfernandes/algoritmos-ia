package br.com.inteligenciaartificial.algoritmos.neural;

public class TrainingDigit extends Digit {
    public static final int OUTPUT_SIZE = 10;
    private final int expectedDigit;

    private final double[] expectedOutput;

    public TrainingDigit(final int[] pixels, final int expectedDigit) {
        super(pixels);
        expectedOutput = new double[OUTPUT_SIZE];
        expectedOutput[expectedDigit] = 1;
        this.expectedDigit = expectedDigit;
    }

    public int getExpectedDigit() {
        return expectedDigit;
    }

    public double[] getExpectedOutput() {
        return expectedOutput;
    }

}
