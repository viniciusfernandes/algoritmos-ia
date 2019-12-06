package br.com.inteligenciaartificial.algoritmos.neural;

public class TrainingData {
  private double[] expectedValue;

  public void setExpectedValue(final double[] expectedValue) {
    this.expectedValue = expectedValue;
  }

  private final double[] inputValues;

  public TrainingData(final double[] inputValue) {
    inputValues = inputValue;
  }

  public TrainingData(final double[] inputValue, final double[] expectedValue) {
    inputValues = inputValue;
    this.expectedValue = expectedValue;
  }

  public double[] getExpectedValue() {
    return expectedValue;
  }

  public double[] getInputValues() {
    return inputValues;
  }

  public int size() {
    return inputValues != null ? inputValues.length : 0;
  }

  public double getInput(final int i) {
    return inputValues[i];
  }
}
