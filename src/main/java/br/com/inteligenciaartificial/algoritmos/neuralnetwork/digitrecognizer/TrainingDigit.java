package br.com.inteligenciaartificial.algoritmos.neuralnetwork.digitrecognizer;

import br.com.inteligenciaartificial.algoritmos.neuralnetwork.TrainingData;

public class TrainingDigit extends TrainingData {
  final int digit;

  public TrainingDigit(final double[] inputValues, final int digit) {
    super(inputValues);
    this.digit = digit;
    final double[] expectedValue = new double[10];
    expectedValue[digit] = 1;
    setExpectedValue(expectedValue);
  }

  public int getDigit() {
    return digit;
  }

}
