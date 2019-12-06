package br.com.inteligenciaartificial.algoritmos.neural.digitrecognizer;

import java.io.IOException;
import java.util.Date;
import java.util.List;

import br.com.inteligenciaartificial.algoritmos.math.Column;
import br.com.inteligenciaartificial.algoritmos.math.Matrix;
import br.com.inteligenciaartificial.algoritmos.neural.ActivationType;
import br.com.inteligenciaartificial.algoritmos.neural.NetworkFactory;
import br.com.inteligenciaartificial.algoritmos.neural.NeuralNetwork;

public class DigitRecognizer {
  private static final String BAR = "--------------------------------------------";
  private static final NeuralNetwork RECOGNIZER =
          NetworkFactory.create(ActivationType.SIGMOID_GRADIENT_DESCENDENT, DigitRecognizer::outputFunction, 78 * 78, 15, 10);

  public static void main(final String[] args) throws IOException {

    final List<TrainingDigit> trainingData = new MnistDataReader().readData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
    int[] total = new int[10];
    for (final TrainingDigit data : trainingData) {
      total[data.getDigit()]++;
    }
    System.out.println(BAR);
    System.out.println("Data report:");
    for (int i = 0; i < total.length; i++) {
      System.out.println(String.format("Digit %d has total %.2f %s", i, DigitRecognizer.percent(total[i], trainingData.size()), "%"));
    }

    System.out.println(BAR);
    System.out.println("LEARNING IS JUST BEGINNIG");
    System.out.println(BAR);

    Date init = new Date();

    RECOGNIZER.training(trainingData);

    Date end = new Date();

    System.out.println(String.format("LEARNING IS COMPLETED in %.2f secs.", seconds(init, end)));
    System.out.println(BAR);

    final List<TrainingDigit> testingData = new MnistDataReader().readData("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");

    int result = -1;
    int matches = 0;
    final int[] unmatchs = new int[10];

    init = new Date();
    total = new int[10];
    for (final TrainingDigit digit : testingData) {
      result = RECOGNIZER.ouclassify(digit);
      total[digit.getValue()]++;

      if (result == digit.getValue()) {
        matches++;
      } else {
        unmatchs[digit.getValue()]++;
      }
    }
    end = new Date();

    System.out.println(BAR);
    System.out.println(String.format("CLASSIFICATION IS COMPLETED in %.2f secs.", seconds(init, end)));
    System.out.println(BAR);

    System.out.println(String.format("Total matching %d from %d. Result %.2f %s", matches, testingData.size(),
            percent(matches, testingData.size()), "%"));
    System.out.println(BAR);

    System.out.println("Unmatching report: ");
    for (int i = 0; i < unmatchs.length; i++) {
      System.out.println(String.format("Digit %d unmatching %d from %d. Result %.2f %s", i, unmatchs[i], total[i],
              DigitRecognizer.percent(unmatchs[i], total[i]), "%"));
    }
    System.out.println(BAR);

  }

  private static double percent(final double a, final double b) {
    return a * 100d / b;
  }

  private static double seconds(final Date init, final Date end) {
    return (end.getTime() - init.getTime()) / 1000;
  }

  private static int recogn(final int digit) {
    final Matrix output=RECOGNIZER.
    for (int i = 0; i < output.getRowNum(); i++) {
      if (output.get(i, 0) != 0) {
        return i;
      }
    }
    throw new IllegalStateException("Fail to recognizing digits.");
  }

  private static Matrix outputFunction(final Matrix input) {
    final double[] out = new double[input.getRowNum()];

    int max = 0;
    final int last = input.getRowNum() - 1;
    for (int i = 0; i < last; i++) {
      if (input.get(max, 0) < input.get(i + 1, 0)) {
        max = i + 1;
      }
    }

    out[max] = 1;
    return new Column(out);
  }
}
