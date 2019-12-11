package br.com.inteligenciaartificial.algoritmos.neural.digitrecognizer;

import java.io.IOException;
import java.util.Date;
import java.util.List;
import java.util.function.UnaryOperator;

import br.com.inteligenciaartificial.algoritmos.math.Column;
import br.com.inteligenciaartificial.algoritmos.math.Matrix;
import br.com.inteligenciaartificial.algoritmos.neural.ActivationType;
import br.com.inteligenciaartificial.algoritmos.neural.NetworkFactory;
import br.com.inteligenciaartificial.algoritmos.neural.NeuralNetwork;
import br.com.inteligenciaartificial.algoritmos.neural.TrainingDigit;

public class DigitRecognizer {

	private static final String BAR = "--------------------------------------------";

	private static final NeuralNetwork network;
	private static final UnaryOperator<Matrix> outputFunction;
	static {
		outputFunction = output -> {
			final double[] out = new double[output.getRowNum()];

			int max = 0;
			final int last = output.getRowNum() - 1;
			for (int i = 0; i < last; i++) {
				if (output.get(max, 0) < output.get(i + 1, 0)) {
					max = i + 1;
				}
			}

			out[max] = 1;
			return new Column(out);
		};

		network = NetworkFactory.create(ActivationType.SIGMOID_GRADIENT_DESCENDENT, outputFunction, 28 * 28, 15, 10);
		network.batchSize(2).errorRate(0.015);
	}

	public static void main(final String[] args) throws IOException {
		final List<TrainingDigit> trainingData = new MnistDataReader().readData("data/train-images.idx3-ubyte",
				"data/train-labels.idx1-ubyte");

		int[] total = new int[10];
		for (final TrainingDigit data : trainingData) {
			total[data.getDigit()]++;
		}
		System.out.println(BAR);
		System.out.println("Data report:");
		for (int i = 0; i < total.length; i++) {
			System.out.println(String.format("Digit %d has total %.2f %s", i,
					DigitRecognizer.percent(total[i], trainingData.size()), "%"));
		}

		System.out.println(BAR);
		System.out.println("LEARNING IS JUST BEGINNIG");
		System.out.println(BAR);

		Date init = new Date();

		network.training(trainingData);

		Date end = new Date();

		System.out.println(String.format("LEARNING IS COMPLETED in %.2f secs.", DigitRecognizer.seconds(init, end)));
		System.out.println(BAR);

		final List<TrainingDigit> testingData = new MnistDataReader().readData("data/t10k-images.idx3-ubyte",
				"data/t10k-labels.idx1-ubyte");

		int result = -1;
		int matches = 0;
		final int[] unmatchs = new int[10];

		init = new Date();
		total = new int[10];
		for (final TrainingDigit data : testingData) {
			result = DigitRecognizer.recognize(data);
			total[data.getDigit()]++;

			if (result == data.getDigit()) {
				matches++;
			} else {
				unmatchs[data.getDigit()]++;
			}
		}
		end = new Date();

		System.out.println(BAR);
		System.out.println(
				String.format("CLASSIFICATION IS COMPLETED in %.2f secs.", DigitRecognizer.seconds(init, end)));
		System.out.println(BAR);

		System.out.println(String.format("Total matching %d from %d. Result %.2f %s", matches, testingData.size(),
				DigitRecognizer.percent(matches, testingData.size()), "%"));
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

	private static int recognize(final TrainingDigit data) {
		final Matrix out = network.apply(data.getInputValues());
		for (int i = 0; i < out.getRowNum(); i++) {
			if (out.get(i, 0) == 1d) {
				return i;
			}
		}
		throw new IllegalStateException(String.format("Fail in classify the digit %d", data.getDigit()));

	}

	private static double seconds(final Date init, final Date end) {
		return (end.getTime() - init.getTime()) / 1000;
	}

}
