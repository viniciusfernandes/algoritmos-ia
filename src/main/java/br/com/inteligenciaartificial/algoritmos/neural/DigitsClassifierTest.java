package br.com.inteligenciaartificial.algoritmos.neural;

import java.util.ArrayList;
import java.util.List;

public class DigitsClassifierTest {
	private static List<TrainingDigit> dataTraining() {

		// digito 1 versao 1
		final TrainingDigit d1_1 = new TrainingDigit(new int[] { 0, 0, 1 }, new double[] { 1, 0, 0 });
		// digito um versao 2
		final TrainingDigit d1_2 = new TrainingDigit(new int[] { 1, 0, 0 }, new double[] { 1, 0, 0 });
		// digito um versao 3
		final TrainingDigit d1_3 = new TrainingDigit(new int[] { 0, 1, 0 }, new double[] { 1, 0, 0 });

		final TrainingDigit d2_1 = new TrainingDigit(new int[] { 1, 1, 0 }, new double[] { 0, 1, 0 });
		final TrainingDigit d2_2 = new TrainingDigit(new int[] { 1, 0, 1 }, new double[] { 0, 1, 0 });
		final TrainingDigit d2_3 = new TrainingDigit(new int[] { 0, 1, 1 }, new double[] { 0, 1, 0 });

		final TrainingDigit d3_1 = new TrainingDigit(new int[] { 1, 1, 1 }, new double[] { 0, 0, 1 });

		final List<TrainingDigit> data = new ArrayList<>();
		data.add(d1_1);
		data.add(d1_2);
		data.add(d1_3);

		data.add(d2_1);
		data.add(d2_2);
		data.add(d2_3);

		data.add(d3_1);
		return data;
	}

	public static void main(final String[] args) {
		final DigitsClassifier classifier = new DigitsClassifier(0.0001, 1);
		classifier.learn(DigitsClassifierTest.dataTraining());

		int digit = classifier.classify(new Digit(new int[] { 1, 1, 1 }));
		System.out.println("The classified digit is: " + digit + " and the expected value is 3");

		digit = classifier.classify(new Digit(new int[] { 1, 1, 1 }));
		System.out.println("The classified digit is: " + digit + " and the expected value is 3");

		digit = classifier.classify(new Digit(new int[] { 1, 1, 1 }));
		System.out.println("The classified digit is: " + digit + " and the expected value is 3");

		digit = classifier.classify(new Digit(new int[] { 1, 0, 0 }));
		System.out.println("The classified digit is: " + digit + " and the expected value is 1");

		digit = classifier.classify(new Digit(new int[] { 0, 1, 0 }));
		System.out.println("The classified digit is: " + digit + " and the expected value is 1");

		digit = classifier.classify(new Digit(new int[] { 0, 0, 1 }));
		System.out.println("The classified digit is: " + digit + " and the expected value is 1");

		digit = classifier.classify(new Digit(new int[] { 1, 1, 0 }));
		System.out.println("The classified digit is: " + digit + " and the expected value is 2");

		digit = classifier.classify(new Digit(new int[] { 1, 0, 1 }));
		System.out.println("The classified digit is: " + digit + " and the expected value is 2");

		digit = classifier.classify(new Digit(new int[] { 0, 1, 1 }));
		System.out.println("The classified digit is: " + digit + " and the expected value is 2");
	}
}
