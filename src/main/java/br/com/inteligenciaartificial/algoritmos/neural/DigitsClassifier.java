package br.com.inteligenciaartificial.algoritmos.neural;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import br.com.inteligenciaartificial.algoritmos.math.Column;
import br.com.inteligenciaartificial.algoritmos.math.Matrix;
import br.com.inteligenciaartificial.algoritmos.math.MultiMatrix;
import br.com.inteligenciaartificial.algoritmos.math.Row;

public class DigitsClassifier {
	private static final int HIDDEN_LAYER_INDEX;
	private static final int HIDDEN_LAYER_SIZE;
	private static final int INPUT_LAYER_INDEX;
	private static final int INPUT_LAYER_SIZE;
	private static final int NUM_LAYERS;
	private static final int OUTPUT_LAYER_SIZE;
	private static final int OUTUPUT_LAYER_INDEX;
	static {
		INPUT_LAYER_INDEX = 0;
		INPUT_LAYER_SIZE = Digit.PIXELS_PER_DIGIT;
		NUM_LAYERS = 3;
		HIDDEN_LAYER_SIZE = 3;
		OUTPUT_LAYER_SIZE = 4;
		HIDDEN_LAYER_INDEX = 1;
		OUTUPUT_LAYER_INDEX = NUM_LAYERS - 1;
	}

	public static void main(final String[] args) {

		final MultiMatrix m = new MultiMatrix(new Row(new double[] { 1, 2, 3 }), new Column(new double[] { 8, 7, 6 }),
				new Matrix(new double[] { 4, 5, 6 }, new double[] { 44, 55, 66 }));

		m.print();
		DigitsClassifier.teste();
	}

	private static void teste() {
		final TrainingDigit digit = new TrainingDigit(new int[INPUT_LAYER_SIZE], new double[] { 0, 0, 1, 0, 0 });
		final List<TrainingDigit> dataTraining = new ArrayList<>();
		dataTraining.add(digit);

		final DigitsClassifier classifier = new DigitsClassifier(0.001, 2);
		classifier.learn(dataTraining);

	}

	private MultiMatrix A;
	private final MultiMatrix B = new MultiMatrix(null, new Column(HIDDEN_LAYER_SIZE), new Column(OUTPUT_LAYER_SIZE));

	private TrainingDigit[][] batchs;
	private int batchSize;

	private final MultiMatrix Error = new MultiMatrix(OUTUPUT_LAYER_INDEX);
	private final Column outputs = new Column(OUTPUT_LAYER_SIZE);

	private Matrix previousError = null;
	// Neural network leanin rate
	private final double rate;

	private Matrix sigmoidDerivative = null;

	private final MultiMatrix W = new MultiMatrix(null, new Matrix(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE),
			new Matrix(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE));

	private Matrix wTranspose = null;

	// Setup the transpose matrix of Z
	private final MultiMatrix Z = new MultiMatrix(new Row(INPUT_LAYER_SIZE), new Row(HIDDEN_LAYER_SIZE), null);

	public DigitsClassifier(final double learningRate, final int batchSize) {
		rate = learningRate;
		this.batchSize = batchSize;

		// Initializing biases and weights with randomly to apply the stoschastic
		// gradient
		// descent algorithm later.
		initBiases();
		initWeights();
	}

	private void backPropagation(final int layer, final Matrix error) {
		if (layer <= INPUT_LAYER_INDEX) {
			return;
		}
		sigmoidDerivative = Z.operate(layer, this::sigmoidDifferential);
		wTranspose = W.get(layer).transpose();

		previousError = wTranspose.multiply(error).dot(sigmoidDerivative);
		backPropagation(layer - 1, previousError);
	}

	private void calcOutputErrors(final Column expectedVal) {
		sigmoidDerivative = Z.copy(HIDDEN_LAYER_INDEX);
		sigmoidDerivative.operate(this::sigmoidDifferential);

		final Matrix gradC = A.get(OUTUPUT_LAYER_INDEX).sub(expectedVal).module();
		final Matrix error = gradC.dot(sigmoidDerivative);

		backPropagation(OUTUPUT_LAYER_INDEX - 1, error);
	}

	public int classify(final Digit digit) {
		int d = -1;
		final double[] output = outputs();
		for (int i = 0; i < output.length - 1; i++) {
			d = output[i] <= output[i + 1] ? i + 1 : i;
		}
		return d;
	}

	private void feedForward() {
		A = new MultiMatrix(NUM_LAYERS);
		feedForward(HIDDEN_LAYER_INDEX, Z.get(0), Z.get(0));
	}

	private void feedForward(final int layer, Matrix Z1, Matrix A1) {
		if (layer >= OUTUPUT_LAYER_INDEX) {
			return;
		}

		Z1 = W.get(layer).transpose().multiply(Z1).sum(B.get(layer));
		A1 = A1.copy().operate(this::sigmoid);

		Z.set(layer, Z1);
		A.set(layer + 1, A1);

		feedForward(layer + 1, Z1, A1);
	}

	private void initBatchs(final List<TrainingDigit> data, int amountBatchs) {

		final int rest = data.size() % amountBatchs;
		batchSize = (data.size() - rest) / amountBatchs;
		if (batchSize == 0) {
			amountBatchs = 1;
			batchSize = data.size();
		}

		batchs = new TrainingDigit[amountBatchs][batchSize];
		int begin = 0;
		int end = 0;

		for (int b = 0; b < amountBatchs; b++) {
			begin = b * batchSize;
			end = begin + batchSize;

			for (int i = begin; i < end; i++) {
				batchs[b][i] = data.get(i);
			}
		}

	}

	private void initBiases() {
		B.initRandom();
	}

	private void initInputs(final Digit digit) {
		final double[] pixels = new double[INPUT_LAYER_SIZE];
		for (int i = 0; i < digit.pixels.length; i++) {
			pixels[i] = digit.pixels[i];
		}
		Z.set(0, new Column(pixels));
	}

	private void initWeights() {
		W.initRandom();
	}

	public void learn(final List<TrainingDigit> data) {
		shuffleData(data);
		initBatchs(data, batchSize);
		updateWeightsAndBiases();
	}

	private double[] outputs() {
		return new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	}

	private void shuffleData(final List<TrainingDigit> data) {
		Collections.shuffle(data);
	}

	private double sigmoid(final double z) {
		return 1d / (1d + Math.pow(Math.E, -z));
	}

	private double sigmoidDifferential(final double z) {
		final double sig = sigmoid(z);
		// Essa eh a expressao algebrica da derivada da funcao sigmoid.
		return sig * (1 - sig);
	}

	private void updateWeightsAndBiases() {
		TrainingDigit data = null;
		for (int i = 0; i < batchs.length; i++) {
			for (int j = 0; j < batchs[i].length; j++) {
				data = batchs[i][j];

				initInputs(data);
				feedForward();
				calcOutputErrors(new Column(data.expectedOutput));
			}

		}
	}
}
