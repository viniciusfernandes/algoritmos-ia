package br.com.inteligenciaartificial.algoritmos.neural;

import java.util.Collections;
import java.util.List;

public class DigitsClassifier {
	private final static int HIDDEN_LAYER_SIZE = 15;
	private final static int OUTPUT_LAYER_SIZE = 10;
	private final double[][] A_ln = new double[][] { new double[HIDDEN_LAYER_SIZE], new double[OUTPUT_LAYER_SIZE] };
	private final double[][] B_ln = new double[][] { new double[HIDDEN_LAYER_SIZE], new double[OUTPUT_LAYER_SIZE] };
	private TrainingBatch[] batchs;
	private final int batchSize;
	private final double[][] Error_ln = new double[][] { new double[Digit.PIXELS_PER_DIGIT],
			new double[HIDDEN_LAYER_SIZE] };

	private final double[] neurons = new double[HIDDEN_LAYER_SIZE];
	private final double[] outputs = new double[OUTPUT_LAYER_SIZE];
	// Neural network leanin rate
	private final double rate;

	private final double[][][] W_lnk = new double[][][] { new double[Digit.PIXELS_PER_DIGIT][HIDDEN_LAYER_SIZE],
			new double[HIDDEN_LAYER_SIZE][OUTPUT_LAYER_SIZE] };
	private final double[][] Z_ln = new double[2][];

	public DigitsClassifier(final double learningRate, final int batchSize) {
		rate = learningRate;
		this.batchSize = batchSize;

		// Initializing biases and weights with randomly to apply the stoschastic
		// gradient
		// descent algorithm later.
		initBiases();
		initWeights();
	}

	private void batching(final List<TrainingDigit> data, final int batchSize) {

		final int rest = data.size() % batchSize;
		int batchNum = data.size() - rest;
		batchNum /= batchSize;

		int ibatch = 0;
		int last = 0;
		while (ibatch <= batchNum) {

			if (rest != 0 && ibatch == batchNum) {
				last = rest;
			} else {
				last = batchSize;
			}

			for (int j = ibatch * batchSize; j < last; j++) {
				batchs[ibatch].add(data.get(j));
			}
			ibatch++;
		}

	}

	private void calcErrors(final List<TrainingDigit> digits) {
		final int L = Error_ln.length - 1;
		for (final TrainingDigit d : digits) {
			for (int l = L; l > 0; l--) {
				for (int n = 0; n < Error_ln[l].length; n++) {
					if (l == L) {
						Error_ln[l][n] = Math.abs(d.expectedOutput - Z_ln[L][n]) * sigmoidDifferential(Z_ln[L][n]);
					}
				}
			}
		}
	}

	private int classify() {
		return 1;
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
		for (int l = 0; l < Z_ln.length - 1; l++) {
			for (int n = 0; n < W_lnk[l].length; n++) {
				for (int k = 0; k < W_lnk[l][n].length; k++) {
					Z_ln[l + 1][n] += Z_ln[l][k] * W_lnk[l][n][k];
				}
				Z_ln[l + 1][n] += Z_ln[l + 1][n] + B_ln[l][n];

				if (l == 0) {
					A_ln[l][n] = Z_ln[l + 1][n];
				} else {
					A_ln[l + 1][n] = sigmoid(Z_ln[l + 1][n]);
				}
			}
		}
	}

	private void initBiases() {

		for (int i = 0; i < B_ln.length; i++) {
			for (final int j = 0; j < B_ln[i].length; i++) {
				B_ln[i][j] = Math.random();
			}

		}
	}

	private void initInputs(final TrainingBatch batch) {
		for (final Digit digit : batch.digits) {
			for (int i = 0; i < digit.pixels.length; i++) {
				Z_ln[0][i] = digit.pixels[i];
			}
		}
	}

	private void initWeights() {
		for (int i = 0; i < W_lnk.length; i++) {
			for (int j = 0; j < W_lnk[i].length; j++) {
				for (final int k = 0; k < W_lnk[i][j].length; j++) {
					W_lnk[i][j][k] = Math.random();
				}
			}
		}
	}

	public void learn(final List<TrainingDigit> data) {
		shuffleData(data);
		batching(data, batchSize);
		// updateWeightsAndBiases();
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

	private void updateWeightsAndBiases(final List<TrainingDigit> data) {
		final int m = batchs.length;
		for (int i = 0; i < batchs.length; i++) {
			initInputs(batchs[i]);
			feedForward();
			calcErrors(data);
		}
	}
}
