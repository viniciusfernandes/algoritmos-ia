package br.com.inteligenciaartificial.algoritmos.neural;

import java.util.Collections;
import java.util.List;

import br.com.inteligenciaartificial.algoritmos.math.Column;
import br.com.inteligenciaartificial.algoritmos.math.Matrix;

public class NeuralNetworkExample {
	// Neural network leanin rate
	private static final double ERROR_RATE = 0.5;
	private static final int HIDDEN_LAYER_SIZE;
	private static final int INPUT_LAYER_SIZE;
	private static final int OUTPUT_LAYER_SIZE;

	static {
		INPUT_LAYER_SIZE = 3;
		HIDDEN_LAYER_SIZE = 1;
		OUTPUT_LAYER_SIZE = 1;
	}
	private final int batchLength = 1;

	private TrainingDigit[][] batchs;

	private final Layer hiddenLayer = new Layer();
	private final Layer inLayer = new Layer();
	private final Layer outLayer = new Layer();

	public NeuralNetworkExample() {
		initLayers();
		// Initializing biases and weights with randomly to apply the stoschastic
		// gradient
		// descent algorithm later.
		initBiases();
		initWeights();
	}

	private void backPropagate(final Column expectedVal) {

		final Matrix input = hiddenLayer.getInput();
		final Matrix actDerivative = outLayer.activateDerivative(input);

		final Matrix output = calOutput();

		final Matrix gradC = output.sub(expectedVal);
		Matrix error = gradC.dot(actDerivative);

		outLayer.addError(error);

		error = outLayer.weightedError().dot(hiddenLayer.activateDerivative());
		hiddenLayer.addError(error);
	}

	private Matrix calOutput() {
		return new Column(new double[10]);
	}

	public int classify(final Digit digit) {
		initInputs(digit);
		feedForward();

		return output(outLayer.getInput());
	}

	private void clearLayers() {
		inLayer.clear();
		hiddenLayer.clear();
		outLayer.clear();
	}

	private void feedForward() {
		outLayer.activate(hiddenLayer.activate(inLayer.getOutput()));
	}

	private void initBatchs(final List<TrainingDigit> data) {
		int batchNum = batchLength;
		final int rest = data.size() % batchLength;
		int size = (data.size() - rest) / batchLength;
		if (size == 0) {
			batchNum = 1;
			size = data.size();
		}

		batchs = new TrainingDigit[batchNum][size];
		int begin = 0;
		int end = 0;

		for (int b = 0; b < batchNum; b++) {
			begin = b * size;
			end = begin + size;

			for (int i = begin; i < end; i++) {
				batchs[b][i] = data.get(i);
			}
		}

	}

	private void initBiases() {
		hiddenLayer.getBiases().initRandom();
	}

	private void initInputs(final Digit digit) {
		final Matrix out = inLayer.getOutput();
		for (int i = 0; i < Digit.PIXELS_PER_DIGIT; i++) {
			out.set(i, 0, digit.getInputValue(i));
		}
	}

	private void initLayers() {
		inLayer.setOutput(new Column(INPUT_LAYER_SIZE));

		hiddenLayer.setBiases(new Column(HIDDEN_LAYER_SIZE));
		hiddenLayer.setWeight(new Matrix(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE));
	}

	private void initWeights() {
		hiddenLayer.getWeight().initRandom();
	}

	private void learn() {
		TrainingDigit data = null;
		for (int i = 0; i < batchs.length; i++) {
			for (int j = 0; j < batchs[i].length; j++) {
				data = batchs[i][j];

				initInputs(data);
				feedForward();
				backPropagate(new Column(data.getExpectedOutput()));
			}
			updateWeights();
			clearLayers();
		}
	}

	private int output(final Matrix activation) {
		int max = 0;
		final int last = activation.getRowNum() - 1;
		for (int i = 0; i < last; i++) {
			if (activation.get(max, 0) < activation.get(i + 1, 0)) {
				max = i + 1;
			}
		}
		return max;
	}

	private void shuffleData(final List<TrainingDigit> data) {
		Collections.shuffle(data);
	}

	public void training(final List<TrainingDigit> data) {
		shuffleData(data);
		initBatchs(data);
		learn();
	}

	private void updateWeights() {
		Matrix output = null;
		Matrix outWeightError = null;
		Matrix outBiasError = null;

		Matrix hiddenWeightError = null;
		Matrix hiddenBiasError = null;

		int tot = outLayer.getErrors().size();
		Matrix error = null;
		for (int i = 0; i < tot; i++) {
			error = outLayer.getError(i);
			output = hiddenLayer.getOutput(i);
			if (outWeightError == null || outBiasError == null) {
				outBiasError = error;
				outWeightError = output.multiply(error.transpose());

			} else {
				outBiasError = outBiasError.sum(error);
				outWeightError = outWeightError.sum(output.multiply(error.transpose()));
			}

		}

		tot = hiddenLayer.getErrors().size();
		for (int i = 0; i < tot; i++) {
			error = hiddenLayer.getError(i);
			output = inLayer.getOutput(i);
			if (hiddenWeightError == null || hiddenBiasError == null) {
				hiddenBiasError = error;
				hiddenWeightError = output.multiply(error.transpose());

			} else {
				hiddenBiasError = hiddenBiasError.sum(error);
				hiddenWeightError = hiddenWeightError.sum(output.multiply(error.transpose()));
			}
		}

		outWeightError = outWeightError.operate(e -> e * ERROR_RATE / batchLength);
		outBiasError = outBiasError.operate(e -> e * ERROR_RATE / batchLength);

		hiddenWeightError = hiddenWeightError.operate(e -> e * ERROR_RATE / batchLength);
		hiddenBiasError = hiddenBiasError.operate(e -> e * ERROR_RATE / batchLength);

		outLayer.subtractWeightError(outWeightError);
		outLayer.subtractBiasError(outBiasError);

		hiddenLayer.subtractWeightError(hiddenWeightError);
		hiddenLayer.subtractBiasError(hiddenBiasError);
	}
}
