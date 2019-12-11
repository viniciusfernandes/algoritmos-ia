package br.com.inteligenciaartificial.algoritmos.neural;

import java.util.Collections;
import java.util.List;
import java.util.function.UnaryOperator;

import br.com.inteligenciaartificial.algoritmos.math.Column;
import br.com.inteligenciaartificial.algoritmos.math.Matrix;

public abstract class NeuralNetwork2 {

	private static final int BATCH_SIZE = 100;
	// Neural network leanin rate
	private static final double ERROR_RATE = 0.015;

	private TrainingData[][] batchs;

	private Matrix biasError = null;
	private List<Matrix> errors;
	protected final LinkedLayer2 firstLayer;

	protected final LinkedLayer2 lastLayer;

	private Matrix output = null;

	protected final UnaryOperator<Matrix> outputFunction;

	private Matrix weightError = null;

	public NeuralNetwork2(final UnaryOperator<Matrix> activationFunction, final UnaryOperator<Matrix> outputFunction,
			final int... numberOfNeurons) {

		if (numberOfNeurons.length <= 2) {
			throw new IllegalArgumentException("All networks must have 2 layers at least.");
		}

		firstLayer = new LinkedLayer2(numberOfNeurons[0]);
		LinkedLayer2 layer = firstLayer;

		for (int i = 1; i < numberOfNeurons.length; i++) {
			layer = layer.next(new LinkedLayer2(numberOfNeurons[i], activationFunction));
			layer.initRandomBiases();
			layer.initRandomWeights();
		}

		lastLayer = layer;
		this.outputFunction = outputFunction;
	}

	public Matrix apply(final double[] data) {
		initInputs(data);
		feedForward();
		return outputFunction.apply(lastLayer.activate());
	}

	private void feedForward() {
		firstLayer.feedForward();
	}

	private void initBatchs(final List<? extends TrainingData> data) {
		final int batchNum = data.size() / BATCH_SIZE;
		batchs = new TrainingData[batchNum][BATCH_SIZE];

		int j = 0;
		for (int b = 0; b < batchNum; b++) {
			for (int i = 0; i < BATCH_SIZE; i++) {
				batchs[b][i] = data.get(j);
				if (++j >= data.size()) {
					return;
				}
			}
		}

	}

	private void initInputs(final double[] data) {
		firstLayer.addInput(new Column(data));
	}

	private void learn() {
		TrainingData data = null;
		for (int i = 0; i < batchs.length; i++) {
			for (int j = 0; j < batchs[i].length; j++) {
				data = batchs[i][j];
				initInputs(data.getInputValues());
				feedForward();
				propagateError(new Column(data.getExpectedValue()));
			}
			updateWeights();
		}
	}

	public abstract void propagateError(final Column expectedVal);

	private void shuffleData(final List<? extends TrainingData> data) {
		Collections.shuffle(data);
	}

	public void training(final List<? extends TrainingData> data) {
		shuffleData(data);
		initBatchs(data);
		learn();
	}

	private void updateWeights() {
		updateWeights(lastLayer);
	}

	private void updateWeights(final LinkedLayer2 layer) {
		if (layer.getPrevious() == null) {

			return;
		}
		weightError = null;
		biasError = null;
		output = null;

		errors = layer.getErrors();
		int idx = -1;
		for (final Matrix error : errors) {
			idx = errors.indexOf(error);
			output = layer.getPrevious().activate(idx);
			if (weightError == null || biasError == null) {
				biasError = error;
				weightError = output.multiply(error.transpose());

			} else {
				biasError = biasError.sum(error);
				weightError = weightError.sum(output.multiply(error.transpose()));
			}
		}

		weightError = weightError.apply(e -> e * ERROR_RATE);
		biasError = biasError.apply(e -> e * ERROR_RATE);

		layer.subtractWeightError(weightError);
		layer.subtractBiasError(biasError);

		updateWeights(layer.getPrevious());
	}

}
