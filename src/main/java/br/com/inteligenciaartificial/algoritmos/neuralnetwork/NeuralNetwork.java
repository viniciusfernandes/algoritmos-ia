package br.com.inteligenciaartificial.algoritmos.neuralnetwork;

import java.util.Collections;
import java.util.List;
import java.util.function.UnaryOperator;

import br.com.inteligenciaartificial.algoritmos.math.Column;
import br.com.inteligenciaartificial.algoritmos.math.Matrix;

public abstract class NeuralNetwork {

	private TrainingData[][] batchs;
	private int batchSize = 1;

	private Matrix biasError = null;

	// Neural network leanin rate
	private double errorRate = 0.01;

	private List<Matrix> errors;
	protected final LinkedLayer firstLayer;

	protected final LinkedLayer lastLayer;

	private Matrix output = null;

	protected final UnaryOperator<Matrix> outputFunction;

	private Matrix weightError = null;

	public NeuralNetwork(final UnaryOperator<Matrix> activationFunction, final UnaryOperator<Matrix> outputFunction,
			final int... numberOfNeurons) {

		if (numberOfNeurons.length <= 2) {
			throw new IllegalArgumentException("All networks must have 2 layers at least.");
		}

		firstLayer = new LinkedLayer(numberOfNeurons[0]);
		LinkedLayer layer = firstLayer;

		for (int i = 1; i < numberOfNeurons.length; i++) {
			layer = layer.next(new LinkedLayer(numberOfNeurons[i], activationFunction));
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

	public NeuralNetwork batchSize(final int batchSize) {
		this.batchSize = batchSize;
		return this;
	}

	private void clearErrors() {
		LinkedLayer layer = firstLayer;
		do {
			layer.clear();
			layer = layer.getNext();
		} while (layer != null);

	}

	public NeuralNetwork errorRate(final double errorRate) {
		this.errorRate = errorRate;
		return this;
	}

	private void feedForward() {
		LinkedLayer layer = firstLayer;
		LinkedLayer next = layer.getNext();
		do {
			next.addInput(layer.activate());
			layer = next;
			next = next.getNext();
		} while (next != null);
	}

	private void initBatchs(final List<? extends TrainingData> data) {
		final int batchNum = data.size() / batchSize;
		batchs = new TrainingData[batchNum][batchSize];

		int j = 0;
		for (int b = 0; b < batchNum; b++) {
			for (int i = 0; i < batchSize; i++) {
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
			clearErrors();
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

		LinkedLayer layer = lastLayer;
		LinkedLayer previous = lastLayer.getPrevious();
		do {
			weightError = null;
			biasError = null;
			output = null;

			errors = layer.getErrors();
			int idx = 0;
			for (final Matrix error : errors) {
				output = previous.activate(idx);
				if (weightError == null || biasError == null) {
					biasError = error;
					weightError = output.multiply(error.transpose());

				} else {
					biasError = biasError.sum(error);
					weightError = weightError.sum(output.multiply(error.transpose()));
				}
			}

			weightError = weightError.apply(e -> e * errorRate);
			biasError = biasError.apply(e -> e * errorRate);

			layer.subtractWeightError(weightError);
			layer.subtractBiasError(biasError);

			layer = previous;
			previous = previous.getPrevious();

			idx++;
		} while (previous != null);

	}

}
