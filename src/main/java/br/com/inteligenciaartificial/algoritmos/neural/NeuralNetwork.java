package br.com.inteligenciaartificial.algoritmos.neural;

import java.util.Collections;
import java.util.List;
import java.util.function.UnaryOperator;

import br.com.inteligenciaartificial.algoritmos.math.Column;
import br.com.inteligenciaartificial.algoritmos.math.Matrix;

public abstract class NeuralNetwork {

  private TrainingData[][] batchs;
  private static final int BATCH_SIZE = 1;

  // Neural network leanin rate
  private static final double ERROR_RATE = 0.015;

  final OutputLinkedLayer lastLayer;

  final LinkedLayer firstLayer;

  public NeuralNetwork(
          final UnaryOperator<Matrix> activationFunction, final UnaryOperator<Matrix> outputFunction, final int... numberOfNeurons) {

    if (numberOfNeurons.length <= 2) {
      throw new IllegalArgumentException("All networks must have 2 layers at least.");
    }

    firstLayer = new LinkedLayer(numberOfNeurons[0]);
    LinkedLayer layer = firstLayer;

    final int lastIdx = numberOfNeurons.length - 1;
    lastLayer = new OutputLinkedLayer(numberOfNeurons[lastIdx], activationFunction, outputFunction);

    for (int i = 1; i < numberOfNeurons.length; i++) {
      if (i == lastIdx) {
        layer.next(lastLayer);
      } else {
        layer = layer.next(new LinkedLayer(numberOfNeurons[i], activationFunction));
      }
    }
  }

  public abstract void propagateError(final Column expectedVal);

  private void feedForward() {
    lastLayer.feedForward();
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

  private void initInputs(final TrainingData data) {
    final double[] col = new double[data.size()];
    for (int i = 0; i < data.size(); i++) {
      col[i] = data.getInput(i);
    }
    firstLayer.addInput(new Column(col));
  }

  private void learn() {
    TrainingData data = null;
    for (int i = 0; i < batchs.length; i++) {
      for (int j = 0; j < batchs[i].length; j++) {
        data = batchs[i][j];

        initInputs(data);
        feedForward();
        propagateError(new Column(data.getExpectedValue()));

      }
      updateWeights();
    }
  }

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

  private void updateWeights(final LinkedLayer layer) {
    if (layer.getPrevious() == null) {
      return;
    }
    Matrix output = null;
    Matrix weightError = null;
    Matrix biasError = null;

    final List<Matrix> errors = layer.getErrors();
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

  public abstract Matrix output();

}
