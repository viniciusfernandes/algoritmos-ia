package br.com.inteligenciaartificial.algoritmos.neural;

import java.util.Collections;
import java.util.List;
import java.util.function.DoubleUnaryOperator;
import java.util.function.UnaryOperator;

import br.com.inteligenciaartificial.algoritmos.math.Column;
import br.com.inteligenciaartificial.algoritmos.math.Matrix;

public abstract class NeuralNetwork {

  private TrainingData[][] batchs;
  private static final int BATCH_SIZE = 1;

  final OutputLinkedLayer lastLayer;

  final LinkedLayer linkedLayer;

  public NeuralNetwork(
          final DoubleUnaryOperator activationFunction, final UnaryOperator<Matrix> outputFunction, final int... layersNumber) {

    if (layersNumber.length <= 2) {
      throw new IllegalArgumentException("All networks must have 2 layers at least.");
    }

    linkedLayer = new LinkedLayer(layersNumber[0], m -> m.apply(activationFunction));
    LinkedLayer layer = linkedLayer;
    final int lastIdx = layersNumber.length - 1;
    lastLayer = new OutputLinkedLayer(layersNumber[lastIdx], m -> m.apply(activationFunction), outputFunction);

    for (int i = 1; i < layersNumber.length; i++) {
      if (i == lastIdx) {
        layer.next(lastLayer);
      } else {
        layer = layer.next(new LinkedLayer(layersNumber[i], m -> m.apply(activationFunction)));
      }
    }
  }

  public abstract void backPropagate(final Column expectedVal);

  private void feedForward() {
    lastLayer.feedForward();
  }

  private void initBatchs(final List<TrainingData> data) {
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
    linkedLayer.addInput(new Column(col));
  }

  private void learn() {
    TrainingData data = null;
    for (int i = 0; i < batchs.length; i++) {
      for (int j = 0; j < batchs[i].length; j++) {
        data = batchs[i][j];

        initInputs(data);
        feedForward();
        backPropagate(new Column(data.getExpectedValue()));

      }
      updateWeights();
    }
  }

  private void shuffleData(final List<TrainingData> data) {
    Collections.shuffle(data);
  }

  public void training(final List<TrainingData> data) {
    shuffleData(data);
    initBatchs(data);
    learn();
  }

  public abstract void updateWeights();

}
