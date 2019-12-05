package br.com.inteligenciaartificial.algoritmos.neural;

import java.util.function.DoubleUnaryOperator;
import java.util.function.UnaryOperator;

import br.com.inteligenciaartificial.algoritmos.math.Column;
import br.com.inteligenciaartificial.algoritmos.math.Matrix;

public class GradientDescentNetwork extends NeuralNetwork {

  public GradientDescentNetwork(
          final DoubleUnaryOperator activationFunction, final UnaryOperator<Matrix> outputFunction, final int[] layersNumber) {
    super(activationFunction, outputFunction, layersNumber);
  }

  @Override
  public void backPropagate(final Column expectedVal) {
    final Matrix input = lastLayer.activate();
    final Matrix derivative = input.apply(MathUtils::sigmoidDerivative);

    final Matrix outputVal = lastLayer.output();

    final Matrix gradient = outputVal.sub(expectedVal);
    final Matrix outError = gradient.dot(derivative);
    lastLayer.addError(outError);

    backPropagate(lastLayer.getPrevious());
  }

  public void backPropagate(final LinkedLayer layer) {
    if (layer.getPrevious() == null) {
      return;
    }
    final Matrix input = layer.activate();
    final Matrix derivative = input.apply(MathUtils::sigmoidDerivative);
    Matrix error = layer.getNext().getError();
    final Matrix outWeight = layer.getNext().getWeight();
    error = outWeight.multiply(error).dot(derivative);
    layer.addError(error);

    backPropagate(layer.getPrevious());
  }

  @Override
  public void updateWeights() {
    // TODO Auto-generated method stub

  }

}
