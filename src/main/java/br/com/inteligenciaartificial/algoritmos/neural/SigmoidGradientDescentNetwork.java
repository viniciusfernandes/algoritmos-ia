package br.com.inteligenciaartificial.algoritmos.neural;

import java.util.function.UnaryOperator;

import br.com.inteligenciaartificial.algoritmos.math.Column;
import br.com.inteligenciaartificial.algoritmos.math.Matrix;

public class SigmoidGradientDescentNetwork extends NeuralNetwork {

  private final UnaryOperator<Matrix> activationDerivativeFunction;

  public SigmoidGradientDescentNetwork(final UnaryOperator<Matrix> outputFunction, final int[] layersNumber) {
    super(m -> m.apply(MathUtils::sigmoid), outputFunction, layersNumber);
    activationDerivativeFunction = m -> m.apply(MathUtils::sigmoidDerivative);
  }

  @Override
  public void propagateError(final Column expectedVal) {
    final Matrix input = lastLayer.activate();
    final Matrix derivative = input.apply(MathUtils::sigmoidDerivative);

    final Matrix outputVal = lastLayer.output();

    final Matrix gradient = outputVal.sub(expectedVal);
    final Matrix outError = gradient.dot(derivative);
    lastLayer.addError(outError);

    propagateError(lastLayer.getPrevious());
  }

  private void propagateError(final LinkedLayer layer) {
    if (layer.getPrevious() == null) {
      return;
    }
    final Matrix input = layer.activate();
    final Matrix derivative = activationDerivativeFunction.apply(input);
    Matrix error = layer.getNext().getError();
    final Matrix outWeight = layer.getNext().getWeight();
    error = outWeight.multiply(error).dot(derivative);
    layer.addError(error);

    propagateError(layer.getPrevious());
  }
}
