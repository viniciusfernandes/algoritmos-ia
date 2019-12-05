package br.com.inteligenciaartificial.algoritmos.neural;

import java.util.function.UnaryOperator;

import br.com.inteligenciaartificial.algoritmos.math.Matrix;

public class OutputLinkedLayer extends LinkedLayer {
    private Matrix expectedValue;
    private UnaryOperator<Matrix> outputFunction;

    public OutputLinkedLayer() {

    }

    public OutputLinkedLayer(final int neuronNumber, final UnaryOperator<Matrix> activation, final UnaryOperator<Matrix> outputFunction) {
        super(neuronNumber, activation);
        this.outputFunction = outputFunction;
    }

    public OutputLinkedLayer expectedValue(final Matrix value) {
        expectedValue = value;
        return this;
    }

    public Matrix propagateError() {
        final Matrix input = activate();
        final Matrix derivative = input.apply(MathUtils::sigmoidDerivative);

        final Matrix outputVal = outputFunction.apply(input);

        final Matrix gradient = outputVal.sub(expectedValue);
        return gradient.dot(derivative);
    }
}
