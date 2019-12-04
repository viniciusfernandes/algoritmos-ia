package br.com.inteligenciaartificial.algoritmos.neural;

import java.util.Collections;
import java.util.List;
import java.util.function.DoubleUnaryOperator;

import br.com.inteligenciaartificial.algoritmos.math.Column;
import br.com.inteligenciaartificial.algoritmos.math.Matrix;

public class NeuralNetwork<K, M> {

    private final DoubleUnaryOperator activationDerivaticeFunction;
    private final DoubleUnaryOperator activationFunction;
    private TrainingData[][] batchs;

    private final int batchSize = 1;

    private final LinkedLayer linkedLayer;
    private final LinkedLayer lastLayer;

    public NeuralNetwork(
                    final DoubleUnaryOperator activationFunction, final DoubleUnaryOperator activationDerivaticeFunction,
                    final int... numberOfneuros) {
        this.activationFunction = activationFunction;
        this.activationDerivaticeFunction = activationDerivaticeFunction;

        if (numberOfneuros.length <= 2) {
            throw new IllegalArgumentException("All networks must have 2 layers at least.");
        }

        linkedLayer = new LinkedLayer(numberOfneuros[0], m -> m.apply(activationFunction));
        LinkedLayer layer = linkedLayer;
        for (int i = 1; i < numberOfneuros.length; i++) {
            layer = layer.next(new LinkedLayer(numberOfneuros[i], m -> m.apply(activationFunction)));
        }
        lastLayer = layer;
    }

    private void initBatchs(final List<TrainingData> data) {
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

    private void initInputs(final TrainingData data) {
        final double[] col = new double[data.size()];
        for (int i = 0; i < data.size(); i++) {
            col[i] = data.getInput(i);
        }
        linkedLayer.setInput(new Column(col));
    }

    private void feedForward(final LinkedLayer layer) {
        if (layer == null) {
            return;
        }
        feedForward(layer.activate(layer.getNext()));
    }

    private void learn() {
        TrainingData data = null;
        for (int i = 0; i < batchs.length; i++) {
            for (int j = 0; j < batchs[i].length; j++) {
                data = batchs[i][j];

                initInputs(data);
                feedForward(linkedLayer);
                backPropagate(new Column(data.getExpectedValue()));

            }
            updateWeights();
            clearLayers();
        }
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
            output = hiddenLayer.activate(i);
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
            output = inLayer.getInput(i);
            if (hiddenWeightError == null || hiddenBiasError == null) {
                hiddenBiasError = error;
                hiddenWeightError = output.multiply(error.transpose());

            } else {
                hiddenBiasError = hiddenBiasError.sum(error);
                hiddenWeightError = hiddenWeightError.sum(output.multiply(error.transpose()));
            }
        }

        outWeightError = outWeightError.apply(e -> e * ERROR_RATE);
        outBiasError = outBiasError.apply(e -> e * ERROR_RATE);

        hiddenWeightError = hiddenWeightError.apply(e -> e * ERROR_RATE);
        hiddenBiasError = hiddenBiasError.apply(e -> e * ERROR_RATE);

        outLayer.subtractWeightError(outWeightError);
        outLayer.subtractBiasError(outBiasError);

        hiddenLayer.subtractWeightError(hiddenWeightError);
        hiddenLayer.subtractBiasError(hiddenBiasError);
    }

    private void backPropagate(final Column expectedVal) {
        Matrix input = outLayer.activate();
        Matrix derivative = activationDerivative(input);

        final Matrix outputVal = output();

        final Matrix gradient = outputVal.sub(expectedVal);
        final Matrix outError = gradient.dot(derivative);
        outLayer.addError(outError);

        input = hiddenLayer.activate();
        derivative = activationDerivative(input);
        final Matrix outWeight = outLayer.getWeight();
        final Matrix hiddenError = outWeight.multiply(outError).dot(derivative);
        hiddenLayer.addError(hiddenError);
    }

    private void shuffleData(final List<TrainingData<K, M>> data) {
        Collections.shuffle(data);
    }

    public void training(final List<TrainingData<K, M>> data) {
        shuffleData(data);
        initBatchs(data);
        learn();
    }

}
