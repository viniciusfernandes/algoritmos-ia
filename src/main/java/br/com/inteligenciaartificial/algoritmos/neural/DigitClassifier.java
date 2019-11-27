package br.com.inteligenciaartificial.algoritmos.neural;

import java.util.Collections;
import java.util.List;

import br.com.inteligenciaartificial.algoritmos.math.Column;
import br.com.inteligenciaartificial.algoritmos.math.Matrix;

public class DigitClassifier {
    // Neural network leanin rate
    private static final double ERROR_RATE = 0.01;
    private static final int HIDDEN_LAYER_SIZE;
    private static final int INPUT_LAYER_SIZE;
    private static final int OUTPUT_LAYER_SIZE;

    static {
        INPUT_LAYER_SIZE = Digit.PIXELS_PER_DIGIT;
        HIDDEN_LAYER_SIZE = 15;
        OUTPUT_LAYER_SIZE = Digit.DIGITS_SIZE_SET;
    }
    private final int batchLength = 1;

    private TrainingDigit[][] batchs;

    private final Layer hiddenLayer = new Layer();
    private final Layer inLayer = new Layer();
    private final Layer outLayer = new Layer();

    public DigitClassifier() {
        initLayers();
        // Initializing biases and weights with randomly to apply the stoschastic
        // gradient
        // descent algorithm later.
        initBiases();
        initWeights();
    }

    private void backPropagation(final Column expectedVal) {
        final Matrix z2 = hiddenLayer.weighting(inLayer.getOutput());
        final Matrix z3 = outLayer.activate(z2);

        final Matrix input = hiddenLayer.getInput();
        final Matrix actDerivative = outLayer.activateDerivative(input);

        final Matrix activation = outLayer.getInput();

        final Matrix gradC = activation.sub(expectedVal);
        Matrix error = gradC.dot(actDerivative);

        outLayer.addError(error);

        error = outLayer.weightedError().dot(hiddenLayer.activateDerivative());
        hiddenLayer.addError(error);
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
        outLayer.getBiases().initRandom();
    }

    private void initInputs(final Digit digit) {
        final Matrix out = inLayer.getOutput();
        for (int i = 0; i < Digit.PIXELS_PER_DIGIT; i++) {
            out.set(i, 0, digit.getPixel(i));
        }
    }

    private void initLayers() {
        inLayer.setOutput(new Column(INPUT_LAYER_SIZE));

        hiddenLayer.setBiases(new Column(HIDDEN_LAYER_SIZE));
        hiddenLayer.setWeight(new Matrix(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE));

        outLayer.setBiases(new Column(OUTPUT_LAYER_SIZE));
        outLayer.setWeight(new Matrix(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE));
    }

    private void initWeights() {
        hiddenLayer.getWeight().initRandom();
        outLayer.getWeight().initRandom();
    }

    private void learn() {
        TrainingDigit data = null;
        for (int i = 0; i < batchs.length; i++) {
            for (int j = 0; j < batchs[i].length; j++) {
                data = batchs[i][j];

                initInputs(data);
                feedForward();
                backPropagation(new Column(data.getExpectedOutput()));
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
