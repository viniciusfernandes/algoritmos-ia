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
    private TrainingDigit[][] batchs;

    private final int batchSize = 3;

    private final Layer hiddenLayer = new Layer(input -> input.apply(this::sigmoid));
    private final Layer inLayer = new Layer(input -> input);
    private final Layer outLayer = new Layer(input -> input.apply(this::sigmoid));

    public DigitClassifier() {
        initLayers();
        // Initializing biases and weights with randomly to apply the stoschastic
        // gradient
        // descent algorithm later.
        initBiases();
        initWeights();
    }

    private Matrix activationDerivative(final Matrix matrix) {
        return matrix.apply(z -> {
            final double sig = sigmoid(z);
            // Essa eh a expressao algebrica da derivada da funcao sigmoid.
            return sig * 1 - sig;
        });

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

    public int classify(final Digit digit) {
        initInputs(digit);
        feedForward();
        final Column out = output();
        for (int i = 0; i < out.getRowNum(); i++) {
            if (out.get(i) == 1d) {
                return i;
            }
        }
        throw new IllegalStateException(String.format("Fail in classify the digit %d", digit.getValue()));
    }

    private void clearLayers() {
        inLayer.clear();
        hiddenLayer.clear();
        outLayer.clear();
    }

    private void feedForward() {
        inLayer.inputing(hiddenLayer).activate(outLayer);
    }

    private void initBatchs(final List<TrainingDigit> data) {
        final int batchNum = data.size() / batchSize;
        batchs = new TrainingDigit[batchNum][batchSize];

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

    private void initBiases() {
        hiddenLayer.getBiases().initRandom();
        outLayer.getBiases().initRandom();
    }

    private void initInputs(final Digit digit) {
        final double[] pixels = new double[Digit.PIXELS_PER_DIGIT];
        for (int i = 0; i < Digit.PIXELS_PER_DIGIT; i++) {
            pixels[i] = digit.getInputValue(i);
        }
        inLayer.addInput(new Column(pixels));
    }

    private void initLayers() {
        inLayer.addInput(new Column(INPUT_LAYER_SIZE));

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
        TrainingDigit digit = null;
        for (int i = 0; i < batchs.length; i++) {
            for (int j = 0; j < batchs[i].length; j++) {
                digit = batchs[i][j];

                initInputs(digit);
                feedForward();
                backPropagate(new Column(digit.getExpectedOutput()));

            }
            updateWeights();
            clearLayers();
        }
    }

    private Column output() {
        final Matrix activation = outLayer.activate();
        final double[] out = new double[activation.getRowNum()];

        int max = 0;
        final int last = activation.getRowNum() - 1;
        for (int i = 0; i < last; i++) {
            if (activation.get(max, 0) < activation.get(i + 1, 0)) {
                max = i + 1;
            }
        }

        out[max] = 1;
        return new Column(out);
    }

    private void shuffleData(final List<TrainingDigit> data) {
        Collections.shuffle(data);
    }

    private double sigmoid(final double z) {
        return 1d / (1d + Math.exp(-z));
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

        outWeightError = outWeightError.apply(e -> e * ERROR_RATE / batchSize);
        outBiasError = outBiasError.apply(e -> e * ERROR_RATE / batchSize);

        hiddenWeightError = hiddenWeightError.apply(e -> e * ERROR_RATE / batchSize);
        hiddenBiasError = hiddenBiasError.apply(e -> e * ERROR_RATE / batchSize);

        outLayer.subtractWeightError(outWeightError);
        outLayer.subtractBiasError(outBiasError);

        hiddenLayer.subtractWeightError(hiddenWeightError);
        hiddenLayer.subtractBiasError(hiddenBiasError);
    }

}
