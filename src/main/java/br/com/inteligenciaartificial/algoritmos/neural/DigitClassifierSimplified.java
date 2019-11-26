package br.com.inteligenciaartificial.algoritmos.neural;

import java.util.Collections;
import java.util.List;

import br.com.inteligenciaartificial.algoritmos.math.Column;
import br.com.inteligenciaartificial.algoritmos.math.Matrix;

public class DigitClassifierSimplified {
    private static final int HIDDEN_LAYER_INDEX;
    private static final int HIDDEN_LAYER_SIZE;
    private static final int INPUT_LAYER_INDEX;
    private static final int INPUT_LAYER_SIZE;
    private static final int NUM_LAYERS;
    private static final int OUTPUT_LAYER_INDEX;
    private static final int OUTPUT_LAYER_SIZE;
    static {
        INPUT_LAYER_INDEX = 0;
        INPUT_LAYER_SIZE = Digit.PIXELS_PER_DIGIT;
        NUM_LAYERS = 3;
        HIDDEN_LAYER_INDEX = 1;
        HIDDEN_LAYER_SIZE = 15;
        OUTPUT_LAYER_SIZE = Digit.DIGITS_SIZE_SET;
        OUTPUT_LAYER_INDEX = NUM_LAYERS - 1;
    }

    private TrainingDigit[][] batchs;
    private int batchSize;

    // Neural network leanin rate
    private final double errorRate;

    private final Layer inLayer = new Layer();
    private final Layer hiddenLayer = new Layer();
    private final Layer outLayer = new Layer();

    public DigitClassifierSimplified(final double errorRate, final int batchSize) {
        this.errorRate = errorRate;
        this.batchSize = batchSize;

        initLayers();
        // Initializing biases and weights with randomly to apply the stoschastic
        // gradient
        // descent algorithm later.
        initBiases();
        initWeights();
    }

    private void backPropagation(final Column expectedVal) {
        final Matrix zDerivative = outLayer.getInput().operate(this::sigmoidDifferential);
        final Matrix activation = outLayer.sigmoid();

        final Matrix gradC = outputValue(activation).sub(expectedVal);
        Matrix error = gradC.dot(zDerivative);

        outLayer.setError(error);

        error = outLayer.weightedError().dot(hiddenLayer.sigmoidDerivative());
        hiddenLayer.setError(error);

    }

    public int classify(final Digit digit) {
        initInputs(digit);
        feedForward();

        return output(outLayer.getInput());
    }

    private void feedForward() {
        Matrix output = inLayer.getOutput();
        hiddenLayer.weightedInput(output);

        output = hiddenLayer.sigmoid();
        outLayer.weightedInput(output);
    }

    private void initBatchs(final List<TrainingDigit> data, int amountBatchs) {

        final int rest = data.size() % amountBatchs;
        batchSize = (data.size() - rest) / amountBatchs;
        if (batchSize == 0) {
            amountBatchs = 1;
            batchSize = data.size();
        }

        batchs = new TrainingDigit[amountBatchs][batchSize];
        int begin = 0;
        int end = 0;

        for (int b = 0; b < amountBatchs; b++) {
            begin = b * batchSize;
            end = begin + batchSize;

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
        inLayer.setOutuput(new Column(INPUT_LAYER_SIZE));

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

        }
    }

    public void learn(final List<TrainingDigit> data) {
        shuffleData(data);
        initBatchs(data, batchSize);
        learn();
        updateWeights();
    }

    private void shuffleData(final List<TrainingDigit> data) {
        Collections.shuffle(data);
    }

    private double sigmoid(final double z) {
        return 1d / (1d + Math.pow(Math.E, -z));
    }

    private double sigmoidDifferential(final double z) {
        final double sig = sigmoid(z);
        // Essa eh a expressao algebrica da derivada da funcao sigmoid.
        return sig * (1 - sig);
    }

    private void updateWeights() {
        updateWeights(layers[OUTPUT_LAYER_INDEX]);
    }

    private void updateWeights(final Layer layer) {
        final int prev = layer.getIndex() - 1;
        if (layer.getIndex() <= HIDDEN_LAYER_INDEX) {
            return;
        }

        final Layer prevLayer = layers[prev];
        Matrix error = layer.getError();
        final Matrix activation = prevLayer.sigmoid();
        final Matrix weightError = activation.multiply(error.transpose());
        weightError.operate(e -> e * errorRate / batchSize);

        Matrix weight = layer.getWeight();
        weight = weight.sub(weightError);
        layer.setWeight(weight);

        Matrix biases = layer.getBiases();
        error = error.operate(e -> e * errorRate / batchSize);
        biases = biases.sub(error);
        layer.setBiases(biases);

        updateWeights(prevLayer);
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

    private Column outputValue(final Matrix activation) {
        final int digit = output(activation);

        final Column out = new Column(activation.getRowNum());
        out.set(digit, 1);
        return out;
    }

}
