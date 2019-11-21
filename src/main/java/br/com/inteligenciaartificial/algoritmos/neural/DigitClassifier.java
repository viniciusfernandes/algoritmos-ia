package br.com.inteligenciaartificial.algoritmos.neural;

import java.util.Collections;
import java.util.List;

import br.com.inteligenciaartificial.algoritmos.math.Column;
import br.com.inteligenciaartificial.algoritmos.math.Matrix;

public class DigitClassifier {
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

    private final Layer[] layers = new Layer[NUM_LAYERS];

    public DigitClassifier(final double errorRate, final int batchSize) {
        this.errorRate = errorRate;
        this.batchSize = batchSize;

        initLayers();
        // Initializing biases and weights with randomly to apply the stoschastic
        // gradient
        // descent algorithm later.
        initBiases();
        initWeights();
    }

    private void backPropagation(final Layer layer) {
        if (layer.getIndex() <= HIDDEN_LAYER_INDEX) {
            return;
        }
        final Layer prevLayer = layers[layer.getIndex() - 1];
        final Matrix error = layer.weightedError().dot(prevLayer.sigmoidDerivative());
        prevLayer.setError(error);
        backPropagation(prevLayer);
    }

    private void calcOutputErrors(final Column expectedVal) {
        final Layer lastLayer = layers[OUTPUT_LAYER_INDEX];
        final Matrix zDerivative = lastLayer.getInput().operate(this::sigmoidDifferential);
        final Matrix activation = lastLayer.sigmoid();

        final Matrix gradC = activation.sub(expectedVal);
        final Matrix error = gradC.dot(zDerivative);

        lastLayer.setError(error);

        backPropagation(lastLayer);
    }

    public int classify(final Digit digit) {
        initInputs(digit);
        feedForward();

        final Matrix result = layers[OUTPUT_LAYER_INDEX].getInput();
        int max = -1;
        final int last = result.getRowNum() - 1;
        for (int i = 0; i <= last; i++) {
            if (i == last) {
                if (max == -1) {
                    max = last;
                }
                break;
            }
            if (result.get(i, 0) >= result.get(i + 1, 0)) {
                max = i;
            }
        }
        if (max != -1) {
            return max;
        }
        throw new IllegalStateException("Fail in digit classification. There is no result.");
    }

    private void feedForward() {
        feedForward(layers[0]);
    }

    private void feedForward(final Layer layer) {
        final int next = layer.getIndex() + 1;
        if (layer.getIndex() >= OUTPUT_LAYER_INDEX) {
            return;
        }

        Matrix output = null;
        if (layer.getIndex() == 0) {
            output = layer.getOutput();
        } else {
            output = layer.sigmoid();
        }

        final Layer nextLayer = layers[next];
        nextLayer.weightedInput(output);
        feedForward(nextLayer);
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
        layers[1].getBiases().initRandom();
        layers[2].getBiases().initRandom();
    }

    private void initInputs(final Digit digit) {
        final Matrix out = layers[0].getOutput();
        for (int i = 0; i < Digit.PIXELS_PER_DIGIT; i++) {
            out.set(i, 0, digit.getPixel(i));
        }
    }

    private void initLayers() {
        final Layer l0 = new Layer(INPUT_LAYER_INDEX);
        l0.setOutuput(new Column(INPUT_LAYER_SIZE));

        final Layer l1 = new Layer(HIDDEN_LAYER_INDEX);
        l1.setBiases(new Column(HIDDEN_LAYER_SIZE));
        l1.setWeight(new Matrix(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE));

        final Layer l2 = new Layer(OUTPUT_LAYER_INDEX);
        l2.setBiases(new Column(OUTPUT_LAYER_SIZE));
        l2.setWeight(new Matrix(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE));

        layers[0] = l0;
        layers[1] = l1;
        layers[2] = l2;
    }

    private void initWeights() {
        layers[1].getWeight().initRandom();
        layers[2].getWeight().initRandom();
    }

    private void learn() {
        TrainingDigit data = null;
        for (int i = 0; i < batchs.length; i++) {
            for (int j = 0; j < batchs[i].length; j++) {
                data = batchs[i][j];

                initInputs(data);
                feedForward();
                calcOutputErrors(new Column(data.getExpectedOutput()));
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
}
