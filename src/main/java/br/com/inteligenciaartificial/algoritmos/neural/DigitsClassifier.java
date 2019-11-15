package br.com.inteligenciaartificial.algoritmos.neural;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import br.com.inteligenciaartificial.algoritmos.math.Column;
import br.com.inteligenciaartificial.algoritmos.math.Matrix;
import br.com.inteligenciaartificial.algoritmos.math.MultiMatrix;
import br.com.inteligenciaartificial.algoritmos.math.Row;

public class DigitsClassifier {
    private static final int HIDDEN_LAYER_SIZE = 3;
    private static final int INPUT_LAYER_SIZE = Digit.PIXELS_PER_DIGIT;
    private static final int LAYERS = 3;
    private static final int OUTPUT_LAYER_SIZE = 10;
    private final MultiMatrix A = new MultiMatrix(null, new Column(HIDDEN_LAYER_SIZE), new Column(OUTPUT_LAYER_SIZE));
    private final MultiMatrix B = new MultiMatrix(null, new Column(HIDDEN_LAYER_SIZE), new Column(OUTPUT_LAYER_SIZE));

    private TrainingDigit[][] batchs;

    private int batchSize;
    private final MultiMatrix Error = new MultiMatrix(LAYERS);

    private final Column neurons = new Column(HIDDEN_LAYER_SIZE);
    private final Column outputs = new Column(OUTPUT_LAYER_SIZE);
    // Neural network leanin rate
    private final double rate;

    private final MultiMatrix W = new MultiMatrix(new Matrix(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE),
                    new Matrix(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE), null);
    // Setup the transpose matrix of Z
    private final MultiMatrix Z = new MultiMatrix(new Row(INPUT_LAYER_SIZE), new Row(HIDDEN_LAYER_SIZE), null);

    public DigitsClassifier(final double learningRate, final int batchSize) {
        rate = learningRate;
        this.batchSize = batchSize;

        // Initializing biases and weights with randomly to apply the stoschastic
        // gradient
        // descent algorithm later.
        initBiases();
        initWeights();
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

    private void calcOutputErrors() {
        backPropagation(LAYERS, Error.get(LAYERS - 1));
    }

    private Matrix zDerivative = null;
    private Matrix wTranspose = null;
    private Matrix previousError = null;

    private void backPropagation(final int layer, final Matrix error) {
        if (layer <= 1) {
            return;
        }
        zDerivative = Z.apply(layer, this::sigmoidDifferential);
        wTranspose = W.get(layer).transpose();

        previousError = wTranspose.multiply(error).dot(zDerivative);
        backPropagation(layer - 1, previousError);
    }

    private int classify() {
        return 1;
    }

    public int classify(final Digit digit) {
        int d = -1;
        final double[] output = outputs();
        for (int i = 0; i < output.length - 1; i++) {
            d = output[i] <= output[i + 1] ? i + 1 : i;
        }
        return d;
    }

    private void feedForward() {

        Matrix Z1 = null;
        final int layer = LAYERS - 1;
        for (int l = 0; l < layer; l++) {
            Z1 = Z.get(l).transpose().multiply(W.get(l));

            W.set(l + 1, Z1);
            A.set(l + 1, Z1.apply(this::sigmoid));
        }
    }

    private void initBiases() {
        B.initRandom();
    }

    private void initInputs(final Digit digit) {
        final double[] pixels = new double[INPUT_LAYER_SIZE];
        for (int i = 0; i < digit.pixels.length; i++) {
            pixels[i] = digit.pixels[i];
        }
        Z.set(0, new Column(pixels));
    }

    private void initWeights() {
        W.initRandom();
    }

    public void learn(final List<TrainingDigit> data) {
        shuffleData(data);
        initBatchs(data, batchSize);
        updateWeightsAndBiases();
    }

    private double[] outputs() {
        return new double[] {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
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

    private void updateWeightsAndBiases() {
        for (int i = 0; i < batchs.length; i++) {
            for (int j = 0; j < batchs[i].length; j++) {
                initInputs(batchs[i][j]);
            }
            feedForward();
            calcOutputErrors();
        }
    }

    public static void main(final String[] args) {
        final TrainingDigit digit = new TrainingDigit(new int[INPUT_LAYER_SIZE], 1);
        final List<TrainingDigit> dataTraining = new ArrayList<>();
        dataTraining.add(digit);

        final DigitsClassifier classifier = new DigitsClassifier(0.001, 2);
        classifier.learn(dataTraining);
    }
}
