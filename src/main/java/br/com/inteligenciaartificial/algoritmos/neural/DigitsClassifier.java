package br.com.inteligenciaartificial.algoritmos.neural;

import java.util.Collections;
import java.util.List;

public class DigitsClassifier {
    private TrainingBatch[] batchs;
    private double[] biases;
    private final double[] A_l = new double[15];
    private final double[] Z_l = new double[Digit.PIXELS_PER_DIGIT];
    private final double[] neurons = new double[15];
    private final double[][] outputErros = new double[1][15];
    private final double[] outputs = new double[10];

    private final double[][] weights = new double[15][Digit.PIXELS_PER_DIGIT];

    // Neural network leanin rate
    private final double rate;
    private final int batchSize;

    public DigitsClassifier(final double learningRate, final int batchSize) {
        rate = learningRate;
        this.batchSize = batchSize;

        // Initializing biases and weights with randomly to apply the stoschastic gradient
        // descent algorithm later.
        initBiases();
        initWeights();
    }

    private void batching(final List<TrainingDigit> data, final int batchSize) {

        final int rest = data.size() % batchSize;
        int batchNum = data.size() - rest;
        batchNum /= batchSize;

        int ibatch = 0;
        int last = 0;
        while (ibatch <= batchNum) {

            if (rest != 0 && ibatch == batchNum) {
                last = rest;
            } else {
                last = batchSize;
            }

            for (int j = ibatch * batchSize; j < last; j++) {
                batchs[ibatch].add(data.get(j));
            }
            ibatch++;
        }

    }

    public int classify(final Digit digit) {
        int d = -1;
        final double[] output = output();
        for (int i = 0; i < output.length - 1; i++) {
            d = output[i] <= output[i + 1] ? i + 1 : i;
        }
        return d;
    }

    private void initBiases() {
        for (int i = 0; i < biases.length; i++) {
            biases[i] = Math.random();
        }
    }

    private void initWeights() {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = Math.random();
            }
        }
    }

    public void learn(final List<TrainingDigit> data) {
        shuffleData(data);
        batching(data, batchSize);
        updateWeightsAndBiases();
    }

    private double[] output() {
        return new double[] {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    }

    private void shuffleData(final List<TrainingDigit> data) {
        Collections.shuffle(data);
    }

    private void updateWeightsAndBiases() {
        final int m = batchs.length;
        for (int i = 0; i < batchs.length; i++) {
            initInputs(batchs[i]);
            feedForward();
        }
    }

    private void initInputs(final TrainingBatch batch) {
        for (final Digit digit : batch.digits) {
            for (int i = 0; i < digit.pixels.length; i++) {
                Z_l[i] = digit.pixels[i];
            }
        }
    }

    private double sigmoid(final double z) {
        return 1d / (1d + Math.pow(Math.E, -z));
    }

    private double[] sigmoid(final double[] z) {
        final double[] outs = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            outs[i] = sigmoid(z[i]);
        }
        return outs;
    }

    private void feedForward() {
        for (int i = 1; i < A_l.length; i++) {
            for (int j = 0; j < Z_l.length; j++) {
                A_l[i] += Z_l[j] * weights[i][j];
            }
            A_l[i] += biases[i];
        }
    }

    private void calcErrors() {}

    private void calcErrors(final double[][] Zs) {

    }
}
