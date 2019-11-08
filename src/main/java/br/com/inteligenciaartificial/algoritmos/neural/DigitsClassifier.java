package br.com.inteligenciaartificial.algoritmos.neural;

import java.util.Collections;
import java.util.List;

public class DigitsClassifier {
    private final int neuronsNumber = 15;
    private TrainingBatch[] batchs;
    private final double[][] B_ln = new double[][] {new double[neuronsNumber], new double[10]};
    private final double[][] A_ln = new double[2][];
    private final double[][] Z_ln = new double[2][];
    private final double[][][] W_lnk = new double[][][] {new double[Digit.PIXELS_PER_DIGIT][neuronsNumber], new double[neuronsNumber][10]};

    private final double[] neurons = new double[neuronsNumber];
    private final double[][] outputErros = new double[1][neuronsNumber];
    private final double[] outputs = new double[10];

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
        final double[] output = outputs();
        for (int i = 0; i < output.length - 1; i++) {
            d = output[i] <= output[i + 1] ? i + 1 : i;
        }
        return d;
    }

    private void initBiases() {

        for (int i = 0; i < B_ln.length; i++) {
            for (final int j = 0; j < B_ln[i].length; i++) {
                B_ln[i][j] = Math.random();
            }

        }
    }

    private void initWeights() {
        for (int i = 0; i < W_lnk.length; i++) {
            for (int j = 0; j < W_lnk[i].length; j++) {
                for (final int k = 0; k < W_lnk[i][j].length; j++) {
                    W_lnk[i][j][k] = Math.random();
                }
            }
        }
    }

    public void learn(final List<TrainingDigit> data) {
        shuffleData(data);
        batching(data, batchSize);
        updateWeightsAndBiases();
    }

    private int classify() {
        return 1;
    }

    private double[] outputs() {
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
            outputs();
        }
    }

    private void initInputs(final TrainingBatch batch) {
        for (final Digit digit : batch.digits) {
            for (int i = 0; i < digit.pixels.length; i++) {
                Z_ln[0][i] = digit.pixels[i];
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
        final double tot = 0;
        for (int l = 0; l < Z_ln.length; l++) {
            for (int n = 0; n < Z_ln[l].length; n++) {
                // tot += Z_ln[l][n] * W_lnk[l][n];
            }

        }

        for (int k = 0; k < Z_ln.length; k++) {
            Z_ln[1][k] += Z_ln[0][k] * W_lnk[0][0][k];
        }
        // A_l[i] += biases[i];

    }

    private void calcErrors() {}

    private void calcErrors(final double[][] Zs) {

    }
}
