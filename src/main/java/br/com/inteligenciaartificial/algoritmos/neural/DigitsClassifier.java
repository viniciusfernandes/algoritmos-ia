package br.com.inteligenciaartificial.algoritmos.neural;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class DigitsClassifier {
    private Digit[][] batchs;
    private double[] biases;
    private final double[] inputs = new double[28 * 28];
    private final double[] neurons = new double[15];
    private final double[] outputs = new double[10];

    private Digit[] trainingData;
    private double[] weights;

    // Neural network leanin rate
    private final double rate;
    private final int batchSize;

    public DigitsClassifier(final double learningRate, final int batchSize) {
        rate = learningRate;
        this.batchSize = batchSize;

        // Initializing biases and weights with randomly to apply the stoschastic gradient
        // descent algorithm later.
        initBiases(biases);
        initWeights(weights);
    }

    private void batching(final Digit[] data, final int batchSize) {

        final int rest = data.length % batchSize;
        int batchNum = data.length - rest;
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
                batchs[ibatch][j] = trainingData[j];
            }
            ibatch++;
        }

    }

    private void calculatingInputs() {

    }

    public int classify(final Digit digit) {
        int d = -1;
        final double[] output = output();
        for (int i = 0; i < output.length - 1; i++) {
            d = output[i] <= output[i + 1] ? i + 1 : i;
        }
        return d;
    }

    private void initBiases(final double[] biases) {
        for (int i = 0; i < biases.length; i++) {
            biases[i] = Math.random();
        }
    }

    private void initWeights(final double[] weights) {
        for (int i = 0; i < weights.length; i++) {
            weights[i] = Math.random();
        }
    }

    public void learn(final TrainingDigit[] data) {

        trainingData = data;
        shuffleData(trainingData);
        batching(trainingData, batchSize);
        updateWeightsAndBiases();
    }

    private double[] output() {
        return new double[] {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    }

    private void shuffleData(final Digit[] data) {
        final List<Digit> list = new ArrayList<>(data.length);
        Collections.shuffle(list);
        for (int i = 0; i < list.size(); i++) {
            data[i] = list.get(i);
        }
    }

    private void updateWeightsAndBiases() {
        final int m = batchs.length;
        for (int i = 0; i < batchs.length; i++) {

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

    private double[] feedForward(final double[] a) {
        final double[] a_ = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            a_[i] = a[i] * weights[i] + biases[i];
        }
        return a_;
    }
}
