package br.com.inteligenciaartificial.algoritmos.neural;

import br.com.inteligenciaartificial.algoritmos.math.Matrix;

public class Layer {
    private Matrix input;
    private Matrix outuput;
    private Matrix weight;
    private final int index;
    private Matrix biases;
    private Matrix error;

    public Layer(final int index) {
        this.index = index;
    }

    public Matrix getInput() {
        return input;
    }

    public void setInput(final Matrix input) {
        this.input = input;
    }

    public Matrix getOutuput() {
        return outuput;
    }

    public void setOutuput(final Matrix outuput) {
        this.outuput = outuput;
    }

    public Matrix getWeight() {
        return weight;
    }

    public void setWeight(final Matrix weight) {
        this.weight = weight;
    }

    public Matrix getBiases() {
        return biases;
    }

    public void setBiases(final Matrix biases) {
        this.biases = biases;
    }

    public int getIndex() {
        return index;
    }

    public Matrix getError() {
        return error;
    }

    public void setError(final Matrix error) {
        this.error = error;
    }

    public Matrix weightedInput() {
        input = weight.transpose().multiply(input).sum(biases);
        return input;
    }

    public void activateOutput() {
        outuput = input.operate(this::sigmoid);
    }

    private double sigmoid(final double z) {
        return 1d / (1d + Math.pow(Math.E, -z));
    }

    private double sigmoidDifferential(final double z) {
        final double sig = sigmoid(z);
        // Essa eh a expressao algebrica da derivada da funcao sigmoid.
        return sig * (1 - sig);
    }
}
