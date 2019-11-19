package br.com.inteligenciaartificial.algoritmos.neural;

import br.com.inteligenciaartificial.algoritmos.math.Matrix;

public class Layer {
    private Matrix biases;
    private Matrix error;
    private final int index;
    private Matrix input;
    private Matrix outuput;
    private Matrix weight;

    public Layer(final int index) {
        this.index = index;
    }

    public Matrix activate() {
        final Matrix out = weight.transpose().multiply(input).sum(biases);
        return out.operate(this::sigmoid);
    }

    public Matrix getBiases() {
        return biases;
    }

    public Matrix getError() {
        return error;
    }

    public int getIndex() {
        return index;
    }

    public Matrix getInput() {
        return input;
    }

    public Matrix getOutput() {
        return outuput;
    }

    public Matrix getWeight() {
        return weight;
    }

    public void setBiases(final Matrix biases) {
        this.biases = biases;
    }

    public void setError(final Matrix error) {
        this.error = error;
    }

    public void setInput(final Matrix input) {
        this.input = input;
    }

    public void setOutuput(final Matrix outuput) {
        this.outuput = outuput;
    }

    public void setWeight(final Matrix weight) {
        this.weight = weight;
    }

    public Matrix sigmoid() {
        return input.operate(this::sigmoid);
    }

    private double sigmoid(final double z) {
        return 1d / (1d + Math.pow(Math.E, -z));
    }

    public Matrix sigmoidDerivative() {
        return input.operate(this::sigmoidDerivative);
    }

    private double sigmoidDerivative(final double z) {
        final double sig = sigmoid(z);
        // Essa eh a expressao algebrica da derivada da funcao sigmoid.
        return sig * (1 - sig);
    }

    public Matrix weightedInput() {
        return weight.transpose().multiply(input).sum(biases);
    }

}
