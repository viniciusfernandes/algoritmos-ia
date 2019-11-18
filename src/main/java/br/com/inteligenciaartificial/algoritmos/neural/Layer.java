package br.com.inteligenciaartificial.algoritmos.neural;

import br.com.inteligenciaartificial.algoritmos.math.Matrix;

public class Layer {
    private Matrix input;
    private Matrix outuput;
    private Matrix weight;
    private final int index;
    private Matrix biases;

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

}
