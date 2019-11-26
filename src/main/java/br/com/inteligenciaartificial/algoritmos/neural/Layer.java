package br.com.inteligenciaartificial.algoritmos.neural;

import java.util.ArrayList;
import java.util.List;

import br.com.inteligenciaartificial.algoritmos.math.Matrix;

public class Layer {
    private Matrix biases;
    private List<Matrix> errors;

    private Matrix input;
    private Matrix error;

    private final List<Matrix> inputs = new ArrayList<>();
    private final List<Matrix> outputs = new ArrayList<>();
    private Matrix output;
    private Matrix weight;

    public Layer() {

    }

    public Matrix getBiases() {
        return biases;
    }

    public Matrix getInput() {
        return input;
    }

    public Matrix getInput(final int index) {
        return inputs.get(index);
    }

    public Matrix getOutput() {
        return output;
    }

    public Matrix getWeight() {
        return weight;
    }

    public void setBiases(final Matrix biases) {
        this.biases = biases;
    }

    public void setInput(final Matrix input) {
        this.input = input;
    }

    public void setOutput(final Matrix outuput) {
        output = outuput;
    }

    public void setWeight(final Matrix weight) {
        this.weight = weight;
    }

    public Matrix activate() {
        final Matrix activation = input.operate(this::sigmoid);
        outputs.add(activation);
        return activation;
    }

    public Matrix getOutput(final int index) {
        return outputs.get(index);
    }

    private double sigmoid(final double z) {
        return 1d / (1d + Math.pow(Math.E, -z));
    }

    public Matrix activateDerivative() {
        return input.operate(this::sigmoidDerivative);
    }

    private double sigmoidDerivative(final double z) {
        final double sig = sigmoid(z);
        // Essa eh a expressao algebrica da derivada da funcao sigmoid.
        return sig * (1 - sig);
    }

    public Matrix weightedError() {
        return weight.multiply(error);
    }

    public Matrix subtractWeightError(final Matrix error) {
        weight = weight.sub(error);
        return weight;
    }

    public Matrix subtractBiasError(final Matrix error) {
        biases = biases.sub(error);
        return biases;
    }

    public Matrix weightedInput(final Matrix input) {
        this.input = weight.transpose().multiply(input).sum(biases);
        inputs.add(input);
        return this.input;
    }

    public List<Matrix> getErrors() {
        return errors;
    }

    public Matrix getError(final int index) {
        return errors.get(index);
    }

    public void setErrors(final List<Matrix> errors) {
        this.errors = errors;
    }

    public void addError(final Matrix error) {
        if (errors == null) {
            errors = new ArrayList<>();
        }
        errors.add(error);
        this.error = error;
    }

    public void clear() {
        errors.clear();
        inputs.clear();
        outputs.clear();
        input = null;
        output = null;
        error = null;
    }

}
