package br.com.inteligenciaartificial.algoritmos.neuralnetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.function.UnaryOperator;

import br.com.inteligenciaartificial.algoritmos.math.Matrix;

public class Layer {
    private final UnaryOperator<Matrix> activation;

    private Matrix biases;

    private Matrix error;
    private List<Matrix> errors = new ArrayList<>();

    private Matrix input;
    private final List<Matrix> inputs = new ArrayList<>();
    private final List<Matrix> outputs = new ArrayList<>();

    private Matrix weight;

    public Layer() {
        activation = null;
    }

    public Layer(final UnaryOperator<Matrix> activation) {
        this.activation = activation;
    }

    public Matrix activate() {
        return activation.apply(estimulate(input));
    }

    public Matrix activate(final int indexInput) {
        return activation.apply(estimulate(getInput(indexInput)));
    }

    public Layer activate(final Layer nextLayer) {
        nextLayer.addInput(activation.apply(estimulate(input)));
        return nextLayer;
    }

    public void addError(final Matrix error) {
        errors.add(error);
        this.error = error;
    }

    public void addInput(final Matrix input) {
        setInput(input);
        inputs.add(input);
    }

    public Layer backPropagate(final Layer previousLayer) {
        return previousLayer;
    }

    public void clear() {
        errors.clear();
        inputs.clear();
        outputs.clear();
        input = null;
        error = null;
    }

    public Matrix estimulate(final Matrix input) {
        return weight.transpose().multiply(input).sub(biases);
    }

    public Layer estimulating(final Layer nextLayer) {
        nextLayer.addInput(estimulate(input));
        return nextLayer;
    }

    public Matrix getBiases() {
        return biases;
    }

    public Matrix getError(final int index) {
        return errors.get(index);
    }

    public List<Matrix> getErrors() {
        return errors;
    }

    public Matrix getInput() {
        return input;
    }

    public Matrix getInput(final int index) {
        return inputs.get(index);
    }

    public Matrix getOutput(final int index) {
        return outputs.get(index);
    }

    public Matrix getWeight() {
        return weight;
    }

    public Layer inputing(final Layer nextLayer) {
        nextLayer.addInput(input);
        return nextLayer;
    }

    public int normalize(final double output) {
        return output > 0 ? 1 : -1;
    }

    public void setBiases(final Matrix biases) {
        this.biases = biases;
    }

    public void setErrors(final List<Matrix> errors) {
        this.errors = errors;
    }

    void setInput(final Matrix input) {
        this.input = input;
    }

    public void setWeight(final Matrix weight) {
        this.weight = weight;
    }

    public Matrix subtractBiasError(final Matrix error) {
        biases = biases.sub(error);
        return biases;
    }

    public Matrix subtractWeightError(final Matrix error) {
        weight = weight.sub(error);
        return weight;
    }

    public Matrix weightedError() {
        return weight.multiply(error);
    }
}
