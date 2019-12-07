package br.com.inteligenciaartificial.algoritmos.neural;

import java.util.ArrayList;
import java.util.List;
import java.util.function.UnaryOperator;

import br.com.inteligenciaartificial.algoritmos.math.Column;
import br.com.inteligenciaartificial.algoritmos.math.Matrix;

public class LinkedLayer {
	private final UnaryOperator<Matrix> activationFunction;
	private Matrix biases;

	private Matrix error;

	private List<Matrix> errors = new ArrayList<>();

	private Matrix input;
	private final List<Matrix> inputs = new ArrayList<>();
	private final int neuronNumber;

	private LinkedLayer next;
	private Matrix output;
	private LinkedLayer previous;
	private Matrix weight;

	public LinkedLayer(final int neuronNumber) {
		this(neuronNumber, m -> m);
	}

	public LinkedLayer(final int neuronNumber, final UnaryOperator<Matrix> activationFunction) {
		this.activationFunction = activationFunction;
		this.neuronNumber = neuronNumber;
	}

	public Matrix activate() {
		return activate(input);
	}

	public Matrix activate(final int indexInput) {
		return activationFunction.apply(estimulate(getInput(indexInput)));
	}

	public Matrix activate(final Matrix input) {
		if (output == null) {
			output = activationFunction.apply(estimulate(input));
		}
		return output;
	}

	public void addError(final Matrix error) {
		errors.add(error);
		this.error = error;
	}

	public void addInput(final Matrix input) {
		this.input = input;
		inputs.add(input);
		output = null;
	}

	public void clear() {
		errors.clear();
		inputs.clear();
		input = null;
		output = null;
		error = null;
	}

	public Matrix estimulate(final Matrix input) {
		return weight.transpose().multiply(input).sub(biases);
	}

	public LinkedLayer estimulating(final LinkedLayer nextLayer) {
		nextLayer.addInput(estimulate(input));
		return nextLayer;
	}

	public void feedForward() {
		feedForward(this);
	}

	private void feedForward(final LinkedLayer layer) {
		if (layer.next == null) {
			return;
		}
		layer.next.addInput(activate(layer.input));
		feedForward(layer.next);
	}

	public Matrix getBiases() {
		return biases;
	}

	public Matrix getError() {
		return getError(0);
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

	public LinkedLayer getNext() {
		return next;
	}

	public LinkedLayer getPrevious() {
		return previous;
	}

	public Matrix getWeight() {
		return weight;
	}

	public LinkedLayer inputing(final LinkedLayer nextLayer) {
		nextLayer.addInput(input);
		return nextLayer;
	}

	public LinkedLayer next(final LinkedLayer next) {
		this.next = next;
		this.next.previous = this;
		next.weight = new Matrix(neuronNumber, next.neuronNumber).initRandom();
		next.biases = new Column(next.neuronNumber).initRandom();
		return next;
	}

	public LinkedLayer propagateError(final LinkedLayer layer) {
		return layer;
	}

	public void setBiases(final Matrix biases) {
		this.biases = biases;
	}

	public void setErrors(final List<Matrix> errors) {
		this.errors = errors;
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
