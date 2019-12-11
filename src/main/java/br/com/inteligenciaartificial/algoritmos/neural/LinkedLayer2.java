package br.com.inteligenciaartificial.algoritmos.neural;

import java.util.ArrayList;
import java.util.List;
import java.util.function.UnaryOperator;

import br.com.inteligenciaartificial.algoritmos.math.Column;
import br.com.inteligenciaartificial.algoritmos.math.Matrix;

public class LinkedLayer2 {
	private final UnaryOperator<Matrix> activationFunction;
	private Matrix biases;

	private Matrix error;

	private List<Matrix> errors = new ArrayList<>(6000);

	private Matrix input;

	private final List<Matrix> inputs = new ArrayList<>(6000);

	private int layerNumber = 1;
	private LinkedLayer2 next;
	private final int numberOfNeurons;
	private Matrix output;

	private LinkedLayer2 previous;

	private Matrix weight;

	public LinkedLayer2(final int neuronNumber) {
		this(neuronNumber, m -> m);
	}

	public LinkedLayer2(final int neuronNumber, final UnaryOperator<Matrix> activationFunction) {
		this.activationFunction = activationFunction;
		numberOfNeurons = neuronNumber;
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
		if (weight == null || biases == null) {
			return input;
		}
		return weight.transpose().multiply(input).sub(biases);
	}

	public LinkedLayer2 estimulating(final LinkedLayer2 nextLayer) {
		nextLayer.addInput(estimulate(input));
		return nextLayer;
	}

	public void feedForward() {
		feedForward(this);
	}

	private void feedForward(final LinkedLayer2 layer) {
		if (layer == null || layer.next == null) {
			return;
		}
		layer.next.addInput(layer.activate());
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

	public LinkedLayer2 getNext() {
		return next;
	}

	public int getNumberOfNeurons() {
		return numberOfNeurons;
	}

	public LinkedLayer2 getPrevious() {
		return previous;
	}

	public Matrix getWeight() {
		return weight;
	}

	public Matrix initRandomBiases() {
		biases = new Column(numberOfNeurons).initRandom();
		return biases;
	}

	public Matrix initRandomWeights() {
		weight = new Matrix(previous.numberOfNeurons, numberOfNeurons).initRandom();
		return weight;
	}

	public LinkedLayer2 inputing(final LinkedLayer2 nextLayer) {
		nextLayer.addInput(input);
		return nextLayer;
	}

	public LinkedLayer2 next(final LinkedLayer2 nextLayer) {
		next = nextLayer;
		next.previous = this;
		next.layerNumber = layerNumber + 1;
		return nextLayer;
	}

	public LinkedLayer2 propagateError(final LinkedLayer2 layer) {
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

	@Override
	public String toString() {
		return "LinkedLayer [layerNumber=" + layerNumber + ", numberOfNeurons=" + numberOfNeurons + "]";
	}

	public Matrix weightedError() {
		return weight.multiply(error);
	}
}