package br.com.inteligenciaartificial.algoritmos.math;

import java.util.function.DoubleConsumer;

public class MultiMatrix {
	private Matrix[] elements;

	public MultiMatrix() {

	}

	public MultiMatrix(final int size) {
		elements = new Matrix[size];
	}

	public MultiMatrix(final Matrix... elements) {
		this.elements = elements;
	}

	public MultiMatrix add(final Matrix matrix) {
		final Matrix[] copy = new Matrix[elements.length + 1];
		for (int i = 0; i < elements.length; i++) {
			copy[i] = elements[i];
		}
		copy[elements.length + 1] = matrix;
		elements = copy;
		return this;
	}

	public Matrix apply(final int index, final DoubleConsumer operation) {
		return elements[index].apply(operation);
	}

	public Matrix get(final int index) {
		return elements[index];

	}

	public MultiMatrix initRandom() {
		for (int i = 0; i < elements.length; i++) {
			if (elements[i] == null) {
				continue;
			}
			elements[i].initRandom();
		}
		return this;
	}

	public Matrix multiply(final int index, final Matrix other) {
		return elements[index].multiply(other);
	}

	public void print() {
		System.out.println(toString());
	}

	public MultiMatrix set(final int index, final int row, final int col, final double value) {
		elements[index].set(row, col, value);
		return this;
	}

	public MultiMatrix set(final int index, final Matrix matrix) {
		elements[index] = matrix;
		return this;
	}

	public int size() {
		return elements.length;
	}

	@Override
	public String toString() {
		final StringBuilder s = new StringBuilder();
		for (int i = 0; i < elements.length; i++) {
			s.append("Indice ").append(i).append(": ").append("\n").append(elements[i]);
			if (i < elements.length - 1) {
				s.append("\n\n");
			}
		}
		return s.toString();
	}
}
