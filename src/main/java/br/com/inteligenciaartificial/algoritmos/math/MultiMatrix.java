package br.com.inteligenciaartificial.algoritmos.math;

public class MultiMatrix {
	private final Matrix[] elements;
	private final int size;

	public MultiMatrix(final int size) {
		this.size = size;
		elements = new Matrix[size];
	}

	public MultiMatrix(final Matrix... elements) {
		this.elements = elements;
		size = elements.length;
	}

	public MultiMatrix multiply(final int index, final Matrix other) {
		return new MultiMatrix(elements[index].multiply(other));
	}
}
