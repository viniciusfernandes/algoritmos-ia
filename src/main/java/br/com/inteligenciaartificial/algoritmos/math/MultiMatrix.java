package br.com.inteligenciaartificial.algoritmos.math;

public class MultiMatrix {
	private final Matrix[] elements;

	public MultiMatrix(final Matrix... elements) {
		this.elements = elements;
	}

	public Matrix multiply(final int index, final Matrix other) {
		return elements[index].multiply(other);
	}

}
