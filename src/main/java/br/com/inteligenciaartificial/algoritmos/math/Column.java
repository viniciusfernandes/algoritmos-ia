package br.com.inteligenciaartificial.algoritmos.math;

public class Column extends Matrix {

	public Column(final double[] elements) {
		super(elements.length, 1);
		for (int i = 0; i < elements.length; i++) {
			set(i, elements[i]);
		}
	}

	public Column(final int size) {
		super(size, 1);
	}

	public double get(final int rowIndex) {
		return super.get(rowIndex, 0);
	}

	public Column set(final int rowIndex, final double value) {
		super.set(rowIndex, 0, value);
		return this;
	}
}
