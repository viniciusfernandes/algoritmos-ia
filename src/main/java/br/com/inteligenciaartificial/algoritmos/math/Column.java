package br.com.inteligenciaartificial.algoritmos.math;

public class Column extends Matrix {

	public Column(final double[] elements) {
		super(elements.length, 1);
		for (int i = 0; i < elements.length; i++) {
			super.set(i, 0, elements[i]);
		}
	}

	public Column(final int size) {
		super(size, 1);
	}

	@Override
	public Column copy() {
		final double[] copy = new double[getRowNum()];
		for (int i = 0; i < copy.length; i++) {
			copy[i] = get(i, 0);
		}
		return new Column(copy);
	}

	public double get(final int rowIndex) {
		return super.get(rowIndex, 0);
	}

	public Column set(final int rowIndex, final double value) {
		super.set(rowIndex, 0, value);
		return this;
	}
}
