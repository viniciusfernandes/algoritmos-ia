package br.com.inteligenciaartificial.algoritmos.math;

public class Matrix {

	private int colNum = -1;
	private double[][] elements;
	private int rowNum = -1;

	public Matrix() {
	}

	public Matrix(final double[]... rows) {
		rowNum = rows.length;
		colNum = rows[0].length;
		elements = new double[rowNum][];

		for (int i = 0; i < rowNum; i++) {
			if (rows[i].length != colNum) {
				throw new IllegalArgumentException("All rows must have the same length");
			}
			elements[i] = rows[i];
		}
	}

	public Matrix(final int rowNum, final int colNum) {
		elements = new double[rowNum][colNum];
	}

	public Matrix add(final int rowIndex, final int colIndex, final double value) {
		elements[rowIndex][colIndex] = value;
		return this;
	}

	public Matrix addRow(final double[] row) {
		if (rowNum < 0) {
			rowNum = 1;
			colNum = row.length;
			elements = new double[1][];
			elements[0] = row;
			return this;
		}

		if (row.length != colNum) {
			throw new IllegalArgumentException("All rows must have the same length");
		} else {
			rowNum++;
			final double[][] copy = new double[rowNum][];
			for (int r = 0; r < elements.length; r++) {
				copy[r] = elements[r];
			}
			copy[rowNum - 1] = row;
			elements = copy;
		}
		return this;
	}

	public double get(final int rowIndex, final int colIndex) {
		return elements[rowIndex][colIndex];
	}

	public int getColNum() {
		return colNum;
	}

	public int getRowNum() {
		return rowNum;
	}

	public Matrix multiply(final Column col) {
		if (colNum != col.getRowNum()) {
			throw new UnsupportedOperationException("Operation is no defined");
		}

		final Matrix m = new Matrix(rowNum, 1);

		for (int r = 0; r < elements.length; r++) {
			for (int c = 0; c < elements.length; c++) {

			}
		}
		return null;
	}

	public Matrix multiply(final Matrix matrix) {
		if (colNum != matrix.rowNum) {
			throw new UnsupportedOperationException("Operation is no defined");
		}

		final Matrix m = new Matrix(rowNum, matrix.colNum);
		double sum = 0;
		for (int r = 0; r < elements.length; r++) {
			for (int c = 0; c < elements[r].length; c++) {
				sum += elements[r][c] * matrix.elements[c][r];
			}

		}
		return null;
	}

	public Matrix multiply(final Row row) {

		return null;
	}

	public void print() {
		final StringBuilder m = new StringBuilder();
		for (int r = 0; r < elements.length; r++) {
			for (int c = 0; c < elements[r].length; c++) {
				m.append(elements[r][c]).append(" ");
			}
			m.append("\n");
		}
		System.out.println(m.toString());
	}

}
