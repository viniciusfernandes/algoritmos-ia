package br.com.inteligenciaartificial.algoritmos.math;

import java.util.Arrays;

public class Matrix {

	public static void main(final String[] args) {
		final Matrix m1 = new Matrix(new double[] { 2, 3 }, new double[] { 4, 5 });
		final Matrix m2 = new Matrix(new double[] { 1, 2 }, new double[] { 3, 2 });

		m1.multiply(m2).print();
	}

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
		this.rowNum = rowNum;
		this.colNum = colNum;
	}

	Matrix addRow(final double[] row) {
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

	public Matrix copy() {
		return new Matrix(Arrays.copyOf(elements, elements.length));
	}

	public Matrix dot(final Matrix other) {
		if (other == null) {
			return null;
		}
		if (rowNum != other.rowNum || colNum != other.colNum) {
			throw new IllegalArgumentException(String.format(
					"You tried operate a matrix %s and another matrix %s, but both must have the same rom and column numbers",
					order(), other.order()));
		}
		final Matrix m = new Matrix(rowNum, colNum);
		for (int r = 0; r < elements.length; r++) {
			for (int c = 0; c < elements[r].length; c++) {
				m.set(r, c, elements[r][c] * other.elements[r][c]);
			}
		}
		return m;
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

	public Matrix initRandom() {
		for (int r = 0; r < rowNum; r++) {
			for (int c = 0; c < colNum; c++) {
				elements[r][c] = Math.random();
			}
		}
		return this;
	}

	public Matrix module() {
		return operate(Math::abs);
	}

	public Matrix multiply(final Matrix other) {
		if (colNum != other.rowNum) {
			throw new UnsupportedOperationException(
					"Operation is not defined. Matriz: " + order() + " and Matrix: " + other.order());
		}

		final Matrix m = new Matrix(rowNum, other.colNum);
		double sum = 0;
		for (int r = 0; r < elements.length; r++) {
			for (int oCol = 0; oCol < other.colNum; oCol++) {
				for (int oRow = 0; oRow < other.rowNum; oRow++) {
					sum += elements[r][oRow] * other.elements[oRow][oCol];
				}
				m.set(r, oCol, sum);
				sum = 0;
			}

		}
		return m;
	}

	public Matrix operate(final MatrixOperation operation) {
		final double[][] result = new double[rowNum][colNum];

		for (int r = 0; r < elements.length; r++) {
			for (int c = 0; c < elements[r].length; c++) {
				result[r][c] = operation.operate(elements[r][c]);
			}
		}
		return new Matrix(result);
	}

	public String order() {
		return rowNum + "X" + colNum;
	}

	public void print() {
		System.out.println(toString());
	}

	public Matrix set(final int rowIndex, final int colIndex, final double value) {
		elements[rowIndex][colIndex] = value;
		return this;
	}

	public Matrix sub(final Matrix other) {
		return sum(other, false);
	}

	public Matrix sum(final Matrix other) {
		return sum(other, true);
	}

	private Matrix sum(final Matrix other, final boolean isPlus) {
		if (other == null) {
			throw new UnsupportedOperationException("Sum operation is no defined to a null matrix");

		}

		if (rowNum != other.rowNum || colNum != other.colNum) {
			throw new UnsupportedOperationException(String.format(
					"Sum operation is no defined between matrix %s and another matrix %s", order(), other.order()));
		}

		final Matrix m = new Matrix(rowNum, colNum);
		for (int r = 0; r < rowNum; r++) {
			for (int c = 0; c < colNum; c++) {
				m.set(r, c, isPlus ? elements[r][c] + other.elements[r][c] : elements[r][c] - other.elements[r][c]);
			}
		}
		return m;
	}

	@Override
	public String toString() {
		final StringBuilder m = new StringBuilder("Matrix ").append(order()).append(": ");
		final StringBuilder space = new StringBuilder();

		for (int i = 0; i < m.length(); i++) {
			space.append(" ");
		}

		for (int r = 0; r < elements.length; r++) {

			for (int c = 0; c < elements[r].length; c++) {
				m.append(elements[r][c]);
				if (c < elements[r].length - 1) {
					m.append(" ");
				}
			}
			if (r < elements.length - 1) {
				m.append("\n");
				m.append(space);
			}

		}
		return m.toString();
	}

	public Matrix transpose() {
		final double[][] copy = new double[colNum][rowNum];
		for (int r = 0; r < rowNum; r++) {
			for (int c = 0; c < colNum; c++) {
				copy[c][r] = elements[r][c];
			}
		}

		return new Matrix(copy);
	}
}
