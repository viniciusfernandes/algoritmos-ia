package br.com.inteligenciaartificial.algoritmos.math;

import java.util.Arrays;
import java.util.function.DoublePredicate;
import java.util.function.DoubleUnaryOperator;

public class Matrix {
	private int colNum = -1;
	private double[][] elements;
	private int rowNum = -1;
	private Matrix transpose = null;

	public Matrix() {
	}

	@Deprecated
	public Matrix(final double[]... rows) {
		rowNum = rows.length;
		colNum = rows[0].length;
		elements = new double[rowNum][];

		transpose.rowNum = colNum;
		transpose.colNum = rowNum;

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

		transpose = new Matrix();
		transpose.elements = new double[colNum][rowNum];

		transpose.rowNum = colNum;
		transpose.colNum = rowNum;
	}

	@Deprecated
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

	public Matrix apply(final DoubleUnaryOperator operation) {
		final Matrix m = new Matrix(rowNum, colNum);
		for (int r = 0; r < elements.length; r++) {
			for (int c = 0; c < elements[r].length; c++) {
				m.set(r, c, operation.applyAsDouble(elements[r][c]));
			}
		}
		return m;
	}

	@Deprecated
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

	@Override
	public boolean equals(final Object other) {
		if (other == null || !(other instanceof Matrix)) {
			return false;
		}

		final Matrix m = (Matrix) other;
		if (elements == null || m.elements == null || rowNum != m.rowNum || colNum != m.colNum) {
			return false;
		}

		for (int i = 0; i < rowNum; i++) {
			for (int j = 0; j < colNum; j++) {
				if (elements[i][j] != m.elements[i][j]) {
					return false;
				}
			}
		}

		return true;
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
				set(r, c, Math.random());
			}
		}
		return this;
	}

	public Matrix module() {
		return apply(Math::abs);
	}

	public Matrix multiply(final Matrix other) {
		if (colNum != other.rowNum) {
			throw new UnsupportedOperationException(
					"Operation is not defined. Matriz: " + order() + " and Matrix: " + other.order());
		}

		final Matrix m = new Matrix(rowNum, other.colNum);
		double sum = 0;
		for (int row = 0; row < elements.length; row++) {

			for (int oCol = 0; oCol < other.colNum; oCol++) {
				for (int oRow = 0; oRow < other.rowNum; oRow++) {
					sum += elements[row][oRow] * other.elements[oRow][oCol];
				}
				m.set(row, oCol, sum);
				sum = 0;
			}

		}
		return m;
	}

	public String order() {
		return rowNum + "X" + colNum;
	}

	public void print() {
		System.out.println(toString());
	}

	public Matrix set(final int rowIndex, final int colIndex, final double value) {

		elements[rowIndex][colIndex] = value;
		transpose.elements[colIndex][rowIndex] = value;
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

	public boolean test(final DoublePredicate validation) {
		for (int r = 0; r < elements.length; r++) {
			for (int c = 0; c < elements[r].length; c++) {
				if (!validation.test(elements[r][c])) {
					return false;
				}
			}
		}
		return true;
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
		if (transpose != null) {
			return transpose;
		}

		transpose = new Matrix(colNum, rowNum);
		for (int r = 0; r < rowNum; r++) {
			for (int c = 0; c < colNum; c++) {
				transpose.elements[c][r] = elements[r][c];
			}
		}

		return transpose;
	}
}
