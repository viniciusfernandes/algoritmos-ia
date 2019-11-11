package br.com.inteligenciaartificial.algoritmos.math;

public class Row extends Matrix {

    public Row(final double[] elements) {
        super(elements);
    }

    public Row(final int size) {
        addRow(new double[size]);
    }
}
