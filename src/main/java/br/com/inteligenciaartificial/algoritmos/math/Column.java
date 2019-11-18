package br.com.inteligenciaartificial.algoritmos.math;

public class Column extends Matrix {

    public Column(final double[] elements) {
        for (int i = 0; i < elements.length; i++) {
            addRow(new double[] {elements[i]});
        }
    }

    public Column(final int size) {
        super(size, 1);
    }

    public Column set(final int rowIndex, final double value) {
        super.set(rowIndex, 0, value);
        return this;
    }

    @Override
    public Column copy() {
        final double[] copy = new double[getRowNum()];
        for (int i = 0; i < copy.length; i++) {
            copy[i] = get(i, 0);
        }
        return new Column(copy);
    }

}
