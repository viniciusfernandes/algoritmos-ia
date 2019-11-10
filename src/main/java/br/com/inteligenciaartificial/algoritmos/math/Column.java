package br.com.inteligenciaartificial.algoritmos.math;

public class Column extends Matrix {

    public Column(final double[] elements) {
        for (int i = 0; i < elements.length; i++) {
            addRow(new double[] {elements[i]});
        }
    }

}
