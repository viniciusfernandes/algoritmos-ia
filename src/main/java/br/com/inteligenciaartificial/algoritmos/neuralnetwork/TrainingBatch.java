package br.com.inteligenciaartificial.algoritmos.neuralnetwork;

import java.util.ArrayList;
import java.util.List;

import br.com.inteligenciaartificial.algoritmos.neuralnetwork.digitrecognizer.Digit;

public class TrainingBatch {
    public final List<Digit> digits;

    public TrainingBatch(final int batchSize) {
        digits = new ArrayList<>(batchSize);
    }

    public void add(final Digit digit) {
        digits.add(digit);
    }
}
