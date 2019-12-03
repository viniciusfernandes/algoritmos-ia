package br.com.inteligenciaartificial.algoritmos.neural;

import java.util.function.Function;

public class NeuralNetwork {

    private final Function<Double, Double> activationFunction;
    private final Function<Double, Double> activationDerivaticeFunction;

    public NeuralNetwork(final Function<Double, Double> activationFunction, final Function<Double, Double> activationDerivaticeFunction) {
        this.activationFunction = activationFunction;
        this.activationDerivaticeFunction = activationDerivaticeFunction;
    }

}
