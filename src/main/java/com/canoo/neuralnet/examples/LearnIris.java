package com.canoo.neuralnet.examples;

import com.canoo.neuralnet.NNMath;
import com.canoo.neuralnet.NeuralNet;
import com.canoo.neuralnet.NeuronLayer;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

/**
 * Created by fabian on 14.04.16.
 */
public class LearnIris {

    /**
     * The goal of this neural net example is to learn the add function by examples
     * category.g.
     * <p>
     * 0.5  0.45 -> 0.95
     *
     * @param args
     */
    public static void main(String args[]) throws IOException, URISyntaxException {

        // create hidden layer that has 4 neurons and 4 inputs per neuron
        NeuronLayer layer1 = new NeuronLayer(NeuronLayer.ActivationFunctionType.SIGMOID, 4, 4);

        // create output layer that has 3 neurons representing the prediction and 4 inputs for this neuron
        // (mapped from the previous hidden layer)
        NeuronLayer layer2 = new NeuronLayer(NeuronLayer.ActivationFunctionType.SIGMOID, 1, 4);

        NeuralNet net = new NeuralNet(layer1, layer2, 1.0);

        List<Plant> plants = readFile();

        int trainingSetSize = (int) Math.floor(plants.size() * 0.65);

        double[][] inputs = new double[trainingSetSize][4];
        double[][] outputs = new double[trainingSetSize][3];
        for (int i = 0; i < trainingSetSize; i++) {
            Plant plant = plants.get(i);
            inputs[i] = NNMath.normalize(new double[]{plant.a, plant.b, plant.c, plant.d});
            // inputs[i] = new double[]{plant.a, plant.b, plant.c, plant.d};
            if (plant.category == 0) {
                outputs[i] = new double[]{0.0};
            } else if (plant.category == 1) {
                outputs[i] = new double[]{0.5};
            } else {
                outputs[i] = new double[]{1.0};
            }

        }

        System.out.println("Training the neural net...");
        net.train(inputs, outputs, 500);
        System.out.println("Finished training");

        System.out.println("Layer 1 weights");
        System.out.println(layer1);

        System.out.println("Layer 2 weights");
        System.out.println(layer2);

        // calculate the predictions on unknown data
        int successful = 0;
        for (int j = trainingSetSize; j < plants.size(); j++) {
            Plant plant = plants.get(j);
            // double[][] testInput = {NNMath.normalize(new double[]{plant.a, plant.b, plant.c, plant.d})};
            double[][] testInput = {new double[]{plant.a, plant.b, plant.c, plant.d}};
            boolean success = predict(testInput, plant.category, net, 0.1);
            if (success) {
                successful++;
            }
        }

        int testSetSize = plants.size() - trainingSetSize;

        System.out.println("Correctly predicted " + successful + " out of " + testSetSize);
        double accuracy = ((double) successful / (double) testSetSize) * 100;
        System.out.println("Accuracy: " + (int) accuracy + " %");

    }

    private static List<Plant> readFile() throws URISyntaxException, IOException {
        List<Plant> plants = new ArrayList<>();

        try (Stream<String> stream = Files.lines(Paths.get(LearnIris.class.getResource("irisShuffle.txt").toURI()))) {
            stream.forEach(l -> {
                String[] parts = l.split(",");
                Plant plant = new Plant(Double.parseDouble(parts[0]), Double.parseDouble(parts[1]), Double.parseDouble(parts[2]), Double.parseDouble(parts[3]), Integer.parseInt(parts[4]));
                plants.add(plant);
            });

        }

        return plants;
    }

    public static boolean predict(double[][] testInput, double expected, NeuralNet net, double errorMargin) {
        net.think(testInput);

        // then
        double expectedValue = expected / 2;
        double predictedValue = net.getOutput()[0][0];

        System.out.println("Prediction on data "
                + format(testInput[0][0]) + " "
                + format(testInput[0][1]) + " "
                + format(testInput[0][2]) + " "
                + format(testInput[0][3]) + ": "
                + format(predictedValue) + ", expected -> " + expectedValue + " ");

        return expectedValue - errorMargin < predictedValue && predictedValue < expectedValue + errorMargin;
    }

    private static String format(double x) {
        return String.format("%.03f", x);

    }

    private static class Plant {
        double a;
        double b;
        double c;
        double d;
        int category;


        public Plant(double a, double b, double c, double d, int category) {
            this.a = a;
            this.b = b;
            this.c = c;
            this.d = d;
            this.category = category;
        }
    }
}
