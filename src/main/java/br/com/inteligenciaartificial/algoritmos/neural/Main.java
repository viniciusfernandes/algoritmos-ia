package br.com.inteligenciaartificial.algoritmos.neural;

import java.io.IOException;
import java.util.Date;
import java.util.List;

public class Main {

    public static void main(final String[] args) throws IOException {
        final List<TrainingDigit> trainingData =
                        new MnistDataReader().readData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");

        final DigitClassifier classifier = new DigitClassifier(0.08, 1);
        System.out.println("-------------------------");
        System.out.println("LEARNING IS JUST BEGINNIG");
        System.out.println("-------------------------");

        final Date init = new Date();

        classifier.learn(trainingData);

        final Date end = new Date();

        System.out.println(String.format("LEARNING IS COMPLETED in %d secs.", (end.getTime() - init.getTime()) / 1000));
        System.out.println("-------------------------");

        TrainingDigit data = null;

        for (int i = 0; i < 10; i++) {
            data = trainingData.get(i);
            System.out.println(
                            String.format("The digit is %d, and was classified as %d", data.getExpectedDigit(), classifier.classify(data)));
        }
    }
}
