package br.com.inteligenciaartificial.algoritmos.neural;

import java.io.IOException;
import java.util.Date;
import java.util.List;

public class Main {
    private static final String BAR = "-------------------------";

    public static void main(final String[] args) throws IOException {
        final List<TrainingDigit> trainingData =
                        new MnistDataReader().readData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");

        final DigitClassifier classifier = new DigitClassifier();
        System.out.println(BAR);
        System.out.println("LEARNING IS JUST BEGINNIG");
        System.out.println(BAR);

        final Date init = new Date();

        classifier.training(trainingData);

        final Date end = new Date();

        System.out.println(String.format("LEARNING IS COMPLETED in %d secs.", (end.getTime() - init.getTime()) / 1000));
        System.out.println(BAR);

        TrainingDigit data = null;

        int result = -1;
        int matches = 0;
        final int total = 10000;
        for (int i = 0; i < total; i++) {
            data = trainingData.get(i);
            System.out.println(String.format("The digit is %d, and was classified as %d", data.getValue(),
                            result = classifier.classify(data)));
            if (result == data.getValue()) {
                matches++;
            }

        }
        System.out.println(BAR);
        System.out.println(String.format("Total matching %d from %d", matches, total));
        System.out.println(BAR);

    }
}
