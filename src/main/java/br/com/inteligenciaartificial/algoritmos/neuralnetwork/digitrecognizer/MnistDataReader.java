package br.com.inteligenciaartificial.algoritmos.neuralnetwork.digitrecognizer;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MnistDataReader {

  public List<TrainingDigit> readData(final String dataFilePath, final String labelFilePath) throws IOException {

    final DataInputStream dataStream = new DataInputStream(new BufferedInputStream(new FileInputStream(dataFilePath)));
    final int magicNumber = dataStream.readInt();
    final int numberOfItems = dataStream.readInt();
    final int nRows = dataStream.readInt();
    final int nCols = dataStream.readInt();

    final int totPixels = nRows * nCols;

    System.out.println("magic number is " + magicNumber);
    System.out.println("number of items is " + numberOfItems);
    System.out.println("number of rows is: " + nRows);
    System.out.println("number of cols is: " + nCols);

    final DataInputStream labelStream = new DataInputStream(new BufferedInputStream(new FileInputStream(labelFilePath)));
    final int labelMagicNumber = labelStream.readInt();
    final int numberOfLabels = labelStream.readInt();

    System.out.println("labels magic number is: " + labelMagicNumber);
    System.out.println("number of labels is: " + numberOfLabels);

    assert numberOfItems == numberOfLabels;
    final List<TrainingDigit> dataList = new ArrayList<>(numberOfItems);
    double[] pixels = null;

    for (int i = 0; i < numberOfItems; i++) {
      pixels = new double[totPixels];
      for (int r = 0; r < totPixels; r++) {
        pixels[r] = dataStream.readUnsignedByte();
      }
      dataList.add(new TrainingDigit(pixels, labelStream.readUnsignedByte()));
    }
    dataStream.close();
    labelStream.close();
    return dataList;
  }
}
