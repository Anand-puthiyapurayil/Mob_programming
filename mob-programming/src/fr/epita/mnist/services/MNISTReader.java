package fr.epita.mnist.services;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import fr.epita.mnist.datamodel.MNISTImage;

public class MNISTReader {
// double[][]  =>
//    [
//     [0,2,3],
//     [1,2,3],
//     [4,5,6]
//    ]

    public static final int MAX_COL = 28;
    public static final int MAX_ROW = 28;


    public List<MNISTImage> readImages(File file, int maxLines) throws FileNotFoundException {
        Scanner scanner = new Scanner(file);
        int counter = 0;

        List<MNISTImage> images = new ArrayList<>();
        scanner.nextLine(); //skip the first line to avoid reading headers
        while (scanner.hasNext() &&  maxLines > 0 && counter < maxLines) {
            String line = scanner.nextLine();
            double[] lineAsDoubleArray = loadLine(line); // read a line under the form of a double array
            double[][] pixels = new double[MAX_COL][MAX_ROW];

            for (int i = 0; i < MAX_ROW; i++) {
                for (int j = 0; j < MAX_COL; j++) {
                    pixels[i][j] = lineAsDoubleArray[i * MAX_ROW + j + 1];
                }
            }
            images.add(new MNISTImage(lineAsDoubleArray[0], pixels));
            counter++;
        }
        scanner.close();
        return images;
    }

    private static double[] loadLine(String sample) {
        String[] entries = sample.split(",");

        double[] entriesAsDouble = new double[entries.length];

        for (int i = 0; i < entriesAsDouble.length; i++){
            entriesAsDouble[i] = Double.parseDouble(entries[i]);
        }
        return entriesAsDouble;
    }


}
