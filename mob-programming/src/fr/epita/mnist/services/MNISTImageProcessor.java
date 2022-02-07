package fr.epita.mnist.services;

import java.util.List;

import fr.epita.mnist.datamodel.MNISTImage;

public class MNISTImageProcessor {

    public MNISTImage computeCentroid(Double label, List<MNISTImage> images) {
        MNISTImage centroid = new MNISTImage(label, new double[28][28]);
        int size = images.size();
        for (MNISTImage image : images) {
            double[][] pixels = image.getPixels();
            for (int i = 0; i < pixels.length; i++) {
                for (int j = 0; j < pixels[i].length; j++) {
                    centroid.getPixels()[i][j] = centroid.getPixels()[i][j] + pixels[i][j] / size;
                }
            }
        }
        return centroid;
    }

    public double computeDistance(MNISTImage image1, MNISTImage image2){
        double distance = 0.0;

        double[][] image1Pixels = image1.getPixels();
        double[][] image2Pixels = image2.getPixels();

        for (int i = 0 ; i < MNISTReader.MAX_ROW; i++){
            for (int j = 0 ; j < MNISTReader.MAX_COL; j++){
                 distance = distance + Math.sqrt(Math.pow(image2Pixels[i][j] - image1Pixels[i][j],2));
            }
        }

        return distance;
    }

    public static void displayImage(MNISTImage image) {
        double[][] pixels = image.getPixels();
        for (int i = 0; i < pixels.length; i++) {
            for (int j = 0; j < pixels[i].length; j++) {
                if (pixels[i][j] < 128) {
                    System.out.print("..");
                } else {
                    System.out.print("xx");
                }
            }
            System.out.println();
        }
    }

}
