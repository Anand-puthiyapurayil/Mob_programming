package fr.epita.mnist.services;

import fr.epita.mnist.datamodel.MNISTImage;

import java.util.List;
//standard deviation computation
public class standarddeviation {

    public MNISTImage computeCentroidUsingStandardDeviation(Double lbl, List<MNISTImage> image) {
        MNISTImage centroid = new MNISTImage(lbl, new double[MNISTImage.MAX_ROW][MNISTImage.MAX_COL]);
        int len = image.size();

        for (MNISTImage images : image) {
            double[][] pixel = images.getPixels();
            for (int x = 0; x < pixel.length; x++) {
                for (int y= 0; y < pixel[x].length; y++) {
                    centroid.getPixels()[x][y] += pixel[x][y]/(double)len;
                }
            }
            for (int x = 0; x < pixel.length; x++) {
                for (int y= 0; y < pixel[x].length; y++) {
                    centroid.getPixels()[x][y] += Math.pow(centroid.getPixels()[x][y]-pixel[x][y],2);
                    centroid.getPixels()[x][y] = Math.sqrt(centroid.getPixels()[x][y]/(double)len);
                }
            }
        }
        return centroid;

    }
}
