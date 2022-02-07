package fr.epita.mnist;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import fr.epita.mnist.datamodel.MNISTImage;
import fr.epita.mnist.services.MNISTImageProcessor;
import fr.epita.mnist.services.MNISTReader;

public class TestMNISTImageProcessor {


    public static void main(String[] args) throws FileNotFoundException {
        MNISTImageProcessor processor = new MNISTImageProcessor();
        MNISTReader reader = new MNISTReader();
        List<MNISTImage> images = reader.readImages(new File("./mob-programming/mnist_train.csv"), 100);

        Map<Double, List<MNISTImage>> imagesByLabel = images.stream().collect(Collectors.groupingBy(MNISTImage::getLabel));

        List<MNISTImage> listOfOnes = imagesByLabel.get(1.0);
        System.out.println(imagesByLabel.get(1.0));

        List<MNISTImage> listOfZeros = imagesByLabel.get(0.0);
        MNISTImage centroidFor1 = processor.computeCentroid(1.0, listOfOnes);

        Map<Double, MNISTImage> centroids = new LinkedHashMap<>();

        for (Map.Entry<Double, List<MNISTImage>> entry : imagesByLabel.entrySet()){
            MNISTImage centroid = processor.computeCentroid(entry.getKey(), entry.getValue());
            centroids.put(centroid.getLabel(), centroid);
        }

        System.out.println(processor.computeDistance(listOfOnes.get(0), centroidFor1));
        System.out.println(processor.computeDistance(listOfZeros.get(0), centroidFor1));


        MNISTImageProcessor.displayImage(centroidFor1);






    }
}
