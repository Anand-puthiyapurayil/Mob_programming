package fr.epita.mnist;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import fr.epita.mnist.datamodel.MNISTImage;
import fr.epita.mnist.services.MNISTReader;
import static fr.epita.mnist.services.MNISTImageProcessor.displayImage;

public class TestMNISTReader {


    public static void main(String[] args) throws Exception {
        MNISTReader reader = new MNISTReader();
        List<MNISTImage> images = reader.readImages(new File("./mob-programming/mnist_train.csv"), 100);


        Map<Double, List<MNISTImage>> imagesByLabel = images.stream().collect(Collectors.groupingBy(MNISTImage::getLabel));


        List<MNISTImage> listOfOnes = imagesByLabel.get(1.0);

        if ( !(images.get(1).getLabel() == 0))
            throw new Exception("verification exception, expected 7 and got: " + images.get(1).getLabel());
        ;

        displayImage(images.get(0));
    }



}
