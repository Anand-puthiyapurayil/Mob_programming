package fr.epita.mnist.Launcher;

import fr.epita.mnist.datamodel.MNISTImage;
import fr.epita.mnist.services.MNISTImageProcessor;
import fr.epita.mnist.services.MNISTReader;
import fr.epita.mnist.services.CentroidClassifier;
import fr.epita.mnist.services.standarddeviation;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;

public class Launcher {

    public static void main(String[] args) throws FileNotFoundException {
        MNISTImageProcessor processor = new MNISTImageProcessor();
        MNISTReader reader = new MNISTReader();

        List<MNISTImage> images = reader.readImages(new File("./mob-programming/mnist_train.csv"), 1000);
        Map<Double, List<MNISTImage>> imagesByLabel = images.stream().collect(Collectors.groupingBy(MNISTImage::getLabel));
        TreeMap<Double, List<MNISTImage>> imagesByLabelSORTED = new TreeMap<>(imagesByLabel);
        System.out.println(imagesByLabelSORTED);

        List<MNISTImage> test_images = reader.readImages(new File("./mob-programming/mnist_test.csv"), 1000);
        Map<Double, List<MNISTImage>> images_ByLabel = test_images.stream().collect(Collectors.groupingBy(MNISTImage::getLabel));
        TreeMap<Double, List<MNISTImage>> images_ByLabelSORTED = new TreeMap<>(images_ByLabel);
        System.out.println(images_ByLabelSORTED);

        List<MNISTImage> listOfOnes = imagesByLabel.get(7.0);
        List<MNISTImage> listOfZeros = imagesByLabel.get(0.0);
        MNISTImage centroidFor1 = processor.computeCentroid(1.0, listOfOnes);


        Map<Double, MNISTImage> centroids = new LinkedHashMap<>();


        for (Map.Entry<Double, List<MNISTImage>> entry : imagesByLabel.entrySet()) {
            MNISTImage centroid = processor.computeCentroid(entry.getKey(), entry.getValue());
            centroids.put(centroid.getLabel(), centroid);


        }

        Map<Double, MNISTImage> centroidss = new LinkedHashMap<>();
        List<MNISTImage> list_OfOnes = images_ByLabel.get(5.0);
        List<MNISTImage> list_OfZeros = images_ByLabel.get(0.0);

        MNISTImage centroidFor2 = processor.computeCentroid(1.0, list_OfOnes);

        for (Map.Entry<Double, List<MNISTImage>> entry : images_ByLabel.entrySet()) {
            MNISTImage centroid = processor.computeCentroid(entry.getKey(), entry.getValue());
            centroidss.put(centroid.getLabel(), centroid);
        }

        System.out.println("Distance\n");
        System.out.println(processor.computeDistance(listOfOnes.get(0), centroidFor1));
        System.out.println(processor.computeDistance(listOfZeros.get(0), centroidFor1));
        System.out.println("Image for train data set\n");
        MNISTImageProcessor.displayImage(centroidFor1);


        System.out.println("Distance\n");
        System.out.println(processor.computeDistance(list_OfOnes.get(0), centroidFor2));
        System.out.println(processor.computeDistance(list_OfZeros.get(0), centroidFor2));
        System.out.println("Image for test data set\n");
        MNISTImageProcessor.displayImage(centroidFor2);

        CentroidClassifier classifier = new CentroidClassifier();
        classifier.train_model_using_trainset(imagesByLabel);
        classifier.train_model_using_trainset(images_ByLabel);



        System.out.println("F. Performing classification using centroid\n");
        Double testLabel = 5.0;



// Task F
        List<MNISTImage> labellist = (imagesByLabelSORTED.get(testLabel)).stream().limit(10).collect(Collectors.toList());
       classifier.calculatedistance_centroid(labellist);


        List<MNISTImage> zerolist = (images_ByLabelSORTED.get(testLabel)).stream().limit(10).collect(Collectors.toList());
        classifier.calculatedistance_centroid(zerolist);


        Double predictionForLabel = classifier.predict((imagesByLabel.get(testLabel)).get(0));
       System.out.println("Predicted Label by calculating distance: " + predictionForLabel);

       Double prediction_ForLabel = classifier.predict((images_ByLabel.get(testLabel)).get(0));
        System.out.println("Predicted Label by calculating distance: " + prediction_ForLabel);

       System.out.println("\nH. Implement classification performance assessment : Centroid Using average distance");

       System.out.println("Using Train dataset");
       classifier.confusionMatrix(imagesByLabel);
       System.out.println("\nUsing Test dataset");
       classifier.confusionMatrix(images_ByLabel);

       // confusion matrix using standard deviation
        standarddeviation standardCalculation = new standarddeviation();
        System.out.println(" Performing classification using standard deviation\n");
        classifier.train_model_using_trainset2(imagesByLabel);


        List<MNISTImage> labellist2 = (imagesByLabelSORTED.get(testLabel)).stream().limit(10).collect(Collectors.toList());
        classifier.calculatedistance_centroid(labellist2);


        List<MNISTImage> zerolist2 = (images_ByLabelSORTED.get(testLabel)).stream().limit(10).collect(Collectors.toList());
        classifier.calculatedistance_centroid(zerolist2);
        standardCalculation.computeCentroidUsingStandardDeviation(0.,labellist2);
        standardCalculation.computeCentroidUsingStandardDeviation(0.,zerolist2);

        System.out.println("Using Train dataset");
        classifier.confusionMatrix(imagesByLabel);
        System.out.println("\nUsing Test dataset");
        classifier.confusionMatrix(images_ByLabel);


    }
}