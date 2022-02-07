package fr.epita.mnist.services;

import fr.epita.mnist.datamodel.MNISTImage;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class CentroidClassifier {
	 Map<Double, MNISTImage> centroids = new LinkedHashMap<>();
	 MNISTImageProcessor imageProcessor= new MNISTImageProcessor();
// training the dataset using centroid
	 public void train_model_using_trainset(Map<Double, List<MNISTImage>> dataset){

		 for (Map.Entry<Double, List<MNISTImage>> entry : dataset.entrySet()){
			 MNISTImage centroid = imageProcessor.computeCentroid(entry.getKey(), entry.getValue());
			 this.centroids.put(centroid.getLabel(), centroid);
		 }


	 }
// training the dataset using standard deviation
	public double train_model_using_trainset2(Map<Double, List<MNISTImage>> dataset){

		for (Map.Entry<Double, List<MNISTImage>> entry : dataset.entrySet()){
			MNISTImage centroid = new standarddeviation().computeCentroidUsingStandardDeviation(entry.getKey(), entry.getValue());
			this.centroids.put(centroid.getLabel(), centroid);
		}


		return 0.;
	}
	 public Double predict (MNISTImage images) {
		 double min_distance = Double.MAX_VALUE;
		 double flag = 0.0D;
		 MNISTImage val = null;
		 Double distance = 0.0;


		 for (Map.Entry<Double, MNISTImage> centroid:centroids.entrySet()){
			 distance = (new MNISTImageProcessor()).computeDistance(images, centroid.getValue());
			 if (distance < min_distance) {
				 min_distance = distance;
				 flag = centroid.getKey();
				 val = centroid.getValue();
			 }
		 }
		 return flag;
	 }
	 //confusion matrix
	public void confusionMatrix(Map<Double, List<MNISTImage>> images) {
		double accurate = 0.0D;
		int[][] mtx = new int[10][10];
		double to_predict = 0.0D;
		
		for (int x = 0; x < images.size(); x++) {
			for(int y = 0; y < ((List)images.get((double)x)).size(); y++) {
				to_predict = this.predict((MNISTImage)((List)images.get((double)x)).get(y));
				int element_index = (int)to_predict;
				mtx[x][element_index]++;
			}
		}
		double cases=0.0D;
		double crct_predict=0.0D;
		System.out.println("CONFUSION MATRIX ");
		
		for(int x = 0; x < 10; x++) {
			for(int y = 0; y < 10; y++) {
				System.out.print(mtx[x][y]+"\t");
				cases+=(double)mtx[x][y];
				if(x==y) {
					crct_predict+=(double)mtx[x][y];
				}
			}
			System.out.println();
		}
		accurate=crct_predict/cases;
		System.out.println("Accuracy : "+accurate+"%");
	}
	
	public double calculatedistance_centroid(List<MNISTImage> images) {
		double min_distance = Double.MAX_VALUE;
		double flag = 0.0;
		MNISTImage foo = null;
		double distance = 0.0;
		
		for(int i = 0; i < images.size(); i++) {
			for (Map.Entry<Double, MNISTImage> centroid:centroids.entrySet()){
				distance= (new MNISTImageProcessor()).computeDistance(images.get(i), centroid.getValue());
				if(distance<min_distance) {
					min_distance=distance;
					flag=centroid.getKey();
					foo=centroid.getValue();
				}
			}
		}

		System.out.println("Minumum Distance "+min_distance+" label "+ flag);
		return flag;
	}



}
