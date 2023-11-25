package ml.classifiers;

import ml.DataSet;
import ml.Example;

public class ClassifierTimer {
	/**
	 * Calculates the time to train and test the classifier averaged over numRuns on
	 * 80/20 splits of the data
	 * 
	 * @param classifier
	 * @param dataset
	 */
	public static void timeClassifier(Classifier classifier, DataSet dataset, int numRuns) {
		long trainSum = 0;
		long classifySum = 0;

		for (int i = 0; i < numRuns; i++) {
			DataSet[] temp = dataset.split(0.8);
			DataSet train = temp[0];
			DataSet test = temp[1];

			System.gc();
			long start = System.currentTimeMillis();
			classifier.train(train);
			trainSum += System.currentTimeMillis() - start;

			System.gc();
			start = System.currentTimeMillis();
			classifyExamples(classifier, test);
			classifySum += System.currentTimeMillis() - start;
		}

		System.out.println("Average train time: " + ((double) trainSum) / numRuns / 1000 + "s");
		System.out.println("Average test time: " + ((double) classifySum) / numRuns / 1000 + "s");
	}

	/**
	 * Classify all of the examples with the classifier. We don't care about the
	 * results
	 * just that the classify function gets called for all of the examples.
	 * 
	 * @param classifier
	 * @param dataset
	 */
	private static void classifyExamples(Classifier classifier, DataSet dataset) {
		for (Example e : dataset.getData()) {
			classifier.classify(e);
		}
	}

	public static void main(String[] args) {
		String csvFile = "data/titanic-train.csv";
		DataSet dataset = new DataSet(csvFile);

		int numRuns = 10;

		/*
		 * Copy your decision tree code from last time to check the timing. If you
		 * didn't get your code working,
		 * come talk to me.
		 * 
		 * System.out.println("------------------------");
		 * System.out.println("Decision Tree:");
		 * DecisionTreeClassifier dt = new DecisionTreeClassifier();
		 * timeClassifier(dt, dataset, numRuns);
		 */

		System.out.println("------------------------");
		System.out.println("Perceptron:");
		PerceptronClassifier original = new PerceptronClassifier();
		timeClassifier(original, dataset, numRuns);

		System.out.println("------------------------");
		System.out.println("Average Perceptron:");
		AveragePerceptronClassifier weighted = new AveragePerceptronClassifier();
		timeClassifier(weighted, dataset, numRuns);

		System.out.println("------------------------");
		System.out.println("Decision Tree:");
		DecisionTreeClassifier DT = new DecisionTreeClassifier();
		timeClassifier(DT, dataset, numRuns);
	}
}
