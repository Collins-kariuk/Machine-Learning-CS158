// Munir Vafai, Collins Kariuki
package ml;

import ml.classifiers.AveragePerceptronClassifier;
import ml.classifiers.KNNClassifier;
import ml.classifiers.Classifier;
import ml.data.CrossValidationSet;
import ml.data.FeatureNormalizer;
import ml.data.ExampleNormalizer;
import ml.data.DataSet;
import ml.data.Example;
import ml.data.DataSetSplit;

public class Experiments {
    public static final int NUM_FOLDS = 10;

    public double[] runTests(DataSet data, int reps, Classifier clsf, String processingType) {
        CrossValidationSet newSet = data.getCrossValidationSet(NUM_FOLDS);
        double[] retval = new double[NUM_FOLDS];

        for (int j = 0; j < NUM_FOLDS; j++) {
            DataSetSplit split = newSet.getValidationSet(j, true);
            DataSet train = split.getTrain();
            DataSet test = split.getTest();
            ExampleNormalizer exampleNormalizer = new ExampleNormalizer();
            FeatureNormalizer featureNormalizer = new FeatureNormalizer();

            switch (processingType) {
                case "none":
                    break;
                case "length":
                    exampleNormalizer.preprocessTrain(train);
                    exampleNormalizer.preprocessTest(test);
                    break;
                case "feature":
                    featureNormalizer.preprocessTrain(train);
                    featureNormalizer.preprocessTest(test);
                    break;
                case "both":
                    featureNormalizer.preprocessTrain(train);
                    featureNormalizer.preprocessTest(test);
                    exampleNormalizer.preprocessTrain(train);
                    exampleNormalizer.preprocessTest(test);
                    break;
                default:
                    break;
            }

            double averageAcc = 0.0;
            clsf.train(train);
            for (int i = 0; i < reps; i++) {
                double numCorrect = 0.0;
                for (Example example : test.getData()) {
                    if (clsf.classify(example) == example.getLabel()) {
                        numCorrect += 1;
                    }
                }
                averageAcc += numCorrect / test.getData().size();
            }
            retval[j] = averageAcc / reps;
        }
        return retval;
    }

    private void printResults(double[] results, int numIterations) {
        double average = 0;
        for (int i = 0; i < 10; i++) {
            System.out.println(results[i]);
            average += results[i];
        }
        System.out.println(average / 10 + "\n");
        // System.out.println("Average accuracy for " + numIterations + " iterations: " + average / 10);
    }

    public static void main(String[] args) {
        Experiments exp = new Experiments();
        DataSet data1 = new DataSet("data/titanic-train.csv");
        DataSet data2 = new DataSet("data/titanic-train.real.csv");
        AveragePerceptronClassifier apc = new AveragePerceptronClassifier();
        int numIterations = NUM_FOLDS;
        apc.setIterations(numIterations);
        int reps = 10;
        KNNClassifier knn = new KNNClassifier();

        // // experiment 1
        // double[] results = exp.runTests(data1, reps, apc, "none");
        // exp.printResults(results, numIterations);

        // // experiment 2
        // double[] results2 = exp.runTests(data2, reps, apc, "none");
        // exp.printResults(results2, numIterations);

        // // experiment 3
        // double[] results3 = exp.runTests(data1, reps, knn, "none");
        // exp.printResults(results3, numIterations);
        // double[] results4 = exp.runTests(data2, reps, knn, "none");
        // exp.printResults(results4, numIterations);

        // experiment 4
        // k-NN with length normalization
        double[] results5 = exp.runTests(data2, reps, knn, "length");
        exp.printResults(results5, numIterations);
        // k-NN with feature normalization
        double[] results6 = exp.runTests(data2, reps, knn, "feature");
        exp.printResults(results6, numIterations);
        // k-NN with length and feature normalization
        double[] results7 = exp.runTests(data2, reps, knn, "both");
        exp.printResults(results7, numIterations);
        // perceptron with length normalization
        double[] results8 = exp.runTests(data2, reps, apc, "length");
        exp.printResults(results8, numIterations);
        // perceptron with feature normalization
        double[] results9 = exp.runTests(data2, reps, apc, "feature");
        exp.printResults(results9, numIterations);
        // perceptron with length and feature normalization
        double[] results10 = exp.runTests(data2, reps, apc, "both");
        exp.printResults(results10, numIterations);
    }
}
