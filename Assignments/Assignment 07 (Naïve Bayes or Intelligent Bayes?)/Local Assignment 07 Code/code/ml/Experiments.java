package ml;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

import ml.classifiers.Classifier;
import ml.classifiers.NBClassifier;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;

public class Experiments {

    public static void main(String[] args) {
        // Load Wine DataSet
        DataSet wineData = new DataSet("data/wines.train", DataSet.TEXTFILE); // Find dataset

        // Initialize variables
        double optimalLambdaAllFeatures = 0.0;
        double optimalLambdaPosFeatures = 0.0;
        double bestAccuracyAllFeatures = 0.0;
        double bestAccuracyPosFeatures = 0.0;

        NBClassifier nb = new NBClassifier();
        DataSetSplit data = wineData.split(.1);
        nb.train(data.getTrain());

        // Experiment 1: Optimal Lambda with All Features
        nb.setUseOnlyPositiveFeatures(false);
        for (double lambda = 0.0001; lambda <= 1; lambda += 0.005) {
            nb.setLambda(lambda);

            double avgAccuracy = averageAccuracy(data, nb);

            if (avgAccuracy > bestAccuracyAllFeatures) {
                bestAccuracyAllFeatures = avgAccuracy;
                optimalLambdaAllFeatures = lambda;
            }
        }
        System.out.println("Experiment 1: Optimal Lambda with All Features is: " + optimalLambdaAllFeatures);

        // Experiment 2: Optimal Lambda with Only Positive Features
        nb.setUseOnlyPositiveFeatures(true);
        for (double lambda = 0.0001; lambda <= 1; lambda += 0.005) {
            nb.setLambda(lambda);

            double avgAccuracy = averageAccuracy(data, nb);

            if (avgAccuracy > bestAccuracyPosFeatures) {
                bestAccuracyPosFeatures = avgAccuracy;
                optimalLambdaPosFeatures = lambda;
            }
        }
        System.out.println("Experiment 2: Optimal Lambda with Only Positive Features is: " + optimalLambdaPosFeatures);

        // Experiment 3: Determine which is better
        String betterVersion;
        if (bestAccuracyAllFeatures > bestAccuracyPosFeatures) {
            betterVersion = "All Features";
        } else {
            betterVersion = "Only Positive Features";
        }

        System.out.println("Experiment 3: Best Version is with " + betterVersion);
        System.out.println("\n We came to this conclusion based on the data." +
                " Since it is text, it makes more sense to use positive only because we have such a large number of words"
                +
                " that are not in each example. Using All features will give us a very large number of negatives and also"
                +
                " take a really long time to iterate over every word in the set for each example.");

        // Experiment 4:
        ArrayList<double[]> conAccs = confidenceAccuracies(data, nb); // Graph this
        System.out.println(conAccs);
    }

    /**
     * Returns the average accuracy of the classifier on the test set
     * 
     * @param dataSplit Data split
     * @param clsf      Classifier
     * @return Average  accuracy
     */
    public static double averageAccuracy(DataSetSplit dataSplit, Classifier clsf) {
        ArrayList<Example> testData = dataSplit.getTest().getData();
        double averageAccuracy = 0.0;
        for (Example example : testData) {
            if (clsf.classify(example) == example.getLabel()) {
                averageAccuracy += 100.0;
            }
        }
        return averageAccuracy / testData.size();
    }

    /**
     * Returns a list of pairs (confidence, isCorrect) sorted by confidence
     * @param dataSplit the data split
     * @param clsf      the classifier
     * @return          a list of pairs (confidence, isCorrect) sorted by confidence
     */
    public static ArrayList<double[]> confidenceAccuracies(DataSetSplit dataSplit, NBClassifier clsf) {
        // Train the classifier
        clsf.train(dataSplit.getTrain());

        // List to store pairs (confidence, isCorrect)
        ArrayList<double[]> pairs = new ArrayList<>();

        // get label prediction and confidence
        for (Example example : dataSplit.getTest().getData()) {
            double predictedLabel = clsf.classify(example);
            double trueLabel = example.getLabel();
            double confidence = clsf.confidence(example);

            double[] pair = new double[2];
            if (predictedLabel == trueLabel) {
                pair[0] = confidence;
                pair[1] = 1.0;
            } else {
                pair[0] = confidence;
                pair[1] = 0.0;
            }
            pairs.add(pair);
        }

        // Sort by confidence
        Collections.sort(pairs, Comparator.comparingDouble(o -> -o[0]));

        return pairs;
    }
}
