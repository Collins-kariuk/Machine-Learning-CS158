// Munir Vafai
// Collins Kariuki
package ml;

import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import ml.classifiers.AVAClassifier;
import ml.classifiers.Classifier;
import ml.classifiers.ClassifierFactory;
import ml.classifiers.DecisionTreeClassifier;
import ml.classifiers.OVAClassifier;
import ml.classifiers.RandomClassifier;
import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;

public class Experimenter {
    private final static int NUM_FOLDS = 10;

    /**
     * Used for question 1
     * @param dataset The dataset to use
     */
    public static void q1(DataSet dataset) {
        final int NUM_REPS = 50;

        DecisionTreeClassifier dtc = new DecisionTreeClassifier();
        dtc.setDepthLimit(5);

        CrossValidationSet newSet = dataset.getRandomCrossValidationSet(5);
        DataSetSplit split = newSet.getValidationSet(0);
        DataSet train = split.getTrain();
        DataSet test = split.getTest();

        for (int i = 0; i <= NUM_REPS; i++) {
            dtc.setDepthLimit(i);
            dtc.train(train);

            // get training accuracy
            double correctTrainPredictions = 0.0;
            for (Example e : train.getData()) {
                double result = dtc.classify(e);
                if (result == e.getLabel()) {
                    correctTrainPredictions++;
                }
            }
            double curTrainAccuracy = correctTrainPredictions / train.getData().size();
            System.out.println("Training accuracy for depth " + i + ": " + curTrainAccuracy);

            // get test accuracy
            double correctTestPredictions = 0.0;
            for (Example e : test.getData()) {
                double result = dtc.classify(e);
                if (result == e.getLabel()) {
                    correctTestPredictions++;
                }
            }
            double curTestAccuracy = correctTestPredictions / test.getData().size();
            System.out.println("Test accuracy for depth " + i + ": " + curTestAccuracy);
        }
    }

    /**
     * Run 10-fold cross validation on the given dataset using the given classifier
     * @param newSet The dataset to use
     * @param reps The number of times to run the test
     * @param clsf The classifier to use
     * @return An array of the accuracies for each fold
     */
    public static double[] runTests(CrossValidationSet newSet, int reps, Classifier clsf) {
        double[] retval = new double[NUM_FOLDS];

        for (int j = 0; j < NUM_FOLDS; j++) {
            DataSetSplit split = newSet.getValidationSet(j);
            DataSet train = split.getTrain();
            DataSet test = split.getTest();

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
            System.out.println("Fold " + j + " accuracy: " + retval[j]);
        }
        return retval;
    }

    private static void printResults(double[] results, int numIterations) {
        double average = 0;
        for (int i = 0; i < 10; i++) {
            System.out.println(results[i]);
            average += results[i];
        }
        System.out.println(average / 10 + "\n");
        // System.out.println("Average accuracy for " + numIterations + " iterations: "
        // + average / 10);
    }

    public static void main(String[] args) {
        DataSet wineDataset = new DataSet("data/wines.train", DataSet.TEXTFILE);
        DecisionTreeClassifier dtc = new DecisionTreeClassifier();
        // QUESTION 1-3
        // dtc.setDepthLimit(5);
        // dtc.train(wineDataset);
        // System.out.println(dtc.toString());
        // q1(wineDataset);

        // QUESTION 4
        CrossValidationSet data = wineDataset.getCrossValidationSet(NUM_FOLDS);
        int reps = 1;
        // OVAClassifier ova1 = new OVAClassifier(new
        // ClassifierFactory(ClassifierFactory.DECISION_TREE, 1));
        // double[] results1 = runTests(data, reps, ova1);
        // printResults(results1, reps);

        // OVAClassifier ova2 = new OVAClassifier(new
        // ClassifierFactory(ClassifierFactory.DECISION_TREE,2));
        // double[] results2 = runTests(data, reps, ova2);
        // printResults(results2, reps);

        // OVAClassifier ova3 = new OVAClassifier(new
        // ClassifierFactory(ClassifierFactory.DECISION_TREE,3));
        // double[] results3 = runTests(data, reps, ova3);
        // printResults(results3, reps);

        // AVAClassifier ava1 = new AVAClassifier(new ClassifierFactory(ClassifierFactory.DECISION_TREE, 1));
        // double[] results4 = runTests(data, reps, ava1);
        // printResults(results4, reps);

        AVAClassifier ava2 = new AVAClassifier(new ClassifierFactory(ClassifierFactory.DECISION_TREE, 2));
        double[] results5 = runTests(data, reps, ava2);
        printResults(results5, reps);

        AVAClassifier ava3 = new AVAClassifier(new ClassifierFactory(ClassifierFactory.DECISION_TREE, 3));
        double[] results6 = runTests(data, reps, ava3);
        printResults(results6, reps);

        // DecisionTreeClassifier dtc1 = new DecisionTreeClassifier();
        // dtc1.setDepthLimit(3);
        // double[] results7 = runTests(data, reps, dtc1);
        // printResults(results7, reps);
    }
}
