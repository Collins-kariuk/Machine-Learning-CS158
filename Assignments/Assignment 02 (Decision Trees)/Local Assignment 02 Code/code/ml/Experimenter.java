// Munir Vafai
// Collins Kariuki
package ml;

import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import ml.classifiers.DecisionTreeClassifer;
import ml.classifiers.RandomClassifier;

public class Experimenter {

    /*
     * What is the accuracy of the random classifier on the Titanic data set from
     * assignment 1. To
     * calculate this, generate a random 80/20 split (using dataset.split(0.8))
     * train the model
     * on the 80% fraction and then evaluate the accuracy on the 20% fraction.
     * Repeat this 100
     * times and average the result (hint: do the repetition in code :).
     */
    public double q1(DataSet dataset) {
        double totalAccuracy = 0.0;
        RandomClassifier randomClassifier = new RandomClassifier();

        for (int i = 0; i < 100; i++) {
            DataSet[] split = dataset.split(0.8);
            double correctPredictions = 0.0;
            randomClassifier.train(split[0]); // not really needed lol
            for (Example e : split[1].getData()) {
                double result = randomClassifier.classify(e);
                if (result == e.getLabel()) {
                    correctPredictions++;
                }
            }
            double curAccuracy = correctPredictions / split[1].getData().size();
            totalAccuracy += curAccuracy;
        }
        return totalAccuracy / 100;
    }

    /*
     * What is the accuracy of your decision tree classifier on the Titanic data set
     * with unlimited
     * depth. As above, average the results over 100 random 80/20 splits.
     */
    public double q2(DataSet dataset, DecisionTreeClassifer tree) {
        double totalAccuracy = 0;
        for (int i = 0; i < 100; i++) {
            DataSet[] split = dataset.split(0.8);
            tree.train(split[0]);
            double correctPredictions = 0;
            for (Example e : split[1].getData()) {
                double result = tree.classify(e);
                if (result == e.getLabel())
                    correctPredictions++;
            }
            double curAccuracy = correctPredictions / split[1].getData().size();
            totalAccuracy += curAccuracy;
        }
        return totalAccuracy / 100;
    }

    /*
     * What is the best depth limit to use for this data? To answer this, do the
     * same calculations
     * as above (average 100 experiments), but do it for increasing depth limits,
     * specifically 0, 1, 2,
     * ..., 10. Show all of your results.
     */
    

    /*
     * Do we see overfitting with this data set? Repeat the experiment from question
     * 3 with increasing depth (0, 1, ..., 10) and calculate the accuracy this time
     * on both the testing data
     * (like before) and the training data. Create a graph with these results and
     * then provide a 1-2
     * sentence answer describing the graph.
     */
    public List<SimpleEntry<Double, Double>> q4(DataSet dataset, DecisionTreeClassifer tree) {
        List<SimpleEntry<Double, Double>> accuracies = new ArrayList<>();

        for (int depth = 0; depth <= 10; depth++) {
            tree.setDepthLimit(depth);
            double totalTestingAccuracy = 0.0;
            double totalTrainingAccuracy = 0.0;

            for (int i = 0; i < 100; i++) {
                DataSet[] split = dataset.split(0.8);
                tree.train(split[0]);

                // training accuracy
                double correctTrainingPredictions = 0.0;
                for (Example e : split[0].getData()) {
                    double result = tree.classify(e);
                    if (result == e.getLabel())
                        correctTrainingPredictions++;
                }
                totalTrainingAccuracy += correctTrainingPredictions / split[0].getData().size();

                // testing accuracy
                double correctTestingPredictions = 0.0;
                for (Example e : split[1].getData()) {
                    double result = tree.classify(e);
                    if (result == e.getLabel())
                        correctTestingPredictions++;
                }
                totalTestingAccuracy += correctTestingPredictions / split[1].getData().size();
            }

            double aveTrainingAccuracy = totalTrainingAccuracy / 100;
            double aveTestingAccuracy = totalTestingAccuracy / 100;

            accuracies.add(new SimpleEntry<>(aveTrainingAccuracy, aveTestingAccuracy));
        }

        return accuracies;
    }

    /*
     * How does the amount of training data affect performance? To answer this, do
     * the same
     * calculations as above (average 100 experiments), but start with splits of
     * 0.05 (5% of the data
     * used for training) and work up to splits of size 0.9 (90% of the data used
     * for training) in
     * increments of 0.05. For these experiments use full depth trees, i.e. trees
     * without any depth limit.
     */
    public List<SimpleEntry<Double, Double>> q5(DataSet dataset, DecisionTreeClassifer tree) {
        List<SimpleEntry<Double, Double>> results = new ArrayList<>();

        for (double splitRatio = 0.05; splitRatio <= 0.9; splitRatio += 0.05) {
            double totalTestingAccuracy = 0.0;
            double totalTrainingAccuracy = 0.0;

            for (int i = 0; i < 100; i++) {
                DataSet[] split = dataset.split(splitRatio);

                // Training Accuracy
                double trainingCorrectPredictions = 0.0;
                tree.train(split[0]);
                for (Example e : split[0].getData()) {
                    double result = tree.classify(e);
                    if (result == e.getLabel())
                        trainingCorrectPredictions++;
                }
                double trainingAccuracy = trainingCorrectPredictions / split[0].getData().size();
                totalTrainingAccuracy += trainingAccuracy;

                // Testing Accuracy
                double testingCorrectPredictions = 0.0;
                for (Example e : split[1].getData()) {
                    double result = tree.classify(e);
                    if (result == e.getLabel())
                        testingCorrectPredictions++;
                }
                double testingAccuracy = testingCorrectPredictions / split[1].getData().size();
                totalTestingAccuracy += testingAccuracy;
            }

            double aveTrainingAccuracy = totalTrainingAccuracy / 100;
            double aveTestingAccuracy = totalTestingAccuracy / 100;

            results.add(new SimpleEntry<>(aveTrainingAccuracy, aveTestingAccuracy));
        }
        return results;
    }

    public static void main(String[] args) {
        Experimenter e = new Experimenter();
        DataSet dataset = new DataSet("data/titanic-train.csv");
        DecisionTreeClassifer d = new DecisionTreeClassifer();
        System.out.println("q1: " + e.q1(dataset));
        System.out.println("q2 :" + e.q2(dataset, d));

        List<Double> q3 = e.q3(dataset, d);
        for (int i = 0; i < q3.size(); i++) {
            System.out.println("Depth: " + i + " Accuracy: " + q3.get(i));
        }
        double max = Collections.max(q3);
        int maxIndex = q3.indexOf(max);
        System.out.println("Ideal depth limit: " + maxIndex + " with accuracy: " + max);
        List<SimpleEntry<Double, Double>> q4 = e.q4(dataset, d);

        System.out.println("Depth\tTraining Accuracy\tTesting Accuracy");
        for (int depth = 0; depth <= 10; depth++) {
            SimpleEntry<Double, Double> accuracies = q4.get(depth);
            double trainingAccuracy = accuracies.getKey();
            double testingAccuracy = accuracies.getValue();

            System.out.println(depth + "\t" + trainingAccuracy * 100 + "\t\t\t" + testingAccuracy * 100);
        }

        List<SimpleEntry<Double, Double>> results = e.q5(dataset, d);

        System.out.println("splitRatio\tTrainingAccuracy\tTestingAccuracy");
        double splitRatio = 0.05;
        for (SimpleEntry<Double, Double> entry : results) {
            System.out.printf("%.2f\t\t%.4f\t\t\t%.4f%n", splitRatio, entry.getKey(), entry.getValue());
            splitRatio += 0.05;
        }

    }
}