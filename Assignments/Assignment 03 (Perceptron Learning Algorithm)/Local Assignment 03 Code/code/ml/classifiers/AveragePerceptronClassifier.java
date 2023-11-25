package ml.classifiers;

import ml.Example;

import java.util.ArrayList;
import java.util.Collections;
import ml.DataSet;

public class AveragePerceptronClassifier
        implements Classifier {
    public int numIterations = 25;
    public ArrayList<Double> currentWeightVector; // w in the pseudocode
    public double bias; // b in the pseudocode
    public double weightedBias; // b2 in the pseudocode
    public double total; // total number of examples seen so far
    public int countCorrect; // updated in the pseudocode, num exmaples the current weights have gotten
                             // correct
    public ArrayList<Double> avgWeightVector; // u in the pseudocode, add to thse a weighted copy of the current
                                              // weight vector when a mistake is made

    /**
     * Sets the number of iterations
     * 
     * @param iterations
     */
    public void setIterations(int iterations) {
        this.numIterations = iterations;
    }

    /**
     * Train this classifier based on the data set
     * 
     * @param data
     */
    public void train(DataSet data) {
        // the examples in the data set
        ArrayList<Example> examples = data.getData();
        // initialize a weights array list
        currentWeightVector = new ArrayList<Double>();
        avgWeightVector = new ArrayList<Double>();

        // the general case for the weights; initially setting all the weights to 0
        // set all the weights in weightsArr to 0
        for (int j = 0; j < data.getAllFeatureIndices().size(); j++) {
            currentWeightVector.add(j, 0.0);
            avgWeightVector.add(j, 0.0);
        }
        // initialize the bias to 0
        bias = 0.0;
        total = 0.0;
        countCorrect = 0;
        weightedBias = 0.0;

        for (int i = 0; i < this.numIterations; i++) {
            Collections.shuffle(examples);
            double prediction = 0.0;
            for (Example e : examples) {
                // array list that wil store the features values
                ArrayList<Double> featuresArr = new ArrayList<Double>();
                // loop through the features and add them to the features array list
                for (int k = 0; k < e.getFeatureSet().size(); k++) {
                    featuresArr.add(e.getFeature(k));
                }
                // the prediction for the current example
                prediction = bias + summation(currentWeightVector, featuresArr);

                // if we nmisclassify examples e using weightVector
                if (prediction * e.getLabel() <= 0) {
                    // loop through final weighted weights and update them
                    for (int x = 0; x < avgWeightVector.size(); x++) {
                        // get the current weighted weight
                        double finalWeightedWeight = avgWeightVector.get(x);
                        // update the weighted weight
                        // we need to cast the product of the feature and label to an int because the
                        finalWeightedWeight = finalWeightedWeight + countCorrect * currentWeightVector.get(x);
                        // replace the previous weight with the new adjusted weight
                        avgWeightVector.set(x, finalWeightedWeight);
                    }
                    // update the weighetd bias
                    weightedBias += countCorrect * bias;

                    // loop through the perceptron weights and update them
                    for (int w = 0; w < currentWeightVector.size(); w++) {
                        // get the current weight
                        double currentWeight = currentWeightVector.get(w);
                        // update the weight
                        // we need to cast the product of the feature and label to an int because the
                        // weight is an int
                        currentWeight = currentWeight + (e.getFeature(w) * e.getLabel());
                        // replace the previous weight with the new adjusted weight
                        currentWeightVector.set(w, currentWeight);
                    }
                    // update the bias
                    bias = bias + e.getLabel();
                    countCorrect = 0;
                } // end of misclassify
                countCorrect += 1;
                total += 1;
            } // end of dataset for loop
        } // end of outer loop

        // do one last weighted update here of the avgWeughtVectir and weightedBias
        // weightes based on the final weights
        for (int z = 0; z < currentWeightVector.size(); z++) {
            double finalWeightedWeight = currentWeightVector.get(z);
            // update the weighted weight
            // we need to cast the product of the feature and label to an int because the
            // weight is an int
            finalWeightedWeight = finalWeightedWeight + countCorrect * currentWeightVector.get(z);
            avgWeightVector.set(z, finalWeightedWeight);
        }
        weightedBias += countCorrect * bias;

        // divide all of the aggregate weights by the total number of exaples
        for (double weightedWeight : avgWeightVector) {
            weightedWeight = weightedWeight / total;
        }

        weightedBias = weightedBias / total;
    }

    /**
     * Compute the summation of the weights times the features for this example
     * n number of times, where n is the number of features
     */
    public double summation(ArrayList<Double> weightsArr, ArrayList<Double> featureArr) {
        // initialize a sum variable
        double sum = 0.0;
        // loop through the weights and features and multiply them together
        for (int i = 0; i < weightsArr.size(); i++) {
            sum = sum + (weightsArr.get(i) * featureArr.get(i));
        }
        // return the summation
        return sum;
    }

    /**
     * Classify the example. Should only be called *after* train has been called.
     * 
     * @param example
     * @return the class label predicted by the classifier for this example
     */
    public double classify(Example example) {
        double prediction = 0.0;
        // get the features of the example
        ArrayList<Double> featuresArr = new ArrayList<Double>();
        // loop through the features and add them to the features array list
        for (int i = 0; i < example.getFeatureSet().size(); i++) {
            featuresArr.add(example.getFeature(i));
        }
        // feature * associated avg weight
        prediction = summation(avgWeightVector, featuresArr) + weightedBias;
        if (prediction >= 0.0) {
            return 1.0;
        } else if (prediction < 0.0) {
            return -1.0;
        } else {
            return 47.7;
        }
    }

    /**
     * string representation of the algorithm, , <feature_number>:<weight>, space
     * separated with the features in increasing
     * order.
     */
    public String toString() {
        String weights = new String();
        for (int i = 0; i < avgWeightVector.size(); i++) {
            weights += i + ":" + avgWeightVector.get(i) + " ";
        }
        return weights + bias;
    }

    /**
     * testing out algorithm, should be testing for the same things as perceptron
     */
    public static void main(String[] args) {
        // create a new average perceptron classifier that will serve as the test
        AveragePerceptronClassifier test = new AveragePerceptronClassifier();
        // create a data set from your file of choice
        DataSet someData = new DataSet("data/titanic-train.csv");
        // split the data set into a train and test set depending on your fraction of choice
        DataSet[] split = someData.split(.8);
        // train the classifier on the train portion of the data
        test.train(split[0]);
        double allAccuracy = 0.0;
        // loop through the test data and classify each example
        for (int i = 0; i < 100; i++) {
            // a variable to keep track of the number of correct classifications
            double correctCount = 0.0;
            // loop through the test data and classify each example
            for (Example e : split[0].getData()) {
                // classify the example
                double prediction = test.classify(e);
                // if the prediction is correct, increment the correct count
                if (prediction == e.getLabel()) {
                    correctCount += 1;
                }
            }
            // calculate the accuracy of the classifier
            // the denominator is the number of examples in the test set
            double currentaccuracy = correctCount / split[0].getData().size();
            allAccuracy += currentaccuracy;
        }
        // calculate the average accuracy of the classifier
        double avg = allAccuracy / 100;
        System.out.print(avg);
    }
}
