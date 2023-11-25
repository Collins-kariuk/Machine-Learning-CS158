package ml.classifiers;

import ml.Example;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import ml.DataSet;

public class PerceptronClassifier implements Classifier {
    public int numIterations = 25;
    public ArrayList<Integer> weightVector;
    public double prediction;
    public double bias;

    public PerceptronClassifier() {

    }

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
     * 
     */
    public void train(DataSet data) {
        // the examples in the data set
        ArrayList<Example> examples = data.getData();
        // initialize a weights array list
        weightVector = new ArrayList<Integer>();

        // the general case for the weights; initially setting all the weights to 0
        // set all the weights in weightsArr to 0
        for (int j = 0; j < data.getAllFeatureIndices().size(); j++) {
            weightVector.add(j, 0);
        }

        bias = 0.0;
        // loop until numIterations = 0
        for (int i = 0; i < this.numIterations; i++) {
            // we want to randomize the examples so we can limit bias
            Collections.shuffle(examples);
            prediction = 0.0;
            // loop through examples
            for (Example e : examples) {
                // array list that wil store the features values
                ArrayList<Double> featuresArr = new ArrayList<Double>();
                // loop through the features and add them to the features array list
                for (int k = 0; k < e.getFeatureSet().size(); k++) {
                    featuresArr.add(e.getFeature(k));
                }
                // the prediction for the current example
                prediction = bias + summation(weightVector, featuresArr);
                // if the prediction is wrong, update the weights
                if ((prediction * e.getLabel()) <= 0.0) {
                    // loop through the weights and update them
                    for (int w = 0; w < weightVector.size(); w++) {
                        // get the current weight
                        int currentWeight = weightVector.get(w);
                        // update the weight
                        // we need to cast the product of the feature and label to an int because the
                        // weight is an int
                        currentWeight = (int) (currentWeight + (e.getFeature(w) * e.getLabel()));
                        // replace the previous weight with the new adjusted weight
                        weightVector.set(w, currentWeight);
                    }
                    // update the bias
                    bias = bias + e.getLabel();
                }
            }
        }
    }

    /**
     * Compute the summation of the weights times the features for this example
     * n number of times, where n is the number of features
     */
    public double summation(ArrayList<Integer> weightsArr, ArrayList<Double> featureArr) {
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
        prediction = summation(weightVector, featuresArr) + bias;
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
        for (int i = 0; i < weightVector.size(); i++) {
            weights += i + ":" + weightVector.get(i) + " ";
        }
        return weights + bias;
    }

    public static void main(String[] args) {
        PerceptronClassifier test = new PerceptronClassifier();
        DataSet someData = new DataSet("data/titanic-train.csv");

        DataSet[] split = someData.split(.8);
        test.train(split[0]); // the train portion of the data
        double allAccuracy = 0.0;

        for (int i = 0; i < 100; i++) {
            double correctCount = 0.0;
            for (Example e : split[0].getData()) {
                double prediction = test.classify(e);
                if (prediction == e.getLabel()) {
                    correctCount += 1;
                }
            }
            double currentaccuracy = correctCount / split[0].getData().size();
            allAccuracy += currentaccuracy;
        }

        double avg = allAccuracy / 100;
        System.out.print(avg);
    }
}
