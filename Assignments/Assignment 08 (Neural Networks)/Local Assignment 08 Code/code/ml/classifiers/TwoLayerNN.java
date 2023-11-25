/**
 * Collins Kariuki
 * Assignment 07
 */

package ml.classifiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Set;

import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;
import ml.utils.HashMapCounter;
import ml.utils.HashMapCounterDouble;

public class TwoLayerNN implements Classifier {
    // constructor that takes the number of hidden nodes as input as an int.

    private int numHiddenNodes;
    private double learningRate = 0.1;
    private double numIteration = 1000;
    private DataSet data;
    public ArrayList<Double> weightVectorInputs;
    public ArrayList<Double> weightVectorHidden;
    public double bias;
    public double finalOutputClassify = 0.0;

    public TwoLayerNN(int numHiddenNodes) {
        this.numHiddenNodes = numHiddenNodes;
    }

    /**
     * sets η, the learning rate, for the network. By default, set η = 0.1.
     * 
     * @param learningRate
     */
    public void setEta(double learningRate) {
        this.learningRate = learningRate;
    }

    /**
     * sets the number of times to iterate over the training
     * data during training. By default, set the number of iterations to 200.
     * 
     * @param numIteration
     */
    public void setIterations(double numIteration) {
        this.numIteration = numIteration;
    }

    /**
     * the neural network should use the tanh activation function for all
     * nodes/neurons and the nodes/neurons should all include a bias (both the
     * hidden and output layers)
     * 
     * @param x
     * @return the tanh of x
     */
    public double tanh(double x) {
        return Math.tanh(x);
    }

    public double tanhDerivative(double x) {
        return 1 - Math.pow(tanh(x), 2);
    }

    /**
     * initialize all the network weights to random values between -0.1 and 0.1
     * 
     * @return a random weight
     */
    public double randomWeight() {
        return Math.random() * 0.2 - 0.1;
    }

    public void train(DataSet data) {
        this.data = data;
        data = data.getCopyWithBias();

        // the examples in the data set
        ArrayList<Example> examples = data.getData();

        // initialize a weights array list
        weightVectorInputs = new ArrayList<Double>();
        weightVectorHidden = new ArrayList<Double>();

        for (int i = 0; i < data.getFeatureMap().size(); i++) {
            for (int j = 0; j < numHiddenNodes; j++) {
                weightVectorInputs.add(randomWeight());
            }
        }

        for (int i = 0; i < numHiddenNodes; i++) {
            weightVectorHidden.add(randomWeight());
        }

        // loop numIterations number of times
        for (int i = 0; i < numIteration; i++) {
            // randomly shuffle training data
            Collections.shuffle(examples);
            // for each example in the training data
            for (Example e : examples) {

                for (Integer featureIndex : data.getFeatureMap().keySet()) {
                    if (!e.getFeatureSet().contains(featureIndex)) {
                        e.setFeature(featureIndex, 0.0);
                    }
                }
                // STEP 1: compute all outputs going forward
                ArrayList<Double> hiddenLayerOutputs = new ArrayList<Double>();
                ArrayList<Double> hiddenLayerOutputsBeforeActivation = new ArrayList<Double>();
                // for each hidden node
                for (int j = 0; j < numHiddenNodes; j++) {
                    double individualHiddenNodeOutput = 0.0;
                    for (int k = 0; k < data.getFeatureMap().size(); k++) {
                        individualHiddenNodeOutput += e.getFeature(k)
                                * weightVectorInputs.get((k * numHiddenNodes) + j);
                    }
                    hiddenLayerOutputsBeforeActivation.add(individualHiddenNodeOutput);
                    double individualHiddenNodeOutputActivated = tanh(individualHiddenNodeOutput);
                    hiddenLayerOutputs.add(individualHiddenNodeOutputActivated);
                }

                double finalOutput = 0.0;
                for (int j = 0; j < numHiddenNodes; j++) {
                    finalOutput += hiddenLayerOutputs.get(j) * weightVectorHidden.get(j);
                }
                double finalOutputActivated = tanh(finalOutput);

                // STEP 2: calculate new weights and modified errors at output layer
                double error = tanhDerivative(finalOutput) * (e.getLabel() - finalOutputActivated);
                for (int j = 0; j < numHiddenNodes; j++) {
                    double newWeight = weightVectorHidden.get(j) + (learningRate * hiddenLayerOutputs.get(j) * error);
                    weightVectorHidden.set(j, newWeight);
                }

                // STEP 3: Recursively calculate new weights and modified errors on hidden
                for (int l = 0; l < data.getFeatureMap().size(); l++) {
                    for (int m = 0; m < numHiddenNodes; m++) {
                        // the weight update for each individual weight is:
                        double newWeightLM = weightVectorInputs.get((l * numHiddenNodes) + m) + (learningRate
                                * e.getFeature(l) * tanhDerivative(hiddenLayerOutputsBeforeActivation.get(m))
                                * weightVectorHidden.get(m) * error); // see pages 53 and 67 of lecture 17 on backpropagation
                        // STEP 4: update model with new weights
                        weightVectorInputs.set((l * numHiddenNodes) + m, newWeightLM);
                    }
                }
            }
        }
    }

    public double classify(Example example) {
        ArrayList<ArrayList<Double>> newWeightVectorInputs = new ArrayList<ArrayList<Double>>();
        for (int i = 0; i < numHiddenNodes; i++) {
            ArrayList<Double> individualHiddenNodeWeights = new ArrayList<Double>();
            for (int j = 0; j < data.getFeatureMap().size(); j++) {
                individualHiddenNodeWeights.add(weightVectorInputs.get((j * numHiddenNodes) + i));
            }
            newWeightVectorInputs.add(individualHiddenNodeWeights);
        }

        ArrayList<Double> hiddenLayerOutputs = new ArrayList<Double>();
        // for each hidden node
        for (int j = 0; j < numHiddenNodes; j++) {
            double individualHiddenNodeOutput = 0.0;
            for (int k = 0; k < data.getFeatureMap().size(); k++) {
                individualHiddenNodeOutput += example.getFeature(k) * newWeightVectorInputs.get(j).get(k);
            }
            double individualHiddenNodeOutputActivated = tanh(individualHiddenNodeOutput);
            hiddenLayerOutputs.add(individualHiddenNodeOutputActivated);
        }

        for (int j = 0; j < numHiddenNodes; j++) {
            finalOutputClassify += hiddenLayerOutputs.get(j) * weightVectorHidden.get(j);
        }

        if (finalOutputClassify > 0) {
            return 1.0;
        } else {
            return 0.0;
        }
    }

    public double confidence(Example example) {
        return Math.abs(finalOutputClassify);
    }

    public static void main(String[] args) {
        // credit to Brisa and Keneth for helping me out with the main function
        TwoLayerNN net = new TwoLayerNN(2);
        DataSet data = new DataSet("code/ml/data/titanic-train.csv", 0);

        CrossValidationSet cv = new CrossValidationSet(data, 10);
        ArrayList<Double> splitAvgs = new ArrayList<>();

        for (int i = 0; i < cv.getNumSplits(); i++) {
            DataSetSplit dataSplit = cv.getValidationSet(i);
            net.setIterations(100);
            net.setEta(0.5);
            net.train(dataSplit.getTrain());

            double allAccuracy = 0.0;
            double avg;
            for (int j = 0; j < 10; j++) {
                double numCorrect = 0.0;
                for (Example e : dataSplit.getTest().getData()) {
                    double prediction = net.classify(e);
                    if (prediction == e.getLabel()) {
                        numCorrect += 1;
                    }
                }
                double currAccuracy = numCorrect / dataSplit.getTest().getData().size();
                allAccuracy += currAccuracy;
            }
            avg = allAccuracy / 10;
            splitAvgs.add(avg);
        }
    }
}
