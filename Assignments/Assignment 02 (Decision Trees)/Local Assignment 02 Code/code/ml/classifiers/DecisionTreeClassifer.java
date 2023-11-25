// Munir Vafai
// Collins Kariuki
package ml.classifiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Set;
import ml.DataSet;
import ml.Example;

/*
 * A class to represent a decision tree classifier
 */
public class DecisionTreeClassifer implements Classifier {
    public int depthLimit;
    public DecisionTreeNode root = null;

    public DecisionTreeClassifer() {
        depthLimit = Integer.MAX_VALUE;
    }

    // calculate the “score” for each feature if we used it to split the data
    private ArrayList<Double> calculateScores(ArrayList<Integer> featuresList, ArrayList<Example> examples) {
        ArrayList<Double> scores = new ArrayList<Double>();
        for (Integer columnNum : featuresList) { // for each remaining feature
            double totalFeatureValues0 = 0;
            double totalFeatureValues1 = 0;
            double feature0Yeses = 0;
            double feature1Yeses = 0;
            for (int i = 0; i < examples.size(); i++) { // for each row
                double featureValue = examples.get(i).getFeature(columnNum);
                double rowLabel = examples.get(i).getLabel();
                if (featureValue == 0) {
                    totalFeatureValues0++;
                    if (rowLabel == 1)
                        feature0Yeses++;
                } else if (featureValue == 1) {
                    totalFeatureValues1++;
                    if (rowLabel == 1)
                        feature1Yeses++;
                }
            }
            double score0 = feature0Yeses / totalFeatureValues0;
            double score1 = feature1Yeses / totalFeatureValues1;
            double maxScore0 = Math.max(score0, 1 - score0);
            double maxScore1 = Math.max(score1, 1 - score1);
            double error0 = 1 - maxScore0;
            double error1 = 1 - maxScore1;

            double weightedError = (error0 * totalFeatureValues0 + error1 * totalFeatureValues1) / examples.size();
            scores.add(weightedError);
        }
        return scores;
    }

    // return the label that is the majority label for the given examples
    private double getMajorityLabel(ArrayList<Example> examples) {
        double numberOfYeses = 0;
        double numExamples = examples.size();
        for (int i = 0; i < numExamples; i++) { // for each feature value
            if (examples.get(i).getLabel() == 1)
                numberOfYeses++;
        }
        if (numberOfYeses > numExamples / 2) {
            return 1.0;
        } else {
            return -1.0;
        }
    }

    // recursively build the tree
    private DecisionTreeNode trainHelper(ArrayList<Example> examples, ArrayList<Example> parentExamples,
            ArrayList<Integer> featuresRemaining, DecisionTreeNode parent, int depth) {
        if (depth >= depthLimit) { // depth limit case
            return new DecisionTreeNode(getMajorityLabel(examples));
        }
        if (examples.size() == 0) { // base case 4
            double majorityLabel = getMajorityLabel(parentExamples);
            return new DecisionTreeNode(majorityLabel);
        }
        if (allExamplesHaveSameLabel(examples)) { // base case 1
            return new DecisionTreeNode(examples.get(0).getLabel());
        }
        if (featuresRemaining.size() == 0 || allExamplesHaveSameFeatures(examples)) { // base case 2 and 3
            double majorityLabel = getMajorityLabel(examples);
            if (majorityLabel == -1 && parentExamples != null)
                majorityLabel = getMajorityLabel(parentExamples);
            return new DecisionTreeNode(majorityLabel);
        }

        // Otherwise (i.e. if none of the base cases apply):
        // pick the feature with the highest score
        ArrayList<Double> scores = calculateScores(featuresRemaining, examples);
        double highestScore = Collections.max(scores);
        int highestScoreFeatureIndex = scores.indexOf(highestScore);
        int chosenFeature = featuresRemaining.get(highestScoreFeatureIndex);

        DecisionTreeNode node = new DecisionTreeNode(chosenFeature);
        if (parent != null) {
            if (parent.getLeft() == null) {
                parent.setLeft(node);
            } else {
                parent.setRight(node);
            }
        }
        // partition the data based on that feature, e.g. data_left and data_right
        ArrayList<Example> dataLeft = new ArrayList<>();
        ArrayList<Example> dataRight = new ArrayList<>();
        for (Example example : examples) {
            if (example.getFeature(chosenFeature) == 0)
                dataLeft.add(example);
            else
                dataRight.add(example);
        }

        ArrayList<Integer> featuresRemainingCopy = new ArrayList<>(featuresRemaining);
        featuresRemainingCopy.remove(Integer.valueOf(chosenFeature));

        node.setLeft(trainHelper(dataLeft, examples, featuresRemainingCopy, node, depth + 1));
        node.setRight(trainHelper(dataRight, examples, featuresRemainingCopy, node, depth + 1));

        return node;
    }

    // return true if all examples have the same features
    private boolean allExamplesHaveSameFeatures(ArrayList<Example> examples) {
        if (examples == null || examples.isEmpty())
            return true;
        Example firstExample = examples.get(0);
        Set<Integer> firstFeatureSet = firstExample.getFeatureSet();

        for (Example e : examples) {
            Set<Integer> currentFeatureSet = e.getFeatureSet();
            if (!firstFeatureSet.equals(currentFeatureSet))
                return false;

            for (int featureIndex : firstFeatureSet) {
                if (e.getFeature(featureIndex) != firstExample.getFeature(featureIndex)) {
                    return false;
                }
            }
        }
        return true;
    }

    // return true if all examples have the same label
    private boolean allExamplesHaveSameLabel(ArrayList<Example> examples) {
        double firstLabel = examples.get(0).getLabel();
        for (Example e : examples) {
            if (e.getLabel() != firstLabel)
                return false;
        }
        return true;
    }

    // train the decision tree on the given data
    public void train(DataSet data) {
        ArrayList<Example> examples = data.getData();
        Set<Integer> integerList = data.getAllFeatureIndices();
        ArrayList<Integer> featuresRemaining = new ArrayList<>(integerList);
        // System.out.println("depthLimit: " + depthLimit);
        DecisionTreeNode tree = trainHelper(examples, null, featuresRemaining, root, 0);
        root = tree;
        // System.out.println("tree toString(): ");
        // System.out.println(toString());
    }

    // classify the example
    public double classify(Example example) {
        return classifyHelper(root, example);
    }

    // recursively classify the example
    private double classifyHelper(DecisionTreeNode node, Example example) {
        if (node.getLeft() == null && node.getRight() == null) // It's a leaf node
            return node.prediction();

        double featureValue = example.getFeature(node.getFeatureIndex());
        if (featureValue == 0.0) {
            return classifyHelper(node.getLeft(), example);
        } else {
            return classifyHelper(node.getRight(), example);
        }
    }

    // set the depth limit for the decision tree
    public void setDepthLimit(int depth) {
        depthLimit = depth;
    }

    public String toString() {
        return root.treeString();
    }

    public static void main(String[] args) {
        DecisionTreeClassifer test = new DecisionTreeClassifer();
        // DataSet dataset = new DataSet("data/default.csv");
        DataSet dataset = new DataSet("data/titanic-train.csv");
        // test.setDepthLimit(3);
        test.train(dataset);
    }

}