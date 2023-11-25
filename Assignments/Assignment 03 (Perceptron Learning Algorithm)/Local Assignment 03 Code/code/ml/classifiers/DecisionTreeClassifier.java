package ml.classifiers;

import ml.DataSet;
import ml.Example;
import java.util.ArrayList;
import java.util.Set;
import java.util.HashMap;
import java.util.Map;
import java.util.HashSet;

public class DecisionTreeClassifier implements Classifier {
    public int featureIndex;
    public double prediction;
    public int depthLimit;
    public DecisionTreeNode tree;
    public DataSet data;

    public void DecicisonTreeClassifier() {
    }

    /**
     * Train the model based on DataSet data, taking into account the depth limit
     * restriction
     * 
     * @param data
     */
    public void train(DataSet data) {
        ArrayList<Example> dataArr = data.getData(); // parent arraybefore splitting
        HashMap<Integer, Double> errorHash = trainingErrorsCalc(dataArr);
        int bestFeature = bestFeature(errorHash); // the feature with the lowest training error
        Set<Integer> usedFeatures = new HashSet<Integer>();
        DecisionTreeNode parent = new DecisionTreeNode(bestFeature);
        double parentLabel = majorityLabel(dataArr);
        // testing:
        this.tree = buildTree(dataArr, depthLimit, usedFeatures, parent, parentLabel);
    }

    /**
     * Predicts the label for a given example
     * 
     * @param example individual example from the dataSet
     * @return returns a label for the input
     */
    public double classify(Example example) {
        return classifyHelp(this.tree, example);
    }

    /**
     * Helper function for classify. Allows us to recurse through the tree
     * 
     * @param tree
     * @param example
     * @return returns label prediciton
     */
    public double classifyHelp(DecisionTreeNode tree, Example example) {
        if (tree.isLeaf()) {
            return tree.prediction();
        } else {
            // get the index of the feature from the tree
            int featureIndex = tree.getFeatureIndex();
            double val = example.getFeature(featureIndex);
            // if we go to the left (0.0)
            if (val == 0.0) {
                return classifyHelp(tree.getLeft(), example);
            }
            // if we go to the right (1.0)
            else {
                return classifyHelp(tree.getRight(), example);
            }
        }
    }

    // BASE CASE HELPER FUNCTIONS -> PICKING THE LABEL

    /**
     * Picks the majority label based on count
     * 
     * @param dataArr
     * @return returns the label with the more instances
     */
    public double majorityLabel(ArrayList<Example> dataArr) {
        double label1 = dataArr.get(0).getLabel();
        double label2 = 0.0;
        int count_label1 = 0;
        int count_label2 = 0;

        // we loop through the data and the labels
        for (int i = 0; i < dataArr.size(); i++) {
            if (dataArr.get(i).getLabel() != label1) { // we want to save the diff label to be able to count it
                label2 = dataArr.get(i).getLabel();
                count_label2 += 1;
            }
            if (dataArr.get(i).getLabel() == label1) {
                count_label1 += 1;
            }
        }
        // check for tie
        if (count_label1 == count_label2) {
            return 3.0;
        }
        // find majority label
        else { // we store the max value
            double maxValue = Math.max(count_label1, count_label2);

            if (maxValue == count_label1) {
                return label1;
            } else {
                return label2;
            }
        }
    }

    /**
     * Checks if the label is the same for all given examples
     * 
     * @param dataArr
     * @return returns false if labels are different, true if all label values are
     *         the same
     */
    public boolean sameLabel(ArrayList<Example> dataArr) {
        if (dataArr.isEmpty()) {
            return false;
        }
        double label1 = dataArr.get(0).getLabel();

        // loop throusugh the examples
        for (int i = 0; i < dataArr.size(); i++) {
            // if all the labels are the same, then you do not split
            if (dataArr.get(i).getLabel() != label1) {
                // returns false if there is a different label
                return false;
            }
        }
        // returns true if the label is the same for all of the examples
        return true;
    }

    /**
     * Checks if all the examples have the same features
     * 
     * @param dataArr
     * @return true if all the features are the same, false if at least one of
     *         features is different
     */
    public boolean sameFeatureValues(ArrayList<Example> dataArr) {
        if (dataArr.isEmpty()) {
            return false;
        }
        Example baseRow = dataArr.get(0);

        // for all of the examples in the data
        for (int i = 0; i < dataArr.size(); i++) {
            // if any of them do not have the same features, return false
            if (!baseRow.equalFeatures(dataArr.get(i))) {
                return false;
            }
        }
        return true;
    }

    /**
     * calculates the training errors for all features and stores these for all of
     * them
     * 
     * @param dataArr
     * @return a hashmap with feature number as the key and the training error as
     *         the value.
     */
    public HashMap<Integer, Double> trainingErrorsCalc(ArrayList<Example> dataArr) {
        Set<Integer> features = dataArr.get(0).getFeatureSet();
        HashMap<Integer, Double> trainingErrors = new HashMap<Integer, Double>();

        // we loop through each feature
        for (int featureNum = 0; featureNum < features.size(); featureNum++) {
            double yesLeft = 0.0;
            double noLeft = 0.0;
            double yesRight = 0.0;
            double noRight = 0.0;
            double trainErrorNumerator = 0.0;
            double trainingError = 0.0;
            // we want to loop though all of the features values of this feature, so through
            // all examples.
            for (int i = 0; i < dataArr.size(); i++) {
                // we are in an example right now
                Example currentExample = dataArr.get(i);
                if (currentExample.getFeature(featureNum) == 0.0 && currentExample.getLabel() == 1.0) {
                    yesLeft++;
                } else if (currentExample.getFeature(featureNum) == 0.0 && currentExample.getLabel() == -1.0) {
                    noLeft++;
                } else if (currentExample.getFeature(featureNum) == 1.0 && currentExample.getLabel() == 1.0) {
                    yesRight++;
                } else if (currentExample.getFeature(featureNum) == 1.0 && currentExample.getLabel() == -1.0) {
                    noRight++;
                }
            }
            // we want to save the min as they are the mistakes that the model made
            trainErrorNumerator += Math.min(yesLeft, noLeft);
            trainErrorNumerator += Math.min(yesRight, noRight);

            // we calculate out training error
            trainingError = trainErrorNumerator / dataArr.size();
            trainingErrors.put(featureNum, trainingError);
        }
        System.out.println(trainingErrors);
        return trainingErrors;
    }

    /**
     * Finds the best feature to split on
     * 
     * @param trainingErrors a HashMap containig all calculated training errors
     * @return the best feature to split on
     */
    public int bestFeature(HashMap<Integer, Double> trainingErrors) {
        // since training error cannot be greater than 1, we start with 2.0
        double minValue = 2.0;
        int minKey = -1;

        // Iterate through the HashMap
        for (Map.Entry<Integer, Double> entry : trainingErrors.entrySet()) {
            double value = entry.getValue();
            int key = entry.getKey();

            // Check if the current value is smaller than the current minimum
            if (value < minValue) {
                minValue = value;
                minKey = key;
            }
        }
        return minKey;
    }

    /**
     * Builds the tree through recursion
     * 
     * @param dataArr     the datase being passed in, from which the tree
     *                    will be
     *                    built
     * @param maxDepth    height of decision tree
     * @param useFeatures set of features that we have used, we can add to
     *                    this
     *                    later as we examine each feature
     *                    make sure to change the return type if I need to
     *                    return
     * @param parent      parent node
     * @param parentArray array of the parent
     * @return returns the tree that we have built
     */
    public DecisionTreeNode buildTree(ArrayList<Example> dataArr, int depthLimit, Set<Integer> usedFeatures,
            DecisionTreeNode parent, double parentLabel) {
        // Base Cases:

        // 1. If all data belong to the same label, pick that label
        if (sameLabel(dataArr)) {
            // here we might want to check if there is a tie then what to return instead
            if (majorityLabel(dataArr) == 3.0) {
                // get the label from the parent node
                return new DecisionTreeNode(parent.prediction());
            }
            return new DecisionTreeNode(majorityLabel(dataArr));

        }
        // 2. if all the data have the same feature values (like the column) pick
        // majority label
        else if (sameFeatureValues(dataArr)) {
            return new DecisionTreeNode(majorityLabel(dataArr));
        }

        // 3. If we’re out of features to examine, pick majority label (if tie, parent
        // majority),
        else if (!dataArr.isEmpty() && usedFeatures.size() == dataArr.get(0).getFeatureSet().size()) {
            // we want to pick the majority label
            if (majorityLabel(dataArr) == 3.0) {
                // pick parent majority
                return new DecisionTreeNode(parent.prediction());
            }
            return new DecisionTreeNode(majorityLabel(dataArr));
        }

        // 4. If the we don’t have any data left, pick majority label of parent (i.e
        // we've ran through all the data)
        else if (dataArr.isEmpty()) {
            return new DecisionTreeNode(parentLabel);
        }
        // 5. This is the fifth case if we are reached the limit
        else if (depthLimit == 0) {
            // return parent majority l;
            return new DecisionTreeNode(parentLabel);
        }

        // Recursive Case(s)
        else {
            HashMap<Integer, Double> errorHash = trainingErrorsCalc(dataArr);
            int bestFeature = bestFeature(errorHash); // the feature with the lowest training error

            DecisionTreeNode root = new DecisionTreeNode(bestFeature); // index of best feature
            Set<Integer> usedFeatureCopy = new HashSet<Integer>(usedFeatures);
            usedFeatureCopy.add(bestFeature); // adds the feature we've split on to the set of features we've used
            // split data based on feature value to be able to recurse through
            ArrayList<Example> dataLeft = new ArrayList<Example>();
            ArrayList<Example> dataRight = new ArrayList<Example>();
            for (int i = 0; i < dataArr.size(); i++) {
                // feature values ar either 0 or 1
                if (dataArr.get(i).getFeature(bestFeature) == 0) {

                    dataLeft.add(dataArr.get(i));
                } else {
                    dataRight.add(dataArr.get(i));
                }
            }
            usedFeatures.add(bestFeature);
            if (dataLeft.size() == 0) {
                root.setLeft(buildTree(dataLeft, depthLimit - 1, usedFeatureCopy, root, parentLabel));
            } else {
                root.setLeft(buildTree(dataLeft, depthLimit - 1, usedFeatureCopy, root, majorityLabel(dataLeft)));
            }

            if (dataRight.isEmpty()) {
                root.setRight(buildTree(dataRight, depthLimit - 1, usedFeatureCopy, root, parentLabel));
            } else {
                root.setRight(buildTree(dataRight, depthLimit - 1, usedFeatureCopy, root, majorityLabel(dataRight)));
            }

            return root;
        }
    }

    /**
     * Sets the depth of the decision tree
     * 
     * @param depth The depth value of the tree
     */
    public void setDepthLimit(int depth) {
        this.depthLimit = depth;
    }

    /**
     * Prints out the decision tree
     */
    public String toString() {
        return tree.treeString((data.getFeatureMap()));
    }

    /**
     * Main function to be able to test our functions as we go
     */
    public static void main(String[] args) {
        DecisionTreeClassifier test = new DecisionTreeClassifier();
        DataSet someData = new DataSet("data/titanic-train.csv");
        ArrayList<Example> testingArray = someData.getData();
        DataSet[] split = someData.split(.9);
        test.train(split[0]); // the train portion of the data
        double allAccuracy = 0.0;
        test.setDepthLimit(20);
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
