// Munir Vafai
// Collins Kariuki
package ml.classifiers;

import java.util.ArrayList;
import java.util.Set;
import java.util.AbstractMap.SimpleEntry;

import ml.data.DataSet;
import ml.data.Example;

/**
 * An OVA classifier that uses a factory to create classifiers for each label.
 */
public class OVAClassifier implements Classifier {
    ClassifierFactory factory;
    ArrayList<SimpleEntry<Double, Classifier>> classifiers = new ArrayList<>();

    /**
     * Creates a new OVA classifier that uses the given factory to create
     * classifiers for each label.
     * 
     * @param factory
     */
    public OVAClassifier(ClassifierFactory factory) {
        this.factory = factory;
    }

    /**
     * Trains a classifier for each label in the dataset. Each classifier is trained
     * to predict 1 if the label matches the label it's training for, and -1 if it
     * doesn't.
     */
    public void train(DataSet data) {
        // for each label, train a classifier that predicts 1 if the label matches
        Set<Double> labels = data.getLabels();
        for (Double label : labels) {
            // make a copy of the data
            DataSet newData = getNewDataSet(data);
            // get the classifier (from the factory)
            Classifier classifier = factory.getClassifier();

            // rewrite labels to be 1 or -1 (1 if label matches, -1 if not)
            for (int i = 0; i < newData.getData().size(); i++) {
                Example example = newData.getData().get(i);
                // if the label of the example matches the label we're training for, set it to 1
                // otherwise, set it to -1
                // we do this for every label in our dataset
                if (example.getLabel() == label) {
                    example.setLabel(1);
                } else {
                    example.setLabel(-1);
                }
            }
            // train the classifier on the new data
            classifier.train(newData);
            if(label == 10) { //zinfandel
                System.out.println(classifier.toString());
            }
            // add the label we're looking at and it's corresponding classifier to the list
            // we'll end up with a list of classifiers, one for each label
            classifiers.add(new SimpleEntry<>(label, classifier));
        }
        System.out.println("END OF LOOP");
    }

    /**
     * Returns a new dataset with the same features as the original, but with the
     * same examples
     * 
     * @param data
     * @return a new dataset with the same features as the original, but with the
     *         same examples
     */
    private DataSet getNewDataSet(DataSet data) {
        // make a new dataset with the same features as the original
        DataSet newData = new DataSet(data.getFeatureMap());
        // add the same examples to the new dataset
        for (Example example : data.getData()) {
            // make a copy of the example
            Example newExample = new Example(example);
            // add the example to the new dataset
            newData.addData(newExample);
        }
        return newData;
    }

    /**
     * Classifies the example by running it through each classifier and returning
     * the label with the highest confidence. If no classifier predicts 1, then the
     * label with the lowest confidence is returned.
     * 
     * @param example
     * @return the label with the highest confidence. If no classifier predicts 1,
     *         then the label with the lowest confidence is returned.
     */
    public double classify(Example example) {
        // for each classifier, get the prediction and confidence
        double maxConfidence = Double.NEGATIVE_INFINITY;
        double minConfidence = Double.POSITIVE_INFINITY;

        double maxConfidenceLabel = -1;
        double minConfidenceLabel = -1;
        // set a flag to see if any classifier predicts 1
        boolean positivePredictionAppears = false;
        for (SimpleEntry<Double, Classifier> entry : classifiers) {
            // get the corresponding label and classifier
            double label = entry.getKey();
            Classifier classifier = entry.getValue();

            // the prediction and confidence gotten from the classifier
            // each classifier from the factory has its own implementation of classify and
            // confidence
            double prediction = classifier.classify(example);
            double confidence = classifier.confidence(example);

            if (prediction == 1) {
                // we know there's at least one classifier that predicts 1 so we can set the
                // flag to true
                positivePredictionAppears = true;
                if (confidence > maxConfidence) {
                    maxConfidence = confidence;
                    maxConfidenceLabel = label;
                }
            } else if (confidence < minConfidence) {
                minConfidence = confidence;
                minConfidenceLabel = label;
            }
        }

        if (positivePredictionAppears)
            return maxConfidenceLabel;
        else
            return minConfidenceLabel;
    }

    public double confidence(Example example) {
        return 0;
    }

    public static void main(String[] args) {
        DataSet wineDataset = new DataSet("data/wines.train", DataSet.TEXTFILE);
        ClassifierFactory factory = new ClassifierFactory(ClassifierFactory.DECISION_TREE, 3);
        OVAClassifier classifier = new OVAClassifier(factory);
        classifier.train(wineDataset);
    }
}
