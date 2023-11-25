// Munir Vafai
// Collins Kariuki
package ml.classifiers;

import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import ml.data.DataSet;
import ml.data.Example;

/**
 * An AVA classifier that uses a factory to create classifiers for each label.
 */
public class AVAClassifier implements Classifier {
    ClassifierFactory factory;
    ArrayList<SimpleEntry<Double[], Classifier>> classifiers = new ArrayList<>();

    /**
     * Creates a new AVA classifier that uses the given factory to create
     * classifiers for each label.
     * 
     * @param factory
     */
    public AVAClassifier(ClassifierFactory factory) {
        this.factory = factory;
    }

    /** 
     * Trains a classifier for each pair of labels in the dataset. 
     * 
     * @param data The dataset to train on
     */
    public void train(DataSet data) {
        Set<Double> labels = data.getLabels();
        ArrayList<Double> labelList = new ArrayList<>(labels);
        for (int i = 0; i < labelList.size(); i++) {
            double label1 = labelList.get(i);
            for (int j = i + 1; j < labelList.size(); j++) {
                double label2 = labelList.get(j);
                Classifier classifier = factory.getClassifier();
                DataSet newData = filterAndRelabelDataSet(data, label1, label2);
                classifier.train(newData);
                classifiers.add(new SimpleEntry<>(new Double[] { label1, label2 }, classifier));
            }
        }
    }

    /**
     * Filters the dataset to only include examples with label1 or label2
     * @param data The dataset to filter
     * @param label1 The first label to include
     * @param label2 The second label to include    
     * @return A new dataset with only examples with label1 or label2  
     */
    private DataSet filterAndRelabelDataSet(DataSet data, double label1, double label2) {
        DataSet newData = new DataSet(data.getFeatureMap());
        for (Example example : data.getData()) {
            if (example.getLabel() == label1) {
                Example newExample = new Example(example);
                newExample.setLabel(1);
                newData.addData(newExample);
            } else if (example.getLabel() == label2) {
                Example newExample = new Example(example);
                newExample.setLabel(-1);
                newData.addData(newExample);
            }
        }
        return newData;
    }

    /**
     * Classifies the given example by using each classifier to predict the label
     * 
     * @param example The example to classify
     */
    public double classify(Example example) {
        Map<Double, Double> scores = new HashMap<>();

        for (SimpleEntry<Double[], Classifier> entry : classifiers) {
            Double[] labels = entry.getKey();
            Classifier classifier = entry.getValue();
            double y = classifier.classify(example) * classifier.confidence(example);

            // Update scores based on prediction
            double label1Score = scores.getOrDefault(labels[0], 0.0);
            double label2Score = scores.getOrDefault(labels[1], 0.0);

            scores.put(labels[0], label1Score + y);
            scores.put(labels[1], label2Score - y);
        }

        // Return the label with the highest score
        return Collections.max(scores.entrySet(), Map.Entry.comparingByValue()).getKey();
    }

    public double confidence(Example example) {
        return 0;
    }

}
