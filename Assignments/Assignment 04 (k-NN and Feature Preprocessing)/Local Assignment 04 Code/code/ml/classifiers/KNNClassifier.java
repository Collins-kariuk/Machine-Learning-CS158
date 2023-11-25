package ml.classifiers;

import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.Comparator;

import ml.data.DataSet;
import ml.data.Example;

/**
 * Knn classifier
 * 
 * @author Munir Vafai, Collins Kariuki
 *
 */
public class KNNClassifier implements Classifier {
    private DataSet data;
    private int k;

    public KNNClassifier() {
        k = 3;
    }

    public void setK(int newK) {
        k = newK;
    }

    @Override
    public void train(DataSet data) {
        this.data = data;
    }

    @Override
    public double classify(Example example) {
        ArrayList<Example> examples = data.getData();
        ArrayList<SimpleEntry<Double, Example>> distances = getDistances(example, examples);

        // find k nearest neighbors
        // Sort this list by the double values, so we can return the examples associated
        // with the k smallest distances
        distances.sort(Comparator.comparingDouble(SimpleEntry::getKey));
        ArrayList<Example> nearestKExamples = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            nearestKExamples.add(distances.get(i).getValue());
        }

        // find the most common label among the k nearest neighbors
        int labelSum = 0;
        for (Example e : nearestKExamples) {
            labelSum += e.getLabel();
        }
        if (labelSum > 0)
            return 1;
        else
            return -1;
    }

    /*
     * Calculate the distance between an example and all other examples
     */
    private ArrayList<SimpleEntry<Double, Example>> getDistances(Example example, ArrayList<Example> examples) {
        ArrayList<SimpleEntry<Double, Example>> distances = new ArrayList<>();

        // Calculate all distances
        for (int i = 0; i < examples.size(); i++) {
            Example e = examples.get(i);
            double distance = 0;
            for (int j = 0; j < e.getFeatureSet().size(); j++) {
                distance += Math.pow(e.getFeature(j) - example.getFeature(j), 2);
            }
            distance = Math.sqrt(distance);
            distances.add(new SimpleEntry<>(distance, e));
        }
        return distances;
    }

    public static void main(String[] args) {
        DataSet data = new DataSet("data/titanic-train.real.csv");
        // DataSet data = new DataSet("data/default.csv");
        KNNClassifier knn = new KNNClassifier();
        knn.setK(1);
        knn.train(data);
        // Example e = data.getData().get(0);
        // double classification = knn.classify(e);
        // System.out.println(e.toString());
        // System.out.println(classification);

        double averageAcc = 0.0;
        for (int i = 0; i < 10; i++) {
            double numCorrect = 0.0;
            for (Example example : data.getData()) {
                if (knn.classify(example) == example.getLabel()) {
                    numCorrect += 1;
                }
            }
            averageAcc += numCorrect / data.getData().size();
        }
        System.out.println("averageAcc: " + averageAcc / 10);
    }
}
