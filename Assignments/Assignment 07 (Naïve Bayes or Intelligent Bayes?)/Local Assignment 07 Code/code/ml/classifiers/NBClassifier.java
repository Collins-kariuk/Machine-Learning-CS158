package ml.classifiers;

import java.util.HashMap;
import java.util.Set;

import ml.data.DataSet;
import ml.data.Example;
import ml.utils.HashMapCounter;
import ml.utils.HashMapCounterDouble;

public class NBClassifier implements Classifier {
    private HashMapCounterDouble<Double> labelProbabilities = new HashMapCounterDouble<Double>();
    private HashMap<Double, HashMapCounter<Integer>> labelFeatureCounters = new HashMap<>();
    
    private double lambda = 0.001;
    private boolean useOnlyPositiveFeatures = false;
    private boolean useSmoothing = true; 
    private Set<Integer> allFeatureIndices;


    // Zero parameter constructor
    public NBClassifier() {}

    public void setLambda(double lambda) {
        this.lambda = lambda;
    }

    public void setUseOnlyPositiveFeatures(boolean onlyPositive) {
        this.useOnlyPositiveFeatures = onlyPositive;
    }

    public void setUseSmoothing(boolean useSmoothing) {
        this.useSmoothing = useSmoothing;
    }

    public void train(DataSet data) {
        labelProbabilities.clear();
        labelFeatureCounters.clear();
        int totalExamples = 0;
        allFeatureIndices = data.getAllFeatureIndices();
    
        HashMapCounter<Double> labelCounts = new HashMapCounter<>();
    
        // Loop through examples to gather counts
        for (Example ex : data.getData()) {
            double label = ex.getLabel();
            totalExamples++;
    
            // Increment count of this label
            labelCounts.increment(label);
    
            // Create a new feature counter for this label if it doesn't exist
            if (!labelFeatureCounters.containsKey(label)) {
                labelFeatureCounters.put(label, new HashMapCounter<>());
            }
    
            // Update feature counts for this label
            for (Integer feature : allFeatureIndices) {
                if (ex.getFeatureSet().contains(feature)) {
                    labelFeatureCounters.get(label).increment(feature);
                }
            }
        }
    
        // Calculate log probabilities for labels
        for (double label : labelCounts.keySet()) {
            labelProbabilities.put(label, Math.log((double) labelCounts.get(label) / totalExamples));
        }
    }
    

    // Here we either implement smoothing or don't
    private double calculateProb(int featureIndex, double label) {
        double smoother = useSmoothing ? lambda : 0.0; // Apply smoothing if set
    
        // Get feature counts for the label
        HashMapCounter<Integer> featureCounter = labelFeatureCounters.get(label);
    
        // Add smoother to numerator and denominator to avoid zero probability
        double numerator = featureCounter.get(featureIndex) + smoother;
        double denominator = labelProbabilities.size() + allFeatureIndices.size() * smoother;
        
        return numerator / denominator;
    }

    /**
     * Computes the log probability of an example given a class label. We either calculate using only positives or negatives and positives
     * 
     * @param ex The example.
     * @param label The class label.
     * @return The log probability of the label.
     */
    public double getLogProb(Example ex, double label) {
        double logProb = labelProbabilities.get(label); // Start with log(P(Y))
        
        
        for (int feature : allFeatureIndices) {
            
            double prob = calculateProb(feature, label);
            double featureValue = ex.getFeature(feature);
    
            if (featureValue > 0) {
                
                logProb += Math.log10(prob); // If feature is present and positive
            } else {
                if (!useOnlyPositiveFeatures) {
                    logProb += Math.log10(1.0 - prob); // If feature is absent (or negative)
                }
            }
        }
        return logProb;
    }
    

    /**
     * Computes the probability of a feature given a class label.
     *
     * @param featureIndex The index of the feature.
     * @param label The class label.
     * @return The log probability of the feature given the label.
     */
    public double getFeatureProb(int featureIndex, double label) {
        return calculateProb(featureIndex, label);
    }

    /**
     * Returns the probability of the predicted class
     *
     * @param ex The example
     * @return confidence
     */
    public double confidence(Example ex) {
        double bestLabel = classify(ex);
        return getLogProb(ex, bestLabel);
    }

    /**
     * Classifies the given example using Naive Bayes.
     *
     * @param example The example to classify.
     * @return The predicted class label (either 1.0 for positive or 0.0 for negative).
     */
    @Override
    public double classify(Example example) {
        double highestLogProb = Double.NEGATIVE_INFINITY;
        double bestLabel = -1.0;
    
        // Loop through all possible labels to find the one with the highest log probability
        for (double label : labelProbabilities.keySet()) {
            double logProb = getLogProb(example, label);
            
            if (logProb > highestLogProb) {
                highestLogProb = logProb;
                bestLabel = label;
            }
        }
    
        return bestLabel;
    }

    public static void main(String[] args) {
        // Test NBClassifier
        NBClassifier nb = new NBClassifier();
        nb.setLambda(0.005);
        nb.setUseOnlyPositiveFeatures(true);
        nb.setUseSmoothing(true);
        // Load Wine DataSet
        DataSet data = new DataSet("data/wines.train", DataSet.TEXTFILE); // Find dataset
        nb.train(data);
        // example 1
        Example ex = data.getData().get(2);
        System.out.println(ex.getLabel());
        System.out.println(nb.classify(ex));
    }
}
