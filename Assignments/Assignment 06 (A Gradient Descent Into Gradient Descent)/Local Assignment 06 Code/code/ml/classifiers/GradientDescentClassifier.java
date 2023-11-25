// Munir Vafai
// Collins Kariuki
package ml.classifiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Set;
import java.util.Random;
import java.math.BigDecimal;

import ml.data.DataSet;
import ml.data.Example;

/**
 * Gradient descent classifier allowing for two different loss functions and
 * three different regularization settings.
 * 
 * @author dkauchak
 *
 */
public class GradientDescentClassifier implements Classifier {
	// constants for the different surrogate loss functions
	public static final int EXPONENTIAL_LOSS = 0;
	public static final int HINGE_LOSS = 1;
	public int lossType;

	// constants for the different regularization parameters
	public static final int NO_REGULARIZATION = 0;
	public static final int L1_REGULARIZATION = 1;
	public static final int L2_REGULARIZATION = 2;
	public int regularizationType;

	public double lambda;
	public double eta;

	protected HashMap<Integer, Double> weights; // the feature weights
	protected double b = 0; // the intersect weight

	private int iterations = 50;

	public GradientDescentClassifier() {
		lossType = EXPONENTIAL_LOSS;
		regularizationType = NO_REGULARIZATION;
		lambda = 0.01;
		eta = 0.01;
	}

	public void setLoss(int lossType) {
		if (lossType == EXPONENTIAL_LOSS || lossType == HINGE_LOSS)
			this.lossType = lossType;
		else
			throw new IllegalArgumentException("Invalid loss type");
	}

	public void setRegularization(int regularizationType) {
		if (regularizationType == NO_REGULARIZATION || regularizationType == L1_REGULARIZATION
				|| regularizationType == L2_REGULARIZATION)
			this.regularizationType = regularizationType;
		else
			throw new IllegalArgumentException("Invalid loss type");
	}

	public void setLambda(double lambda) {
		this.lambda = lambda;
	}

	public void setEta(double eta) {
		this.eta = eta;
	}

	/**
	 * Get a weight vector over the set of features with each weight
	 * set to 0
	 * 
	 * @param features the set of features to learn over
	 * @return
	 */
	protected HashMap<Integer, Double> getZeroWeights(Set<Integer> features) {
		HashMap<Integer, Double> temp = new HashMap<Integer, Double>();

		for (Integer f : features) {
			temp.put(f, 0.0);
		}

		return temp;
	}

	/**
	 * Initialize the weights and the intersect value
	 * 
	 * @param features
	 */
	protected void initializeWeights(Set<Integer> features) {
		weights = getZeroWeights(features);
		b = 0;
	}

	/**
	 * Set the number of iterations the perceptron should run during training
	 * 
	 * @param iterations
	 */
	public void setIterations(int iterations) {
		this.iterations = iterations;
	}

	/**
	 * Train the perceptron on the given data
	 * The labels for the data should be {-1,1}
	 * 
	 * @param data
	 */
	public void train(DataSet data) {
		initializeWeights(data.getAllFeatureIndices());

		ArrayList<Example> training = (ArrayList<Example>) data.getData().clone();
		ArrayList<Double> losses = new ArrayList<>();

		for (int it = 0; it < iterations; it++) {
			Collections.shuffle(training);

			for (Example e : training) {
				// print out the current example
				System.out.println("The example we're looking at " + e);

				double label = e.getLabel();
				double prediction = getDistanceFromHyperplane(e, weights, b);
				System.out.println("The prediction gotten is " + prediction);
				for (Integer featureIndex : e.getFeatureSet()) {
					double oldWeight = weights.get(featureIndex);
					double featureValue = e.getFeature(featureIndex);
					double newWeight = 0;
					double regularization = 0;
					if (regularizationType == L1_REGULARIZATION) {
						regularization = Math.signum(oldWeight);
					} else if (regularizationType == L2_REGULARIZATION) {
						regularization = oldWeight;
					}

					if (lossType == EXPONENTIAL_LOSS) {
						newWeight = oldWeight
								+ eta * (featureValue * label * Math.exp(-label * prediction)
										- lambda * regularization);
					} else if (lossType == HINGE_LOSS) {
						double yyPrime = label * prediction;
						double hingeLoss = yyPrime < 1 ? 1 : 0; // c = 1[yy' < 1]
						newWeight = oldWeight + eta * ((featureValue * label * hingeLoss) - (lambda * regularization));
						System.out.println("The new weight " + (featureIndex + 1) + " is " + newWeight);
					}
					weights.put(featureIndex, newWeight);
					// print out the weights
					System.out.println("The weights after the update " + weights);
				}

				double biasRegularization = 0;
				if (regularizationType == L1_REGULARIZATION) {
					biasRegularization = Math.signum(b); // r = sign(b) for L1
				} else if (regularizationType == L2_REGULARIZATION) {
					biasRegularization = b; // r = b for L2
				}

				if (lossType == EXPONENTIAL_LOSS) {
					b += eta * (label * Math.exp(-label * prediction) - lambda * biasRegularization);
				} else if (lossType == HINGE_LOSS) {
					double yyPrime = label * prediction;
					double hingeLoss = yyPrime < 1 ? 1 : 0;
					b += eta * ((label * hingeLoss) - (lambda * biasRegularization));
				}
				// print out the bias
				System.out.println("The bias after the update " + b);
			}

			double currentLoss = getTotalLoss(training);
			losses.add(currentLoss);
		}
	}

	@Override
	public double classify(Example example) {
		return getPrediction(example);
	}

	@Override
	public double confidence(Example example) {
		return Math.abs(getDistanceFromHyperplane(example, weights, b));
	}

	/**
	 * Get the prediction from the current set of weights on this example
	 * 
	 * @param e the example to predict
	 * @return
	 */
	protected double getPrediction(Example e) {
		return getPrediction(e, weights, b);
	}

	/**
	 * Get the prediction from the on this example from using weights w and inputB
	 * 
	 * @param e      example to predict
	 * @param w      the set of weights to use
	 * @param inputB the b value to use
	 * @return the prediction
	 */
	protected static double getPrediction(Example e, HashMap<Integer, Double> w, double inputB) {
		double sum = getDistanceFromHyperplane(e, w, inputB);

		if (sum > 0) {
			return 1.0;
		} else if (sum < 0) {
			return -1.0;
		} else {
			return 0;
		}
	}

	protected static double getDistanceFromHyperplane(Example e, HashMap<Integer, Double> w, double inputB) {
		double sum = inputB;

		for (Integer featureIndex : e.getFeatureSet()) {
			sum += w.get(featureIndex) * e.getFeature(featureIndex);
		}

		return sum;
	}

	/**
	 * Get a string representation of the weights and the bias
	 * for this classifier
	 * 
	 */
	public String toString() {
		StringBuffer buffer = new StringBuffer();

		ArrayList<Integer> temp = new ArrayList<Integer>(weights.keySet());
		Collections.sort(temp);

		for (Integer index : temp) {
			buffer.append(index + ":" + weights.get(index) + " ");
		}

		return buffer.substring(0, buffer.length() - 1) + " b:" + b;
	}

	/**
	 * Get the total loss over the given training set
	 * depending on the current loss function, whether or not
	 * the loss type is exponential or hinge loss
	 * 
	 * @param training
	 * @return the total loss
	 */
	private double getTotalLoss(ArrayList<Example> training) {
		double totalLoss = 0;
		for (Example e : training) {
			double label = e.getLabel();
			double prediction = getDistanceFromHyperplane(e, weights, b);

			if (lossType == EXPONENTIAL_LOSS) {
				totalLoss += Math.exp(-label * prediction);
			} else if (lossType == HINGE_LOSS) {
				totalLoss += Math.max(0, 1 - label * prediction);
			}
		}
		return totalLoss;
	}

	public static void main(String[] args) {
		GradientDescentClassifier g = new GradientDescentClassifier();
		DataSet data = new DataSet("code/ml/data/testing.csv", DataSet.CSVFILE);
		g.setIterations(1);
		g.setLoss(HINGE_LOSS);
		g.train(data);
		System.out.println(g.toString());
	}
}
