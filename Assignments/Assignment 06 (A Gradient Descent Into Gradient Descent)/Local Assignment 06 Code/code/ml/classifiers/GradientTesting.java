package ml.classifiers;

import ml.data.DataSet;
import ml.data.Example;

import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;

public class GradientTesting extends GradientDescentClassifier {
    private double[] lambdaGrid = { 0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01 };
    private double[] etaGrid = { 0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01 };

    public GradientTesting() {
        super();
    }

    public void setLambdaGrid(double[] lambdaGrid) {
        this.lambdaGrid = lambdaGrid;
    }

    public void setEtaGrid(double[] etaGrid) {
        this.etaGrid = etaGrid;
    }

    public void optimizeParameters(DataSet data) {
        double bestLoss = Double.MAX_VALUE;
        double bestLambda = lambda;
        double bestEta = eta;

        for (double lambdaCandidate : lambdaGrid) {
            for (double etaCandidate : etaGrid) {
                lambda = lambdaCandidate;
                eta = etaCandidate;
                train(data);
                double currentLoss = getTotalLoss(new ArrayList<>(data.getData()));

                if (currentLoss < bestLoss) {
                    bestLoss = currentLoss;
                    bestLambda = lambda;
                    bestEta = eta;
                }
            }
        }

        lambda = bestLambda;
        eta = bestEta;
        train(data);
    }

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
        GradientTesting g = new GradientTesting();
        DataSet data = new DataSet("code/ml/data/testing.csv", DataSet.CSVFILE);

        g.optimizeParameters(data);
        System.out.println("Optimized lambda: " + g.lambda);
        System.out.println("Optimized eta: " + g.eta);
    }
}
