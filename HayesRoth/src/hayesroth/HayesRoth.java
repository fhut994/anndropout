
package hayesroth;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.TrainingSetImport;
import org.neuroph.util.TransferFunctionType;

import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.slf4j.Logger;

/**
 *
 * @author Freddy Hutchinson
 * Group: fhut994/vnai522/jcur884
 * 
 */


public class HayesRoth  {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        double momentum = 0.6;
        int numTrials = 10;
        double dropOut = 0.0;

        // force logger to rear its ugly head
        MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, 16, 30, 3);

        System.out.println("Trials\tMomentum\tDropOut\tAvgIterations\tMaxIterations\tAvgTrainingError\tAvgTestError");

        while (dropOut < 1.0) {
            int iterations = 0;
            int maxIterations = Integer.MIN_VALUE;
            double trainingError = 0;
            double testingError = 0;
            ArrayList<Future<Trial.TrialResult>> trials = new ArrayList<Future<Trial.TrialResult>>();
            ExecutorService es = Executors.newFixedThreadPool(10);

            // map
            for (int i = 0; i < numTrials; i++) {

                Future<Trial.TrialResult> trial = es.submit(new Trial(dropOut, momentum));
                trials.add(trial);

            }

            // reduce
            for (int i = 0; i < numTrials; i++) {
                Trial.TrialResult res =trials.get(i).get();

                if (res.Iterations > maxIterations)
                {
                    maxIterations = res.Iterations;
                }
                iterations += res.Iterations;
                trainingError += res.TrainingError;
                testingError += res.TestError;
            }

            es.shutdown();

            // avg
            iterations /= numTrials;
            trainingError /= numTrials;
            testingError /= numTrials;

            System.out.println(numTrials + "\t" + momentum + "\t" + dropOut + "\t" + iterations + "\t" + maxIterations + "\t" + trainingError + "\t" + testingError);
            dropOut += 0.01;
        }



    }
}
