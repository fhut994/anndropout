/**
 * Copyright 2012 Neuroph Project http://neuroph.sourceforge.net
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

/**
 *
 * @author Ivana Lukic
 */


public class HayesRoth  {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {

        double momentum = 0.6;
        int numTrials = 10;
        double dropOut = 0.0;

        System.out.println("After a total of " + numTrials + " trials each");
        System.out.println("Average iterations taken to converge to 0.01 error (momentum = " + momentum + "):");


        while (dropOut < 1.0) {
            int iterations = 0;
            ArrayList<Future<Integer>> trials = new ArrayList<Future<Integer>>();
            ExecutorService es = Executors.newFixedThreadPool(10);
            for (int i = 0; i < numTrials; i++) {

                Future<Integer> trial = es.submit(new Trial(dropOut, momentum));
                trials.add(trial);

            }
            for (int i = 0; i < numTrials; i++) {
                iterations += trials.get(i).get();
            }
            es.shutdown();

            iterations /= numTrials;
            System.out.println("DropOut " + dropOut*100 + "%: " + iterations);
            dropOut += 0.01;
        }



    }


    public static void testHayesRoth(NeuralNetwork nnet, DataSet dset) {

        for (DataSetRow trainingElement : dset.getRows()) {

            nnet.setInput(trainingElement.getInput());
            nnet.calculate();
            double[] networkOutput = nnet.getOutput();
            System.out.print("Input: " + Arrays.toString(trainingElement.getInput()));
            System.out.print("Expected: " + Arrays.toString(trainingElement.getDesiredOutput()));
            System.out.println(" Output: " + Arrays.toString(networkOutput));
        }

    }
}
