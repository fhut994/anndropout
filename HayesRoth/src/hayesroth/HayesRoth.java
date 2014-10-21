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
import java.util.Arrays;

import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.Weight;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.DataSet;
import org.neuroph.core.learning.DataSetRow;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.TrainingSetImport;
import org.neuroph.util.TransferFunctionType;

/**
 *
 * @author Ivana Lukic
 */

public class HayesRoth implements LearningEventListener {
    static double _randFactor;

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {

        String trainingSetFileName = "HayesRoth.txt";
        int inputsCount = 16;
        int outputsCount = 3;

        System.out.println("Running Sample");
        System.out.println("Using training set " + trainingSetFileName);

        // create training set
        DataSet trainingSet = null;
        try {
            trainingSet = TrainingSetImport.importFromFile(trainingSetFileName, inputsCount, outputsCount, ",");
        } catch (FileNotFoundException ex) {
            System.out.println("File not found!");
        } catch (IOException ex) {
            System.out.println("Error reading file or bad number format!");
        }


        // create multi layer perceptron
        System.out.println("Creating neural network");
        MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, 16, 20, 3);

        // set learning parametars
        MomentumBackpropagation learningRule = (MomentumBackpropagation) neuralNet.getLearningRule();
        learningRule.setLearningRate(0.2);
        learningRule.setMomentum(0.7);
        learningRule.setMaxError(0.01);
        _randFactor = 0.9;

        HayesRoth eventHandle = new HayesRoth();
        learningRule.addListener(eventHandle);

        // learn the training set
        System.out.println("Training neural network...");
        neuralNet.learn(trainingSet);
        System.out.println("Done!");

        // test perceptron
        System.out.println("Testing trained neural network");
        testHayesRoth(neuralNet, trainingSet);

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

    public void handleLearningEvent(LearningEvent l) {
        MomentumBackpropagation rule = (MomentumBackpropagation)l.getSource();
        Layer[] layers = rule.getNeuralNetwork().getLayers();
        Layer hiddenLayer = layers[1];

        // On each presentation of each training case, each hidden unit is randomly omitted from the network with a probability of 0.5
        // From: http://arxiv.org/pdf/1207.0580.pdf


        //System.out.println("i lernd " + rule.getCurrentIteration() + " " + Arrays.toString(hiddenLayer.getNeuronAt(0).getWeights()));

        for (Neuron n : hiddenLayer.getNeurons()) {
            if (Math.random() > _randFactor) {
                for (Weight w : n.getWeights())
                    w.setValue(0.0);
            }
        }

    }
}
