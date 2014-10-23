package hayesroth;

import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.TrainingSetImport;
import org.neuroph.util.TransferFunctionType;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.Callable;

/** 
 * Created by fhut994 on 22/10/2014.
 */
public class Trial implements Callable<Trial.TrialResult> {


    public class TrialResult {

        public int Iterations;
        public double TrainingError;
        public double TestError;
    }

    public double dropOut;
    public double momentum;

    public Trial(double dropOut, double momentum) {
        this.dropOut = dropOut;
        this.momentum = momentum;
    }


    public TrialResult call() {
        String trainingSetFileName = "HayesRoth.txt";
        int inputsCount = 16;
        int outputsCount = 3;

        // create training set
        DataSet trainingSet = null;
        DataSet testSet = null;

        try {
            trainingSet = TrainingSetImport.importFromFile(trainingSetFileName, inputsCount, outputsCount, ",");
            DataSet[] sampledSets = trainingSet.sample(20);
            testSet = sampledSets[0];
            trainingSet = sampledSets[1];
        } catch (FileNotFoundException ex) {
            System.out.println("File not found!");
        } catch (IOException ex) {
            System.out.println("Error reading file or bad number format!");
        }



        // create multi layer perceptron
        //System.out.println("Creating neural network");
        MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, 16, 30, 3);
        // set learning parametars
        MomentumBackpropagation learningRule = (MomentumBackpropagation) neuralNet.getLearningRule();
        learningRule.setLearningRate(0.3);
        learningRule.setMomentum(momentum);
        learningRule.setMaxError(0.01);


        // this wraps the existing transfer function of the hidden neurons
        // in a special dropout function
        Layer hiddenLayer = neuralNet.getLayerAt(1);

        if (dropOut > 0) {
            for (Neuron item : hiddenLayer.getNeurons()) {
                item.setTransferFunction(new DropOutFunction(item.getTransferFunction(), dropOut));
            }
        }
        neuralNet.learn(trainingSet);

        TrialResult result = new TrialResult();
        result.Iterations = learningRule.getCurrentIteration();
        result.TrainingError = test(neuralNet, trainingSet); // learningRule.getTotalNetworkError();
        result.TestError = test(neuralNet, testSet);

        return result;
    }

    // calculates mean squared error
    public double test(NeuralNetwork nnet, DataSet dset) {


        double sum = 0;
        for (DataSetRow trainingElement : dset.getRows()) {

            nnet.setInput(trainingElement.getInput());
            nnet.calculate();
            double err = calculateOutputError(trainingElement.getDesiredOutput(), nnet.getOutput());
            sum += err;
        }

        return sum  / dset.size();

    }

    // calculates squared error
    protected double calculateOutputError(double[] desiredOutput, double[] output) {

        double sum = 0;
        for (int i = 0; i < output.length; i++) {
            double err = desiredOutput[i] - output[i];
            sum += (err * err) * 0.5; // taken from meansquarederror.java
        }

        return sum;
    }
}