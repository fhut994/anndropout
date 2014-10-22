package hayesroth;

import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.TrainingSetImport;
import org.neuroph.util.TransferFunctionType;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.concurrent.Callable;

/**
 * Created by swit012 on 22/10/2014.
 */
public class Trial implements Callable<LearningRule> {

    public double dropOut;
    public double momentum;

    public Trial(double dropOut, double momentum) {
        this.dropOut = dropOut;
        this.momentum = momentum;
    }


    public LearningRule call() {
        String trainingSetFileName = "HayesRoth.txt";
        int inputsCount = 16;
        int hiddenLayers = 20;
        int outputsCount = 3;

//        String trainingSetFileName = "NormalisedWinesDataSet.txt";
//        int inputsCount = 13;
//        int hiddenLayers = 21;
//        int outputsCount = 3;

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
        //System.out.println("Creating neural network");
        MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(TransferFunctionType.SIGMOID, inputsCount, hiddenLayers, outputsCount);
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


        System.out.print(".");
        //learningRule.getMaxError();
        return learningRule;
    }
}