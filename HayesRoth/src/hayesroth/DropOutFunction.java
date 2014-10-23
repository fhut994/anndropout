/**
 * Copyright 2010 Neuroph Project http://neuroph.sourceforge.net
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
 *  
 */

package hayesroth;

import java.io.Serializable;

import org.neuroph.core.transfer.TransferFunction;
import org.neuroph.util.Properties;
import org.neuroph.core.Layer;
import org.neuroph.core.Neuron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.learning.MomentumBackpropagation;

/**
 * <pre>
 * Sigmoid neuron transfer function.
 *
 * output = 1/(1+ e^(-slope*input))
 * </pre>
 * @author Zoran Sevarac <sevarac@gmail.com>
 */
public class DropOutFunction extends TransferFunction implements Serializable {
    /**
     * The class fingerprint that is set to indicate serialization
     * compatibility with a previous version of the class.
     */
    private static final long serialVersionUID = 2L;

    /**
     * The slope parametetar of the sigmoid function
     */
    private double slope = 1d;

    /**
     * Creates an instance of Sigmoid neuron transfer function with default
     * slope=1.
     */
    public DropOutFunction() {
    }

    TransferFunction wrappedFunction;
    double dropPercent;

    /**
     * Creates an instance of Sigmoid neuron transfer function with the
     * specified properties.
     * @param wrappedFunction properties of the sigmoid function
     */
    public DropOutFunction(TransferFunction wrappedFunction, double dropPercent) {
        this.wrappedFunction = wrappedFunction;
        this.dropPercent = dropPercent;
    }


    @Override
    public double getOutput(double net) {
        // pins the output to always zero on dropout
        if (Math.random() < dropPercent) {
            return 0.0; // drop out
        }
        return wrappedFunction.getOutput(net);
    }

    @Override
    public double getDerivative(double net) { // remove net parameter? maybe we dont need it since we use cached output value
        return wrappedFunction.getDerivative(net);
    }

}
