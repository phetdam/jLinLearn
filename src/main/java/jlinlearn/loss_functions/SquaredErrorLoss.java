package jlinlearn.loss_functions;

import jlinlearn.loss_functions.LossFunction;

/**
 * Hinge loss function.
 */
public class SquaredErrorLoss implements LossFunction {
    
    @Override
    public double evaluate(double y, double y_hat) {
        return Math.pow(y - y_hat, 2);
    }

}