package jlinlearn.loss_functions;

import jlinlearn.loss_functions.LossFunction;

/**
 * Hinge loss function.
 */
public class HingeLoss implements LossFunction {
    
    @Override
    public double evaluate(double y, double y_hat) {
        return Math.max(0, 1 - y * y_hat);
    }

}
