package jlinlearn.loss_functions;

import jlinlearn.loss_functions.LossFunction;

/**
 * Modified Huber loss function, for use in classification. Labels should be
 * either (@code -1) or {@code +1} for binary classification. This has a
 * similar profile to the squared hinge loss, i.e. for large negative margins,
 * the reported error will be higher than that given by the hinge loss.
 */
public class ModifiedHuberLoss implements LossFunction {
    
    @Override
    public double evaluate(double y, double y_hat) {
        // compute margin
        double margin = y * y_hat;
        // if (incorrect) margin is not too big, act like squared hinge loss
        if (margin <= -1) {
            return Math.pow(Math.max(0, 1 - margin), 2);
        }
        // else act linearly if the margin is incorrect and > 1 in magnitude
        return -4 * margin;
    }

}
