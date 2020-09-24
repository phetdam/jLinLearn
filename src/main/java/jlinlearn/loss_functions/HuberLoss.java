package jlinlearn.loss_functions;

import jlinlearn.loss_functions.LossFunction;

/**
 * Huber loss function. Here we use {@code delta == 1} so that the function is
 * scaled similarly to the squared-error loss. Regression error function.
 */
public class HuberLoss implements LossFunction {

    @Override
    public double evaluate(double y, double y_hat) {
        // get absolute difference between y and y_hat
        double abs_delta = Math.abs(y - y_hat);
        // if |abs_delta| <= 1, then act as halved squared error function
        if (abs_delta <= 1) {
            return 0.5 * Math.pow(abs_delta, 2);
        }
        // else act like absolute error translated down by 1 / 2
        return abs_delta - 0.5;
    }
}