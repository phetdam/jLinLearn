package jlinlearn.loss_functions;

/**
 * Loss function interface since there are no function pointers in Java.
 */
public interface LossFunction {
    
    /**
     * Evaluate loss function on response {@code y}, predicted {@code y_hat}.
     */
    public double evaluate(double y, double y_hat);
}