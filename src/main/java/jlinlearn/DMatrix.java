package jlinlearn;

import java.util.*;

import static jlinlearn.Utils.randomSubset;

/**
 * Implementation for a thin data set class.
 */
public class DMatrix {

    // total rows + number of rows in training and validation data sets. note
    // we use public final for scalars to remove unnecessary getters.
    public final int n_tot;
    public final int n_train;
    public final int n_val;
    // number of input dimensions
    public final int n_dims;
    // training and validation input and output matrix/vectors. these cannot be
    // public final since the elements can still be modified.
    public final double _X_train[][];
    public final double _X_val[][];
    public final double _y_train[];
    public final double _y_val[];

    /**
     * Convenience constructor for DMatrix. Splits the incoming data into
     * training and test data sets in a non-deterministic fashion, with ~20% of
     * the rows allocated to the validation data set.
     * 
     * @param X Input matrix, dimension (n_obs, n_dims)
     * @param y Output vector, dimension (n_obs,)
     */
    public DMatrix(final double X[][], final double y[]) {
        this(X, y, null);
    }

    /**
     * Constructor for DMatrix with seedable Random instance, ~20% validation.
     * 
     * @param X Input matrix, dimension (n_obs, n_dims)
     * @param y Output vector, dimension (n_obs,)
     * @param rng java.util.Random instance for reproducibility across calls.
     */
    public DMatrix(final double X[][], final double y[], Random rng) {
        this(X, y, rng, 0.2);
    }
    
    /**
     * Constuctor for DMatrix with seedable Random instance and variable
     * percentage of data allocated to the validation data set.
     * 
     * Set rng to null to create a new Random instance internally.
     * 
     * @param X Input matrix, dimension (n_obs, n_dims)
     * @param y Output vector, dimension (n_obs,)
     * @param rng java.util.Random instance for reproducibility across calls.
     * @param vfrac Fraction of data to use for validation in (0, 1).
     */
    public DMatrix(final double X[][], final double y[], Random rng,
        final double vfrac) {
        // error checking
        if (X == null) {
            throw new NullPointerException("X is null");
        }
        if (y == null) {
            throw new NullPointerException("y is null");
        }
        if (X.length != y.length) {
            throw new InputMismatchException("X and y rows must be equal");
        }
        if (X.length == 0) {
            throw new InputMismatchException("X, y must have nonzero length");
        }
        if ((vfrac >= 1) || (vfrac <= 0)) {
            throw new InputMismatchException("vfrac must be in (0, 1)");
        }
        // if rng == null, then instantiate a new Random instance
        if (rng == null) {
            rng = new Random();
        }
        // save total number of data points and dimensions
        n_tot = X.length;
        n_dims = X[0].length;
        // select number of rows to use in validation set. note that the indices
        // returned by random_subset are going to be in ascending order.
        n_val = (int) (vfrac * n_tot);
        // if this is 0, need to raise an exception
        if (n_val == 0) {
            throw new InputMismatchException("Not enough validation points. " +
                "Please increase the value of vfrac or use a larger data set.");
        }
        // number of rows in training set
        n_train = n_tot - n_val;
        // indices in the validation set and the training set, respectively
        int ixs_val[] = randomSubset(rng, n_tot, n_val);
        int ixs_train[] = new int[n_train];
        /** 
         * select all the other indices not in ixs_val to be in training set.
         * use ci_val to mark the next index in ixs_val that we have not seen
         * yet and use ci_train to mark the next index in ixs_train to fill.
         */ 
        int ci_val = 0;
        int ci_train = 0;
        for (int i = 0; i < n_tot; i++) {
            /**
             * if i == ixs_val[ci_val], don't write to ixs_train (one of the
             * indices in ixs_val), and increment ci_val instead. obviously,
             * ci_val must be less than n_val.
             */
            if ((ci_val < n_val) && (i == ixs_val[ci_val])) {
                ci_val++;
            }
            // else just write i to ixs_train[ci_train] and increment ci_train
            else if (ci_train < n_val) {
                ixs_train[ci_train] = i;
                ci_train++;
            }
        }
        // initialize _X_train, _X_val, _y_train, _y_val and copy values
        _X_train = new double[n_train][n_dims];
        _X_val = new double[n_val][n_dims];
        _y_train = new double[n_train];
        _y_val = new double[n_val];
        for (int i = 0; i < n_train; i++) {
            // current row index from training indices
            int ci = ixs_train[i];
            // copy y training data
            _y_train[i] = y[ci];
            // copy X training data
            for (int j = 0; j < n_dims; j++) {
                _X_train[i][j] = X[ci][j];
            }
        }
        for (int i = 0; i < n_val; i++) {
            // current row index from validation indices
            int ci = ixs_val[i];
            _y_val[i] = y[ci];
            for (int j = 0; j < n_dims; j++) {
                _X_val[i][j] = X[ci][j];
            }
        }
    }

    /**
     * toString method for DMatrix. reports n_train, n_val, and n_dims.
     */
    public String toString() {
        return String.format("DMatrix(n_train = %d, n_val = %d, n_dims = %d)",
            n_train, n_val, n_dims);
    }

    /** Getters **/

    public double[][] get_X_train() {
        return _X_train;
    }

    public double[][] get_X_val() {
        return _X_val;
    }

    public double[] get_y_train() {
        return _y_train;
    }

    public double[] get_y_val() {
        return _y_val;
    }
}