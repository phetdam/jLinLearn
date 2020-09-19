package jlinlearn;

import java.util.InputMismatchException;
import java.util.Random;

/**
 * Class for various utility methods.
 */
public class Utils {

    /** Functions for generating random input matrices **/
    
    /**
     * Create a random standard normal Gaussian matrix. All elements are i.i.d.
     * 
     * @param n_rows Number of rows in the matrix
     * @param n_cols Number of columns in the matrix
     * @return Random Gaussian double matrix with shape (n_rows, n_cols)
     */
    public static double[][] gaussianMatrix(int n_rows, int n_cols) {
        return gaussianMatrix(n_rows, n_cols, null);
    }

    /**
     * Create a Gaussian matrix, where all elements are i.i.d standard normal.
     * If rng is null, then a new Random instance is created internally.
     * 
     * @param n_rows Number of rows in the matrix
     * @param n_cols Number of columns in the matrix
     * @param rng Random instance for reproducibility across calls.
     * @return Double matrix with shape (n_rows, n_cols)
     */
    public static double[][] gaussianMatrix(
        int n_rows,
        int n_cols,
        Random rng)
    {
        // sanity checking
        if (n_rows <= 0) {
            throw new InputMismatchException("n_rows must be positive");
        }
        if (n_cols <= 0) {
            throw new InputMismatchException("n_cols must be positive");
        }
        // seed random number generator; autoseed if rng == null
        if (rng == null) {
            rng = new Random();
        }
        // allocate new double matrix and fill with Gaussian inputs
        double mat[][] = new double[n_rows][n_cols];
        for (int i = 0; i < n_rows; i++) {
            for (int j = 0; j < n_cols; j++) {
                mat[i][j] = rng.nextGaussian();
            }
        }
        return mat;
    }

    /**
     * Creates a random matrix where elements are i.i.d. uniformly distributed
     * and drawn from the interval [0, 1].
     * 
     * @param n_rows Number of rows in the matrix
     * @param n_cols Number of columns in the matrix
     * @return Double matrix with shape (n_rows, n_cols)
     */
    public static double[][] uniformMatrix(int n_rows, int n_cols) {
        return uniformMatrix(n_rows, n_cols, null);
    }

    /**
     * Creates a matrix where elements are i.i.d uniformly distributed and drawn
     * drawn from the interval [0, 1]. If rng is null, then a new Random
     * instance will be created in the function.
     * 
     * @param n_rows Number of rows in the matrix
     * @param n_cols Number of columns in the matrix
     * @param rng Random instance for reproducibility across calls.
     * @return Double matrix with shape (n_rows, n_cols)
     */
    public static double[][] uniformMatrix(int n_rows, int n_cols, Random rng) {
        // sanity checking
        if (n_rows <= 0) {
            throw new InputMismatchException("n_rows must be positive");
        }
        if (n_cols <= 0) {
            throw new InputMismatchException("n_cols must be positive");
        }
        // seed random number generator; autoseed if rng == null
        if (rng == null) {
            rng = new Random();
        }
        // allocate new double matrix and fill with uniform inputs
        double mat[][] = new double[n_rows][n_cols];
        for (int i = 0; i < n_rows; i++) {
            for (int j = 0; j < n_cols; j++) {
                mat[i][j] = rng.nextDouble();
            }
        }
        return mat;
    }

    /** Functions for generating targets from input matrices **/

    /**
     * Generates binary classification targets using the rule described in
     * Hastie et al. 2009, Example 10.2, from the Elements of Statistical
     * Learning. Resulting vector will be in {-1, 1} ^ X.length.
     * 
     * Assumes the input matrix is not ragged and has standard Gaussian entries.
     * 
     * @param X Input matrix, shape {@code (n_rows, n_cols)}.
     * @return A {@code n_rows} length vector of with only -1s and 1s.
     */
    public static double[] clsHastieTargets(double[][] X) {
        // error checks
        if (X == null) {
            throw new NullPointerException("X is null");
        }
        if (X.length == 0) {
            throw new InputMismatchException("X must have positive length");
        }
        if (X[0].length == 0) {
            throw new InputMismatchException("X must have positive dimension");
        }
        // get number of rows and number of columns
        int n_rows = X.length;
        int n_cols = X[0].length;
        // allocate new y array and return
        double y[] = new double[n_rows];
        for (int i = 0; i < n_rows; i++) {
            // compute sum of the squared row elements
            double rsum = 0;
            for (int j = 0; j < n_cols; j++) {
                rsum = rsum + Math.pow(X[i][j], 2);
            }
            // determine value of y[i] using classification rule
            if (rsum > 9.34) {
                y[i] = 1;
            }
            else {
                y[i] = -1;
            }
        }
        return y;
    }

    /**
     * Generates the regression problem described in [1] and [2]. This method
     * does not add any noise to the generated regression targets.
     * 
     * [1] J. Friedman, Multivariate adaptive regression splines, The Annals of
     *     Statistics 19 (1), pages 1-67, 1991.
     * [2] L. Breiman, Bagging predictors, Machine Learning 24, p 123-140, 1996.
     * 
     * @param X Input matrix, shape (n_rows, n_cols), where n_cols >= 5.
     * @return A vector of regression targets, length n_rows.
     */
    public static double[] regFriedman1Targets(double[][] X) {
        // actually does not matter what value of the seed is since noise == 0
        return regFriedman1Targets(X, 0, null);
    }

    /**
     * Generates the regression problem described in [1] and [2]. If noise > 0,
     * the output vector will be stochastic, with added Gaussian noise.
     * 
     * [1] J. Friedman, Multivariate adaptive regression splines, The Annals of
     *     Statistics 19 (1), pages 1-67, 1991.
     * [2] L. Breiman, Bagging predictors, Machine Learning 24, p 123-140, 1996.
     * 
     * @param X Input matrix, shape (n_rows, n_cols), where n_cols >= 5.
     * @param noise Standard deviation of Gaussian noise to apply to the output.
     * @return A vector of regression targets, length n_rows.
     */
    public static double[] regFriedman1Targets(double[][] X, double noise) {
        return regFriedman1Targets(X, noise, null);
    }

    /**
     * Generates the regression problem described in [1] and [2]. Gaussian
     * noise may be optionally added to the targets.
     * 
     * Assumes the input matrix is not ragged and has entries uniformly
     * distributed over the interval [0, 1]. It must have >=5 input features.
     * 
     * If rng is null, then a Random instance will be created internally.
     * 
     * [1] J. Friedman, Multivariate adaptive regression splines, The Annals of
     *     Statistics 19 (1), pages 1-67, 1991.
     * [2] L. Breiman, Bagging predictors, Machine Learning 24, p 123-140, 1996.
     * 
     * @param X Input matrix, shape (n_rows, n_cols), where n_cols >= 5.
     * @param noise Standard deviation of Gaussian noise to apply to the output.
     * @param rng Random instance for reproducibility across calls.
     * @return A vector of regression targets, length n_rows.
     */
    public static double[] regFriedman1Targets(
        double[][] X, 
        double noise,
        Random rng)
    {
        // error checks
        if (X == null) {
            throw new NullPointerException("X is null");
        }
        if (X.length == 0) {
            throw new InputMismatchException("X must have positive length");
        }
        if (X[0].length == 0) {
            throw new InputMismatchException("X must have positive dimension");
        }
        if (X[0].length < 5) {
            throw new InputMismatchException("X must have at least 5 features");
        }
        if (noise < 0) {
            throw new InputMismatchException("noise must be nonnegative");
        }
        // get number of rows and number of columns
        int n_rows = X.length;
        int n_cols = X[0].length;
        // if noise is greater than 0, initialize the random number generator.
        // again, if rng == null, automatically seed.
        if (noise > 0) {
            if (rng == null) {
                rng = new Random();
            }
        }
        // allocate new y array and return
        double y[] = new double[n_rows];
        for (int i = 0; i < n_rows; i++) {
            // compute target according to regression rule
            y[i] = 10 * Math.sin(Math.PI * X[i][0] * X[i][1]) + 20 * 
                Math.pow(X[i][2], 2) + 10 * X[i][3] + 5 * X[i][4];
            // if noise > 0, also add random innovation to the target
            if (noise > 0) {
                y[i] = y[i] + noise * rng.nextGaussian();
            }
        }
        return y;
    }

    /**
     * Sample indices without replacement from {@code 0, ... n - 1}.
     * 
     * Useful for obtaining k random indices to get a subcollection of an array.
     * By default, the selected indices are in their original ordering.
     * 
     * @param rng Seeded java.util.Random instance
     * @param n Control upper bound of integers, i.e. {@code 0, ... n - 1}
     * @param k Number of integers to select from {@code 0, ... n - 1},
     *     {@code k <= n}
     * @return An int array containing a subset of {@code 0, ... n - 1}
     */
    public static int[] randomSubset(Random rng, final int n, int k) {
        return randomSubset(rng, n, k, false);
    }

    /**
     * Sample indices without replacement from {@code 0, ... n - 1}.
     * 
     * Useful for obtaining k random indices to get a subcollection of an array.
     * 
     * @param rng Seeded java.util.Random instance
     * @param n Control upper bound of integers, i.e. {@code 0, ... n - 1}
     * @param k Number of integers to select from {@code 0, ... n - 1},
     *     {@code k <= n}
     * @param shuffle Whether or not to shuffle the returned indices. If this is
     *     {@code false}, then the elements are in ascending sorted order.
     * @return An int array containing a subset of {@code 0, ... n - 1}
     */
    public static int[] randomSubset(
        Random rng, 
        final int n, 
        int k,
        boolean shuffle)
    {
        // if k > n, error
        if (k > n) {
            throw new InputMismatchException("k must be <= n");
        }
        // indices to return
        int ixs[] = new int[k];
        // total indices selected and total indices already visited
        int ci = 0;
        int ti = 0;
        // random number in [0, 1)
        double u;
        // while total selected < goal
        while (ci < k) {
            // uniformly sample
            u = rng.nextDouble();
            // not really sure what's going on here :(
            if ((n - ti) * u >= (k - ci)) {

            }
            else {
                ixs[ci] = ti;
                ci++;
            }
            // move onto next index
            ti++;
        }
        // perform fisher-yates shuffle to shuffle the indices if indicated
        if (shuffle) {
            for (int i = 0; i < k; i++) {
                // select index from 0, ... i
                int j = rng.nextInt(i + 1);
                // swap elements i and j
                int temp = ixs[i];
                ixs[i] = ixs[j];
                ixs[j] = temp;
            }
            for (int i = 0;i < k; i++) {
                System.out.println(ixs[i]);
            }
        }
        // all done!
        return ixs;
    }
}
