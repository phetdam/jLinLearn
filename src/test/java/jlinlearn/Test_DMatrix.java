package jlinlearn;

import java.util.Random;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import jlinlearn.DMatrix;
import static jlinlearn.Utils.*;

class Test_DMatrix {

    // random gaussian and uniform distribution matrices, shape (150, 10)
    private static double X_gauss[][];
    private static double X_unif[][];
    // hastie targets that go with X_gauss
    private static double y_hastie[];
    // friedman (1) targets that go with X_unif
    private static double y_friedman[];

    /**
     * Initialize {@code X_gauss}, {@code X_unif}, {@code y_hastie},
     * {@code y_friedman} with values. Uses fixed seed.
     */
    @BeforeAll
    static void initMatrices() {
        final long seed = 7;
        Random rng = new Random(seed);
        // make random gaussian matrix and random uniform matrix
        X_gauss = gaussianMatrix(150, 10, rng);
        X_unif = uniformMatrix(150, 10, rng);
        // make hastie targets and friedman (1) targets with unit gaussian noise
        y_hastie = clsHastieTargets(X_gauss);
        y_friedman = regFriedman1Targets(X_unif, 1, rng);
    }

    /**
     * Test that seeded index selection works as intended.
     */
    @Test
    void testIndexSelection() {
        // new double arrays
        double X[][] = {
            {0, 1, 2, 3, 1},
            {2, 0, 1, 2, 1},
            {0, 1, 2, 0, 0},
            {3, 1, 2, 3, 3}
        };
        double y[] = {0.2, 0.1, 0.1, 0.3};
        // new random number generator
        Random rng = new Random(7);
        // instantiate DMatrix; need vfrac == 0.5 or else n_val == 0
        DMatrix data = new DMatrix(X, y, rng, 0.5);
        // check number of training and validation points + dims
        assertEquals(2, data.n_train);
        assertEquals(2, data.n_val);
        assertEquals(5, data.n_dims);
        // check that indices chosen for training and validation are correct;
        // val data should be first two rows and train data should be last two.
        for (int i = 0; i < data.n_train; i++) {
            assertEquals(y[i], data.get_y_val()[i]);
            for (int j = 0; j < data.n_dims; j++) {
                assertEquals(X[i][j], data.get_X_val()[i][j]);
            }
        }
        for (int i = 0; i < data.n_val; i++) {
            assertEquals(y[i + 2], data.get_y_train()[i]);
            for (int j = 0; j < data.n_dims; j++) {
                assertEquals(X[i + 2][j], data.get_X_train()[i][j]);
            }
        }
    }
}