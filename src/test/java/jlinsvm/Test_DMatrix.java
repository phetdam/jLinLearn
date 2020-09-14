package jlinsvm;

import java.util.Random;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import jlinsvm.DMatrix;
import static jlinsvm.Utils.*;

class Test_DMatrix {

    // X and y matrices, deterministic and stochastic
    double X_det[][];
    double X_rand[][];
    double y_det[];
    double y_rand[];

    /**
     * Create small dummy data set deterministically
     */
    @BeforeAll
    static void new_data_det() {

    }

    /**
     * Test that seeded indice selection works as intended.
     */
    @Test
    void small_det_mat() {
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