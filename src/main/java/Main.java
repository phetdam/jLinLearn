/**
 * Main entry point. Test classfication on hastie targets.
 * @author Derek Huang
 */

import java.util.Random;
import java.util.function.DoubleFunction;

import jlinlearn.DMatrix;
import static jlinlearn.Utils.*;

public class Main {
    
    public static void main(String[] args) {
        // number of observations and number of features
        int n_rows = 1500;
        int n_cols = 13;
        // new random number generator for reproducibility
        Random rng = new Random(7);
        // create new gaussian matrix and hastie classification targets
        double X[][] = gaussianMatrix(n_rows, n_cols, rng);
        double y[] = clsHastieTargets(X);
        // instantiate DMatrix; use 20% for validation
        DMatrix data = new DMatrix(X, y, rng);
        System.out.printf("%s\n", data.toString());
        // data preview (only first few elements)
        for (int i = 0; i < 5; i++) {
            System.out.printf("%-4d  ", i);
            for (int j = 0; j < Math.min(data.n_dims, 5); j++) {
                System.out.printf("%13.6e  ", data.get_X_train()[i][j]);
            }
            System.out.printf("---  %2d\n", (int) data.get_y_train()[i]);
        }
    }

}