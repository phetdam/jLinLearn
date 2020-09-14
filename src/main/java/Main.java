/**
 * Main entry point. Test
 * @author Derek Huang
 */

import java.util.Random;

import jlinsvm.DMatrix;

public class Main {
    
    public static void main(String[] args) {
        // new double arrays
        double X[][] = {
            {0, 1, 2, 3},
            {2, 0, 1, 2},
            {0, 1, 2, 0},
            {3, 1, 2, 3}
        };
        System.out.println(X.length);
        double y[] = {0.2, 0.1, 0.1, 0.3};
        // instantiate DMatrix
        Random rng = new Random(7);
        DMatrix data = new DMatrix(X, y, rng, 0.5);
        System.out.printf("%s\n", data.toString());
    }
}