
package br.com.inteligenciaartificial.algoritmos;

import br.com.inteligenciaartificial.algoritmos.math.Matrix;
import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

/**
 * Unit test for simple App.
 */
public class MatrixTest extends TestCase {
    /**
     * Create the test case
     *
     * @param testName
     *        name of the test case
     */
    public MatrixTest(final String testName) {
        super(testName);
    }

    /**
     * @return the suite of tests being tested
     */
    public static Test suite() {
        return new TestSuite(MatrixTest.class);
    }

    public void testEqualMatrix() {
        final Matrix m1 = generateMatrix();
        final Matrix m2 = generateMatrix();
        assertTrue("Both matrixes must have the same elements", m1.equals(m2));
    }

    public void testNotEqualMatrix() {
        final Matrix m1 = generateMatrix();
        final Matrix m2 = generateMatrix();
        m2.set(0, 0, 10);

        assertFalse("All the matrixes doesnt have the same elements", m1.equals(m2));
    }

    public void testNullMatrix() {
        final Matrix m1 = generateMatrix();
        final Matrix m2 = null;

        assertFalse("The second matrix is null and must be not ", m1.equals(m2));
    }

    public void testMatrixDifferentOrder() {
        final Matrix m1 = generateMatrix();
        final Matrix m2 = new Matrix(3, 4);

        assertFalse("The second matrix has different order", m1.equals(m2));
    }

    public void testTransposeMatrix() {
        final Matrix m1 = generateMatrix().transpose();
        final Matrix m2 = generateTransposeMatrix();

        assertTrue("Both matrixes must have the same elements", m1.equals(m2));
    }

    public void testSumMatrix() {
        final Matrix m1 = generateMatrix();
        final Matrix m2 = generateMatrix();

        final Matrix sum = generateDoubleValuedMatrix();

        assertTrue("Both matrixes must have the same elements", sum.equals(m1.sum(m2)));
    }

    public void testSubtractionMatrix() {
        final Matrix m1 = generateMatrix();
        final Matrix m2 = generateMatrix();

        final Matrix sub = new Matrix(m1.getRowNum(), m2.getColNum());

        assertTrue("Both matrixes must have the same elements", sub.equals(m1.sub(m2)));
    }

    public void testCopyMatrix() {
        final Matrix m1 = generateMatrix();
        final Matrix m2 = m1.copy();

        assertTrue("Both matrixes must have the same elements", m1.equals(m2));
    }

    public void testDotMatrix() {
        final Matrix m1 = generateMatrix();
        final Matrix square = generateSquareValuedMatrix();

        assertTrue("Both matrixes must have the same elements", square.equals(m1.dot(m1)));
    }

    public void testInitRandomMatrix() {
        final Matrix m1 = generateMatrix().initRandom();

        assertTrue("All matrix elements must be in [0,1] interval", m1.test(z -> z >= 0 && z <= 1));
    }

    private Matrix generateMatrix() {
        final Matrix m = new Matrix(2, 3);
        m.set(0, 0, 1);
        m.set(0, 1, 2);
        m.set(0, 2, 3);
        m.set(1, 0, 4);
        m.set(1, 1, 5);
        m.set(1, 2, 6);
        return m;
    }

    private Matrix generateTransposeMatrix() {
        final Matrix m = new Matrix(3, 2);
        m.set(0, 0, 1);
        m.set(1, 0, 2);
        m.set(2, 0, 3);
        m.set(0, 1, 4);
        m.set(1, 1, 5);
        m.set(2, 1, 6);
        return m;
    }

    private Matrix generateDoubleValuedMatrix() {
        return generateMatrix().apply(z -> 2 * z);
    }

    private Matrix generateSquareValuedMatrix() {
        return generateMatrix().apply(z -> z * z);
    }
}
