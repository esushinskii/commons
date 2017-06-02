package com.spbsu.commons.math.vectors;

import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.random.FastRandom;
import com.spbsu.commons.util.Pair;
import junit.framework.TestCase;

public class MxToolsTest extends TestCase {

  public void testSolveSystemLq() throws Exception {
    final Mx A = new VecBasedMx(2, new ArrayVec(
        1, 2,
        1, 5
    ));
    final Vec b = new ArrayVec(5, 11);
    final Vec x = MxTools.solveSystemLq(A, b);
    assertTrue(VecTools.distance(MxTools.multiply(A, x), b) < 1e-4);
  }

  private final Mx matrix = new VecBasedMx(4, new ArrayVec(
      2, 0.25, 0, 0,
      0.25, 0.0625, 0, 0,
      0, 0, 2, 1.25,
      0, 0, 1.25, 0.8125
  ));

  private final Vec vector = new ArrayVec(2, 0.5, 402, 251.5);

  public void testHouseHolderLQ()
  {
    Mx l = new VecBasedMx(4, 4);
    Mx q = new VecBasedMx(4, 4);
    MxTools.householderLQ(matrix, l, q);
    assertEquals(0, VecTools.distance(matrix, MxTools.multiply(l, MxTools.transpose(q))), 1e-3);
  }

  public void testCholesky()
  {
    Mx l = MxTools.choleskyDecomposition(matrix);
    assertEquals(0, VecTools.distance(matrix, MxTools.multiply(l, MxTools.transpose(l))), 1e-3);
  }

  public void testCholeskySolve()
  {
    Vec answer = MxTools.solveSystemCholesky(matrix, vector);
    assertEquals(0, VecTools.distance(vector, MxTools.multiply(matrix, answer)), 1e-3);
  }

  public void testLQSolve()
  {
    Vec answer = MxTools.solveSystemLq(matrix, vector);
    assertEquals(0, VecTools.distance(vector, MxTools.multiply(matrix, answer)), 1e-3);

  }
  public void testIterativeSolve() {
    Vec answer = MxTools.solveSystemGaussZeildel(matrix, vector);
    assertEquals(0, VecTools.distance(vector, MxTools.multiply(matrix, answer)), 1e-3);
  }

  // tests for the Lanczos algorithm
  public void testLanczosTridiagonalization() {
    FastRandom rng = new FastRandom();

    int test_num = 10;
    double[] err = new double[test_num];

    int n = 1000;

    for (int i = 0; i < test_num; i++) {
      Mx matrix = new VecBasedMx(n, VecTools.fillUniform(new ArrayVec(n * n), rng));
      final Pair<Mx, Mx> pair = MxTools.lanczosTridiagonalization(matrix, matrix.columns());
      Mx tPart = pair.first;
      Mx vPart = pair.second;
      err[i] = VecTools.distance(matrix, MxTools.multiply(MxTools.multiply(vPart, tPart), MxTools.inverse(vPart)));
    }

    assertEquals(0, VecTools.sum(new ArrayVec(err)) / test_num, 1000);
  }

  public void testLanczosTridiagonalizationPerformance() {
    FastRandom rng = new FastRandom();

    int test_num = 1000;
    int n = 1000;

    long startTime = System.currentTimeMillis();
    for (int i = 0; i < test_num; i++) {
      Mx matrix = new VecBasedMx(n, VecTools.fillUniform(new ArrayVec(n * n), rng));
      final Pair<Mx, Mx> pair = MxTools.lanczosTridiagonalization(matrix, matrix.columns());
    }
    long finishTime = System.currentTimeMillis();

    assertTrue((finishTime - startTime) < 1000 * test_num);
  }

  public void testLanczosTridiagonalizationArrayVersionPerformance() {
    FastRandom rng = new FastRandom();

    int test_num = 1000;
    int n = 1000;

    long startTime = System.currentTimeMillis();
    for (int i = 0; i < test_num; i++) {
      Mx matrix = new VecBasedMx(n, VecTools.fillUniform(new ArrayVec(n * n), rng));
      final Pair<Mx, Mx> pair = MxTools.lanczosTridiagonalizationArrayVersion(matrix, matrix.columns());
    }
    long finishTime = System.currentTimeMillis();

    assertTrue((finishTime - startTime) < 1000 * test_num);
  }
}