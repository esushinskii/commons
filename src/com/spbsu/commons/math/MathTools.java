package com.spbsu.commons.math;

/**
 * @author vp
 */
public abstract class MathTools {

  public static final double EPSILON = 1e-6;

  private MathTools() {
  }

  public static int max(final int[] values) {
    int max = values[0];
    for (int value : values) {
      if (value > max) max = value;
    }
    return max;
  }

  public static double max(final double[] values) {
    double max = values[0];
    for (double value : values) {
      if (value > max) max = value;
    }
    return max;
  }

  public static double sum(final double[] values) {
    double sum = 0.0;
    for (double value : values) {
      sum += value;
    }
    return sum;
  }

  public static int factorial(final int v) {
    int result = 1;
    for (int i = 2; i <= v; i++) {
      result *= i;
    }
    return result;
  }

  public static double logFactorial(final int v) {
    double logSum = 0;
    for (int i = 0; i < v; i++) {
      logSum += Math.log(i + 1);
    }
    return logSum;
  }

  public static double logFactorialRamanujan(final int v) {
    return (v == 1 || v == 0) ? 0 : v * Math.log(v) - v + Math.log(v * (1 + 4 * v * (1 + 2 * v))) / 6. + Math.log(Math.PI) / 2;
  }

  public static double poissonProbability(final double lambda, final int k) {
    return Math.exp(-lambda + k * Math.log(lambda) - logFactorial(k));
  }

  public static double poissonProbabilityFast(final double lambda, final int k) {
    return Math.exp(-lambda + k * Math.log(lambda) - logFactorialRamanujan(k));
  }

  public static double conditionalNonPoissonExpectation(final double noiseExpectation, final int observationCount) {
    double expectation = 0;
    for (int trialOccurenceCount = 1; trialOccurenceCount <= observationCount; trialOccurenceCount++) {
      final double trialProbability = poissonProbability(noiseExpectation, observationCount - trialOccurenceCount);
      expectation += trialOccurenceCount * trialProbability;
    }
    return expectation;
  }

  public static double conditionalNonPoissonExpectationFast(final double noiseExpectation, final int observationCount) {
    final double lambdaLog = Math.log(noiseExpectation);

    double expectation = 0;
    double logSum = 0;
    for (int noiseCount = 0; noiseCount < observationCount; noiseCount++) {
      final double trialProbability = Math.exp(-noiseExpectation + noiseCount * lambdaLog - logSum);
      expectation += (observationCount - noiseCount) * trialProbability;
      logSum += Math.log(noiseCount + 1);
    }
    return expectation;
  }

  public static double triroot(double a) {
    return Math.signum(a) * Math.pow(Math.abs(a), 1./3.);
  }

  public static int quadratic(double[] x, double a, double b, double c) {
    if (Math.abs(a) > EPSILON) {
      double D = b * b - 4 * a * c;
      if (D < 0) {
        return 0;
      }
      x[0] = (-b + Math.sqrt(D)) / 2. / a;
      x[1] = (-b - Math.sqrt(D)) / 2. / a;
      return 2;
    }
    else {
      x[0] = -c / b;
      return 1;
    }
  }

  public static int cubic(double[] x, double a, double b, double c, double d) {
    if (Math.abs(a) < EPSILON) {
      return quadratic(x, b, c, d);
    }
    b /= a; c /= a; d /= a;
    double p = - b * b /3. + c;
    double q = 2. * b * b * b / 27. - b * c / 3 + d;
    x[0] = x[1] = x[2] = -b/3.;
    double Q = p * p * p /27. + q * q /4.;
    if (Q > 0) {
      double alpha = triroot(-q/2 + Math.sqrt(Q));
      double beta = triroot(-q/2 - Math.sqrt(Q));
      x[0] += alpha + beta;
      return 1;
    }
    else if (Q < 0) {
      double ab = 2 * triroot(Math.sqrt(-Q));
      double ambi = 2 * triroot(Math.sqrt(-Q));
      x[0] += ab;
      x[1] += -ab/2. + ambi/2.*Math.sqrt(3.);
      x[2] += -ab/2. - ambi/2.*Math.sqrt(3.);
      return 3;
    }
    else {
      double ab = 2 * triroot(-q/2);
      x[0] += ab;
      x[1] += -ab/2.;
      return 2;
    }
  }

}

