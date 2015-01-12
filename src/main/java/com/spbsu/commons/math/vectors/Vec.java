package com.spbsu.commons.math.vectors;

import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.seq.Seq;

/**
 * User: solar
 * Date: 16.01.2010
 * Time: 13:10:48
 */
public interface Vec extends Seq<Double> {
  double get(int i);
  Vec set(int i, double val);
  Vec adjust(int i, double increment);
  VecIterator nonZeroes();

  int dim();

  double[] toArray();

  @Override
  Vec sub(int start, int len);

  abstract class Stub extends Seq.Stub<Double> implements Vec {
    @Override
    public final Double at(final int i) {
      return get(i);
    }

    @Override
    public final int length() {
      return Vec.Stub.this.dim();
    }

    @Override
    public final String toString() {
      return MathTools.CONVERSION.convert(this, CharSequence.class).toString();
    }

    @Override
    public abstract Vec sub(final int start, final int end);

    @Override
    public final int hashCode() {
      return VecTools.hashCode(this);
    }

    @Override
    public boolean equals(final Object obj) {
      return obj instanceof Vec && VecTools.equals(this, (Vec)obj);
    }

    @Override
    public boolean isImmutable() {
      return false;
    }

    @Override
    public double[] toArray() {
      final double[] result = new double[dim()];
      final VecIterator nz = nonZeroes();
      while (nz.advance())
        result[nz.index()] = nz.value();
      return result;
    }

    @Override
    public Class<Double> elementType() {
      return double.class;
    }
  }
}

