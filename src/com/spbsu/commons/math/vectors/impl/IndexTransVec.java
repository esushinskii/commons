package com.spbsu.commons.math.vectors.impl;

import com.spbsu.commons.func.converters.Vec2StringConverter;
import com.spbsu.commons.math.vectors.*;
import com.spbsu.commons.math.vectors.impl.idxtrans.SubMxTransformation;
import com.spbsu.commons.math.vectors.impl.iterators.MxIteratorImpl;
import com.spbsu.commons.math.vectors.impl.iterators.SkipVecNZIterator;
import com.spbsu.commons.math.vectors.impl.iterators.TransformedSparseVecIterator;
import com.spbsu.commons.util.ArrayTools;
import gnu.trove.TIntArrayList;

/**
 * User: solar
 * Date: 9/14/12
 * Time: 1:17 PM
 */
public class IndexTransVec implements Vec {
  private final Vec base;
  private final IndexTransformation transformation;

  public IndexTransVec(Vec base, IndexTransformation transformation) {
    if (base instanceof IndexTransVec) {
      final IndexTransVec transVec = (IndexTransVec) base;
      this.base = transVec.base;
      this.transformation = transformation.apply(transVec.transformation);
    }
    else {
      this.base = base;
      this.transformation = transformation;
    }
  }

  public double get(int i) {
    return base.get(transformation.forward(i));
  }

  public Vec set(int i, double val) {
    return base.set(transformation.forward(i), val);
  }

  public Vec adjust(int i, double increment) {
    return base.adjust(transformation.forward(i), increment);
  }

  public VecIterator nonZeroes() {
    Vec base = this.base;
    if (this.base instanceof VecBasedMx) {
      base = ((VecBasedMx)this.base).vec;
    }
    VecIterator result;
    if (base instanceof SparseVec) {
      SparseVec sparseVec = (SparseVec)base;
      TIntArrayList indices = sparseVec.indices;

      final TIntArrayList nzIndices = new TIntArrayList(indices.size());
      final TIntArrayList transformed = new TIntArrayList(indices.size());
      int end = transformation.oldIndexEndHint();
      int firstRelevant = indices.binarySearch(transformation.oldIndexStartHint());
      firstRelevant = firstRelevant >= 0 ? firstRelevant : -firstRelevant - 1;
      for (int i = firstRelevant; i < indices.size(); i++) {
        int index = indices.getQuick(i);
        if (index > end)
          break;
        int newIndex = transformation.backward(index);
        if (newIndex >= 0) {
          nzIndices.add(newIndex);
          transformed.add(i);
        }
      }
      int[] transA = transformed.toNativeArray();
      int[] nzIndicesA = nzIndices.toNativeArray();
      ArrayTools.parallelSort(nzIndicesA, transA);
      result = new TransformedSparseVecIterator(indices, sparseVec.values, new TIntArrayList(nzIndicesA), new TIntArrayList(transA));
    }
    else if (base instanceof ArrayVec) {
      result = new SkipVecNZIterator(this);
    }
    else throw new IllegalArgumentException("Can not produce NZ itarator for base type " + base.getClass().toString());
    return this.base instanceof VecBasedMx ? new MxIteratorImpl(result, ((VecBasedMx) this.base).columns()) : result;
  }

  public int dim() {
    return transformation.newDim();
  }

  @Override
  public double[] toArray() {
    double[] result = new double[transformation.newDim()];
    if (base instanceof SparseVec) {
      VecIterator iter = base.nonZeroes();
      while (iter.advance()) {
        final int newIndex = transformation.backward(iter.index());
        if (newIndex >= 0)
          result[newIndex] = iter.value();
      }
    }
    else {
      for (int i = 0; i < result.length; i++) {
        result[i] = base.get(transformation.forward(i));
      }
    }
    return result;
  }

  @Override
  public Vec sub(int start, int len) {
    return new IndexTransVec(this, new SubMxTransformation(dim(), start, 0, len, 1));
  }

  @Override
  public String toString() {
    return new Vec2StringConverter().convertTo(this);
  }

  @Override
  public boolean equals(Object obj) {
    return obj instanceof Vec && VecTools.equals(this, (Vec)obj);
  }

  @Override
  public int hashCode() {
    return VecTools.hashCode(this);
  }
}