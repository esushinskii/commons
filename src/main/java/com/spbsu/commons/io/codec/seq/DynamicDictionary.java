package com.spbsu.commons.io.codec.seq;

import com.spbsu.commons.seq.ArraySeq;
import com.spbsu.commons.seq.CharSeqTools;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.seq.SeqTools;
import gnu.trove.impl.Constants;
import gnu.trove.map.hash.TObjectIntHashMap;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * User: solar
 * Date: 30.09.15
 * Time: 18:37
 */
public class DynamicDictionary<T extends Comparable<T>> extends DictionaryBase<T> {
  private final Dictionary<T> composites;
  private final TObjectIntHashMap<T> singles = new TObjectIntHashMap<>();
  private final List<Seq<T>> increment = new ArrayList<>();

  public DynamicDictionary(Seq<T>... sex) {
    this(new ArraySeq<>(sex));
  }

  public DynamicDictionary(Collection<Seq<T>> sex) {
    //noinspection unchecked
    this(new ArraySeq<>((Seq<T>[])sex.toArray(new Seq[sex.size()])));
  }

  public DynamicDictionary(Seq<Seq<T>> sex) {
    final List<Seq<T>> effectiveDict = new ArrayList<>();
    for (int i = 0; i < sex.length(); i++) {
      final Seq<T> seq = sex.at(i);
      if (seq.length() <= 1)
        continue;
      final T first = seq.at(0);
      if (!singles.containsKey(first)) {
        singles.put(first, -singles.size() - 1);
        effectiveDict.add(CharSeqTools.<T>create(new Object[]{first}));
      }
      effectiveDict.add(seq);
    }
    //noinspection unchecked
    composites = effectiveDict.size() > 0 ? new ListDictionary<>(effectiveDict.toArray(new Seq[effectiveDict.size()])) : EMPTY;
  }

  private final ReadWriteLock lock = new ReentrantReadWriteLock();
  @Override
  public int search(Seq<T> seq) {
    final T first;
    int index;
    lock.readLock().lock();
    try {
      first = seq.at(0);
      index = singles.get(first);
      if (index != Constants.DEFAULT_INT_NO_ENTRY_VALUE) {
        if (index < 0)
          index = composites.search(seq);
        return index;
      }
    } finally {
      lock.readLock().unlock();
    }
    lock.writeLock().lock();
    try {
      index = increment.size() + composites.size();
      increment.add(CharSeqTools.<T>create(new Object[]{SeqTools.copy(first)}));
      singles.put(first, index);
      return index;
    }
    finally {
      lock.writeLock().unlock();
    }
  }

  @Override
  public Seq<T> get(int index) {
    lock.readLock().lock();
    try {
      return index < composites.size() ? composites.get(index) : increment.get(index - composites.size());
    }
    finally {
      lock.readLock().unlock();
    }
  }

  @Override
  public int size() {
    lock.readLock().lock();
    try {
      return composites.size() + increment.size();
    }
    finally {
      lock.readLock().unlock();
    }
  }

  @Override
  public List<? extends Seq<T>> alphabet() {
    lock.readLock().lock();
    try {
      final List<Seq<T>> result = new ArrayList<>();
      result.addAll(composites.alphabet());
      result.addAll(increment);
      return result;
    }
    finally {
      lock.readLock().unlock();
    }
  }

  @Override
  public int parent(int index) {
    lock.readLock().lock();
    try {
      return index < composites.size() ? composites.parent(index) : -1;
    }
    finally {
      lock.readLock().unlock();
    }
  }

  public Collection<? extends Seq<T>> composite() {
    lock.readLock().lock();
    try {
      final List<? extends Seq<T>> result = composites.alphabet();
      final Iterator<? extends Seq<T>> it = result.iterator();
      while (it.hasNext()) {
        final Seq<T> next = it.next();
        if (next.length() == 1)
          it.remove();
      }
      return result;
    }
    finally {
      lock.readLock().unlock();
    }
  }
}
