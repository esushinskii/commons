package com.spbsu.commons.io.codec.seq;

import com.spbsu.commons.func.impl.WeakListenerHolderImpl;
import com.spbsu.commons.math.MathTools;
import com.spbsu.commons.seq.CharSeqTools;
import com.spbsu.commons.seq.IntSeq;
import com.spbsu.commons.seq.Seq;
import com.spbsu.commons.util.ArrayTools;
import com.spbsu.commons.util.JSONTools;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.procedure.TIntDoubleProcedure;
import org.jetbrains.annotations.NotNull;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import static java.lang.Integer.max;
import static java.lang.Math.log;
import static java.lang.Math.min;

/**
 * Created with IntelliJ IDEA.
 * User: solar
 * Date: 04.06.12
 * Time: 18:23
 */
public class DictExpansion<T extends Comparable<T>> extends WeakListenerHolderImpl<DictExpansion<T>> {
  public static final double POISSON_SIGNIFICANCE = 0.05;
  public static final double EXTENSION_FACTOR = 1.33;
  public static final double MAX_POWER = 10000000;
  public static final double MAX_MIN_PROBABILITY = 0.002;
  public static final int AGG_POWER = 100000;
  private final boolean isDynamic;
  private int size;
  private final PrintStream trace;
  private final Dictionary<T> initial;
  private volatile DictionaryWithStat<T> suggest;
  private volatile DictionaryWithStat<T> current;
  private DictionaryWithStat<T> result;

  private final int alphabetSize;
  private double probFound = 0.1;
  private double bestCompressionRate = 1;
  private int noRateIncreaseTurns = 0;

  public DictExpansion(final Collection<T> alphabet, final int size) {
    this(alphabet, size, null);
  }

  @Deprecated
  public DictExpansion(final Collection<T> alphabet, final int size, final boolean trace) {
    this(new ListDictionary<>(ArrayTools.toArray(alphabet)), size, trace ? System.out : null);
  }

  public DictExpansion(final Collection<T> alphabet, final int size, final PrintStream trace) {
    this(new ListDictionary<>(ArrayTools.toArray(alphabet)), size, trace);
  }

  public DictExpansion(final Dictionary<T> alphabet, final int size) {
    this(alphabet, size, null);
  }

  public DictExpansion(final Dictionary<T> alphabet, final int size, final PrintStream trace) {
    this.size = size;
    this.trace = trace;
    this.alphabetSize = alphabet.size();
    initial = alphabet;
    isDynamic = !(alphabet instanceof ListDictionary);
    //noinspection unchecked
    current = createDict((Collection<Seq<T>>)alphabet.alphabet(), isDynamic, MAX_MIN_PROBABILITY);
    //noinspection unchecked
    suggest = createDict((Collection<Seq<T>>)alphabet.alphabet(), isDynamic, MAX_MIN_PROBABILITY);
  }

  public DictExpansion(int slots) {
    //noinspection unchecked
    this(Dictionary.EMPTY, slots, null);
  }

  public DictExpansion(int size, PrintStream trace) {
    //noinspection unchecked
    this(Dictionary.EMPTY, size, trace);
  }

  @NotNull
  private static <T extends Comparable<T>> DictionaryWithStat<T> createDict(Collection<Seq<T>> alphabet, boolean isDynamic, double minProbResult) {
    //noinspection unchecked,Convert2Diamond
    return new DictionaryWithStat<T>(isDynamic ? new DynamicDictionary<>(alphabet) : new ListDictionary<T>(alphabet.toArray(new Seq[alphabet.size()])), minProbResult);
  }

  public Dictionary<T> result() {
    return result != null ? result.dict : null;
  }

  public Dictionary<T> alpha() {
    return initial;
  }

  private final ReadWriteLock lock = new ReentrantReadWriteLock();
  public void accept(final Seq<T> seq) {
    lock.readLock().lock();
    try {
      current.pairsFreqs.populate(pairsFreq -> {
        final IntSeq parseResult = current.parse(seq);
        final int length = parseResult.length();
        int prev = -1;
        for(int i = 0; i < length; i++) {
          final int symbol = parseResult.intAt(i);
          current.updateSymbol(symbol, 1);
          if (prev >= 0)
            pairsFreq.adjustOrPutValue((long) prev << 32 | symbol, 1, 1);
          prev = symbol;
        }
      });
      { // parsing with suggest
        final IntSeq parse = suggest.parse(seq);
        for (int i = 0; i < parse.length(); i++) {
          suggest.updateSymbol(parse.intAt(i), 1);
        }
      }
    }
    finally {
      lock.readLock().unlock();
    }

    update();
  }

  private boolean update() {
    lock.writeLock().lock();
    try {
      if ((!current.enough(probFound) || !suggest.enough(probFound)) && suggest.power < MAX_POWER)
        return false;
      double sum = 0;
      double textLength = 0;
      for (int i = 0; i < current.size(); i++) {
        final int freq = current.freq(i);
        textLength += current.get(i).length() * freq;
        if (freq > 0)
          sum -= freq * log(freq) / log(2);
      }
      final double codeLength = (sum + current.power * log(current.power) / log(2)) / 8.;
      final double compressionRate = codeLength / textLength;
      if (compressionRate < bestCompressionRate) {
        bestCompressionRate = compressionRate;
        noRateIncreaseTurns = 0;
      } else if (++noRateIncreaseTurns > 3) {
        probFound *= 0.8;
      }

      result = current;
      invoke(this);

      if (trace != null) {
        final String message = "Size: " + current.size() + " rate: " + compressionRate + " minimal probability: " + suggest.minProbability;
        trace.println(message);
      }
      final DictionaryWithStat<T> currentLocal = current;
      current = suggest.reduce(size - alphabetSize, isDynamic);
      int slots;
      if ((int) (currentLocal.size() * (EXTENSION_FACTOR - 1)) < 10)
        slots = size - alphabetSize;
      else
        slots = min(size - alphabetSize, (int) (currentLocal.size() * (EXTENSION_FACTOR - 1)));
      suggest = currentLocal.expand(slots, isDynamic);

      return true;
    }
    finally {
      lock.writeLock().unlock();
    }
  }

  public int[] resultFreqs() {
    lock.writeLock().lock();
    try {
      if (result.size() > result.symbolFreqs.size())
        result.symbolFreqs.fill(result.symbolFreqs.size(), result.size(), 0);
      return result.symbolFreqs.toArray();
    }
    finally {
      lock.writeLock().unlock();
    }
  }

  public void printPairs(Writer ps) throws IOException {
    lock.readLock().lock();
    final List<? extends Seq<T>> alphabet = result.alphabet();
    final TIntArrayList indices = new TIntArrayList();
    final TDoubleArrayList weights = new TDoubleArrayList();
    final int[] indicesArr = new int[result().size()];
    final double[] weightsArr = new double[result().size()];

    ps.append("{\n");
    try {
      for (int i = 0; i < alphabet.size(); i++) {
        final Seq<T> tSeq = alphabet.get(i);
        indices.clear();
        weights.clear();
        result.visitAssociations(i, (j, val) -> {
          indices.add(j);
          weights.add(val);
          return false;
        });
        weights.toArray(weightsArr, 0, weights.size());
        indices.toArray(indicesArr, 0, indices.size());
        ArrayTools.parallelSort(weightsArr, indicesArr, 0, indices.size());

        final String symbol = tSeq.toString();
        ps.append(JSONTools.escape(symbol)).append(": {");
        for (int j = indices.size() - 1; j >= 0 && weightsArr[j] > 0.001; j--) {
          if (j != indices.size() - 1)
            ps.append(",");
          final String expansion = alphabet.get(indicesArr[j]).toString();
          ps.append("\n").append(JSONTools.escape(expansion)).append(": ").append(CharSeqTools.ppDouble(weightsArr[j] / (double) result.freq(i)));
        }
        ps.append("\n},\n");
      }
    }
    finally {
      lock.readLock().unlock();
    }
    ps.append("}\n");
  }

  public void print(FileWriter fileWriter) throws IOException {
    lock.readLock().lock();

    try {
      for (int i = 0; i < result.size(); i++) {
        final Seq<T> seq = result.get(i);
        fileWriter.append(seq.toString());
        fileWriter.append('\t');
        fileWriter.append(CharSeqTools.itoa(result.freq(i)));
        fileWriter.append('\n');
      }
    }
    finally {
      lock.readLock().unlock();
      fileWriter.close();
    }
  }

  public static class DictionaryWithStat<T extends Comparable<T>> extends DictionaryBase<T> {
    private final Dictionary<T> dict;
    private final TIntArrayList symbolFreqs;
    private double power = 0;
    private final LongIntMappingAsyncBuilder pairsFreqs;
    private final double minProbability;

    public DictionaryWithStat(Dictionary<T> dict, double minProbResult) {
      this.dict = dict;
      symbolFreqs = new TIntArrayList(max(dict.size(), 1_000_000));
      pairsFreqs = new LongIntMappingAsyncBuilder(AGG_POWER);
      minProbability = minProbResult;
    }

    public void updateSymbol(int index, int freq) {
      // trash double-locking, _pos in symbolFreq is not volatile, won't work in some cases, never ever do like this!
      // in this code I trade performance to certainty of symbolFreqs values.
      // In case it will update the same symbol from different thread the value update will be result of race condition.
      // So, the code is total garbage, but it works fast :)
      // No idea how to rewrite this correctly without dramatic loss of performance
      if (index >= symbolFreqs.size()) {
        synchronized (this) {
          if (index >= symbolFreqs.size())
            symbolFreqs.fill(symbolFreqs.size(), index + 1, 0);
        }
      }
      final int val = symbolFreqs.getQuick(index);
      symbolFreqs.setQuick(index, val + freq);
      power += freq;
    }

    @Override
    public int search(Seq<T> seq) {
      return dict.search(seq);
    }

    @Override
    public Seq<T> get(int index) {
      return dict.get(index);
    }

    @Override
    public int size() {
      return dict.size();
    }

    @Override
    public List<? extends Seq<T>> alphabet() {
      return dict.alphabet();
    }

    @Override
    public int parent(int second) {
      return dict.parent(second);
    }

    public int freq(int index) {
      return index < symbolFreqs.size() ? symbolFreqs.getQuick(index) : 0;
    }

    public double codeLength() {
      double sum = 0;
      for (int i = 0; i < size(); i++) {
        final int freq = freq(i);
        if (freq > 0)
          sum -= freq * log(freq);
      }
      return sum + power * log(power);
    }

    private DictionaryWithStat<T> reduce(int slots, boolean isDynamic) {
      final List<StatItem> items = new ArrayList<>();
      final double codeLength = codeLength();
      final List<Seq<T>> newDict = new ArrayList<>(size());

      for (int s = 0; s < size(); s++) {
        final int parent = parent(s);
        final int count = freq(s);
        Seq<T> seq = get(s);
        if (parent < 0)
          newDict.add(seq);
        else if (count > 0) {
          double codeLengthWOSymbol = codeLength + count * log(count);
          double newStatPower = power - count;
          int next = parent;
          do {
            seq = seq.sub(get(next).length(), seq.length());
            final int oldFreq = freq(next);
            final int newFreq = oldFreq + count;
            newStatPower += count;
            codeLengthWOSymbol -= newFreq * log(newFreq) - (oldFreq > 0 ? oldFreq * log(oldFreq) : 0);
          }
          while (seq.length() > 0 && (next = search(seq)) >= 0);
          codeLengthWOSymbol += newStatPower * log(newStatPower + size() - 1) - power * log(power + size());
          items.add(new StatItem(s, -1, s, codeLengthWOSymbol - codeLength, count));
        }
      }
      Collections.sort(items, (o1, o2) -> Double.compare(o2.score, o1.score));

      double minProbResult = min(1. / size(), MAX_MIN_PROBABILITY);
      for (final StatItem item : items) {
        if (item.score < 0. || --slots < 0)
          break;
        final double p = (item.count + 1) / (power + size());
        minProbResult = min(p, minProbResult);
        final Seq<T> symbol = get(item.second);
        newDict.add(symbol);
      }
      //noinspection unchecked
      return createDict(newDict, isDynamic, minProbResult);
    }

    private DictionaryWithStat<T> expand(int slots, boolean isDynamic) {
      final List<StatItem> items = new ArrayList<>();
      pairsFreqs.visit((code, count) -> {
        final int first = (int) (code >>> 32);
        final int second = (int) (code & 0xFFFFFFFFL);
        final double pairProbIndependentDirichlet = freq(first) * freq(second) / power / power;
        final double lambda = pairsFreqs.accumulatedValuesTotal() * pairProbIndependentDirichlet;
        final double logProb = MathTools.logPoissonProbability(lambda, count);
        items.add(new StatItem(code, first, second, count > lambda ? logProb : 0, count));
        return true;
      });

      Collections.sort(items, (o1, o2) -> Double.compare(o1.score, o2.score));
      final List<Seq<T>> newDict = new ArrayList<>(alphabet());
      double minProbResult = minProbability;
      for (final StatItem item : items) {
        if (item.score >= Math.log(POISSON_SIGNIFICANCE) || --slots < 0)
          break;
        newDict.add(CharSeqTools.concat(get(item.first), get(item.second)));
        minProbResult = min(minProbResult, item.count / (double)pairsFreqs.accumulatedValuesTotal());
      }
      //noinspection unchecked
      return createDict(newDict, isDynamic, minProbResult);
    }

    public boolean enough(double probFound) {
      return power > -log(probFound) / minProbability;
    }

    public void visitAssociations(int start, TIntDoubleProcedure procedure) {
      pairsFreqs.visitRange(((long) start) << 32, ((long) start + 1L) << 32, (a, b) -> procedure.execute((int)(a & 0x7FFFFFFFL), b));
    }

    public IntSeq parse(Seq<T> seq) {
//      if (symbolFreqs.size() < dict.size())
//        symbolFreqs.fill(symbolFreqs.size(), dict.size(), 0);
      return parse(seq, symbolFreqs, power);
    }

    private final class StatItem {
      long code;
      int first;
      int second;
      double score;
      int count;

      private StatItem(final long code, final int first, final int second, final double score, final int count) {
        this.code = code;
        this.first = first;
        this.second = second;
        this.score = score;
        this.count = count;
      }

      @Override
      public String toString() {
        final StringBuilder result = new StringBuilder();
        if (first >= 0)
          result.append(get(first));
        result.append(get(second));
        result.append("->(");
        result.append(count);
        result.append(", ").append(score);
        result.append(")");
        return result.toString();
      }
    }
  }
}
