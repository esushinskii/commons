package util;

import java.util.Random;


import com.spbsu.commons.func.Computable;
import com.spbsu.commons.util.Holder;
import com.spbsu.commons.util.cache.CacheStrategy;
import com.spbsu.commons.util.cache.impl.FixedSizeCache;
import com.spbsu.commons.util.cache.impl.LRUStrategy;
import junit.framework.TestCase;

/**
 * User: Igor Kuralenok
 * Date: 31.08.2006
 */
public class CacheTest extends TestCase {
  public static final String KEY1 = "key1";
  public static final String KEY2 = "key2";
  public static final String KEY3 = "key3";
  public static final String KEY4 = "key4";

  public void testLRU1(){
    final LRUStrategy strategy = new LRUStrategy(3);
    assertEquals(0, strategy.getStorePosition());
    strategy.registerAccess(0);
    assertEquals(1, strategy.getStorePosition());
    strategy.registerAccess(1);
    assertEquals(2, strategy.getStorePosition());
    strategy.registerAccess(2);
    assertEquals(0, strategy.getStorePosition());
  }

  public void testLRU2(){
    final LRUStrategy strategy = new LRUStrategy(3);
    assertEquals(0, strategy.getStorePosition());
    strategy.registerAccess(0);
    assertEquals(1, strategy.getStorePosition());
    strategy.registerAccess(1);
    strategy.registerAccess(0);
    assertEquals(2, strategy.getStorePosition());
    strategy.registerAccess(2);
    assertEquals(1, strategy.getStorePosition());
  }

  public void testLRU3(){
    final LRUStrategy strategy = new LRUStrategy(3);
    assertEquals(0, strategy.getStorePosition());
    strategy.registerAccess(0);
    assertEquals(1, strategy.getStorePosition());
    strategy.registerAccess(1);
    assertEquals(2, strategy.getStorePosition());
    strategy.registerAccess(2);
    strategy.registerAccess(0);
    assertEquals(1, strategy.getStorePosition());
  }

  // TODO: randomly fails
  /*
  public void testCache1(){
    final FixedSizeCache<String, Object> cache = new FixedSizeCache<String, Object>(3, CacheStrategy.Type.LRU);
    assertNull(cache.get(KEY1));
    cache.put(KEY1, new Object());
    cache.put(KEY2, new Object());
    cache.put(KEY3, new Object());
    cache.put(KEY4, new Object());
    System.gc();
    assertNull(cache.get(KEY1));
    assertNotNull(cache.get(KEY2));
    assertNotNull(cache.get(KEY3));
    for (int i = 0; i < 100000; i++) // let cache to process reference queue
      assertNotNull(cache.get(KEY4));
    assertTrue(cache.checkEqualSizes());
  }
  */

  public void testStress1(){
    final FixedSizeCache<Integer, Object> cache = new FixedSizeCache<Integer, Object>(1000, CacheStrategy.Type.LRU);
    int count = 100000;
    for(int i = 0; i < 10000; i++){
      cache.put(i, new Object());
    }
    System.gc();
    Interval.start();
    while(count-- > 0) {
      final int key = (int) (Math.random() * 10000);
      if(cache.get(key) == null) cache.put(key, new Object());
      if(count % 10000 == 0) System.gc();
    }
    System.out.println("Successful accesses: " + cache.getStrategy().getAccessCount());
    System.out.println("Missed: " + cache.getStrategy().getCacheMisses());
    Interval.stopAndPrint();
  }

  public void testPerformanceUniform(){
    FixedSizeCache<Integer, Integer> cache = new FixedSizeCache<Integer, Integer>(1000, CacheStrategy.Type.LRU);
    long time = System.currentTimeMillis();
    final Holder<Integer> missCount = new Holder<Integer>(0);
    for(int i = 0; i < 1000000; i++){
      int key = (int) (Math.random() * 10000);
      cache.get(key, new Computable<Integer, Integer>() {
        public Integer compute(Integer argument) {
          missCount.setValue(missCount.getValue() + 1);
          return 1;
        }
      });
    }
    System.out.println("Total 1000000 tries, " + missCount + " misses");
    System.out.println("Time: " + (System.currentTimeMillis() - time));
  }


  public void testPerformanceNormal(){
    FixedSizeCache<Integer, Integer> cache = new FixedSizeCache<Integer, Integer>(1000, CacheStrategy.Type.LRU);
    long time = System.currentTimeMillis();
    Random rnd = new Random();
    final Holder<Integer> missCount = new Holder<Integer>(0);
    int maxKey = 0;

    for(int i = 0; i < 1000000; i++){
      final double r = rnd.nextGaussian();
      int key = (int) (r * r) * 10000;
      maxKey = Math.max(maxKey, key);
      cache.get(key, new Computable<Integer, Integer>() {
        public Integer compute(Integer argument) {
          missCount.setValue(missCount.getValue() + 1);
          return 1;
        }
      });
    }
    System.out.println("Total 1000000 tries, " + missCount + " misses. Maximal key: " + maxKey);
    System.out.println("Time: " + (System.currentTimeMillis() - time));
  }
}
