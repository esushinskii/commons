package com.spbsu.util.converters;

import java.nio.ByteBuffer;

/**
 * Created by IntelliJ IDEA.
 * User: igorkuralenok
 * Date: 21.06.2009
 * Time: 14:53:15
 * To change this template use File | Settings | File Templates.
 */
public class ConverterUtil {
  public static void storeSize(int toEncode, ByteBuffer buffer) {
    if(toEncode >= (1 << (7 * 4))) throw new RuntimeException("Such long arrays are not supported by this converter");
    boolean started = false;
    for(int i = 0; i < 4; i++){
      byte current = (byte)(toEncode >> ((3 - i) * 7) & 0x7F);
      if(current != 0 || i == 3 || started){
        if(i != 3) current |= 0x80;
        buffer.put(current);
        started = true;
      }
      else
        buffer.limit(buffer.limit() - 1);
    }
  }

  public static int restoreSize(ByteBuffer source) {
    byte currentByte = source.get();
    int size = 0;
    while((currentByte & 0x80) != 0){
      size += currentByte & 0x7F;
      size <<= 7;
      currentByte = source.get();
    }
    size += currentByte;
    return size;
  }
}
