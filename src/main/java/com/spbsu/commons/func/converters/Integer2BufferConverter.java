package com.spbsu.commons.func.converters;


import com.spbsu.commons.func.Converter;
import com.spbsu.commons.io.Buffer;
import com.spbsu.commons.io.BufferFactory;

/**
 * User: Igor Kuralenok
 * Date: 02.09.2006
 */
public class Integer2BufferConverter implements Converter<Integer, Buffer> {
  @Override
  public Integer convertFrom(final Buffer source) {
    if(source.remaining() < 4) return null;
    return source.getInt();
  }

  @Override
  public Buffer convertTo(final Integer object) {
    final Buffer buffer = BufferFactory.wrap(new byte[4]);
    buffer.putInt(object);
    buffer.position(0);
    return buffer;
  }
}