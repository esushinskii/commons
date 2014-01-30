package com.spbsu.commons.func.converters;

import com.spbsu.commons.func.Computable;
import com.spbsu.commons.func.Converter;
import com.spbsu.commons.io.Buffer;
import com.spbsu.commons.io.BufferFactory;
import com.spbsu.commons.func.converters.NioConverterTools;

import java.nio.BufferUnderflowException;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.Charset;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CharsetEncoder;

/**
 * User: Igor Kuralenok
 * Date: 02.09.2006
 */
public class CharSequence2BufferConverter<T extends CharSequence> implements Converter<CharSequence, Buffer> {
  static CharsetDecoder DECODER;
  static CharsetEncoder ENCODER;
  private Computable<char[], T> factory;

  public CharSequence2BufferConverter(Computable<char[], T> factory) {
    this.factory = factory;
  }

  static {
    Charset cs = Charset.forName("UTF-8");
    DECODER = cs.newDecoder();
    ENCODER = cs.newEncoder();
  }

  public static synchronized CharBuffer decode(ByteBuffer byteBuffer) throws CharacterCodingException {
    return DECODER.decode(byteBuffer);
  }

  public static synchronized ByteBuffer encode(CharBuffer charBuffer) throws CharacterCodingException {
    return ENCODER.encode(charBuffer);
  }

  public CharSequence convertFrom(Buffer source) {
    if (source.remaining() < 1)
      throw new BufferUnderflowException();
    final int length = NioConverterTools.restoreSize(source);
    final byte[] bytes = new byte[length];
    if (source.get(bytes) != length)
      throw new RuntimeException("Corrupted char sequence");

    try {
      final CharBuffer buffer = decode(ByteBuffer.wrap(bytes));
      final char[] chars = new char[buffer.length()];
      buffer.get(chars);
      return factory.compute(chars);
    }
    catch (CharacterCodingException e) {
      throw new RuntimeException(e);
    }
  }

  public Buffer convertTo(CharSequence cs) {
    try {
      final ByteBuffer contents = encode(CharBuffer.wrap(cs));
      return BufferFactory.join(NioConverterTools.storeSize(contents.remaining()), BufferFactory.wrap(contents));
    }
    catch (CharacterCodingException e) {
      throw new RuntimeException(e);
    }
  }
}