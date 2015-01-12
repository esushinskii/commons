package com.spbsu.commons.func.converters;

import com.spbsu.commons.func.Converter;
import com.spbsu.commons.io.Buffer;
import com.spbsu.commons.util.Factories;

import java.util.List;

/**
 * User: terry
 * Date: 13.12.2009
 */
public class ArrayList2BufferConverter<T> extends List2BufferConverter<T> {
  public ArrayList2BufferConverter(final Converter<T, Buffer> dataConverter) {
    super(dataConverter);
  }

  @Override
  protected List<T> createList() {
    return Factories.arrayList();
  }
}
