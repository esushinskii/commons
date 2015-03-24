package com.spbsu.commons.seq;

import org.jetbrains.annotations.Nullable;

import java.io.IOException;
import java.io.Reader;

/**
 * User: solar
 * Date: 21.03.15
 * Time: 1:42
 */
public class ReaderChopper {
  private final Reader base;
  private char[] buffer = new char[4096];
  private int offset = 0;
  private int read = 0;

  public ReaderChopper(Reader base) {
    this.base = base;
  }

  @Nullable
  public CharSequence chop(char delimiter) throws IOException {
    if (read < 0)
      return null;
    final CharSeqBuilder builder = new CharSeqBuilder();
    int start = offset;
    while (true) {
      if (offset >= read) {
        builder.append(buffer, start, read);
        readNext();
        if (read < 0)
          return builder.build();
        start = 0;
      }
      if (buffer[offset++] == delimiter) {
        builder.append(buffer, start, offset - 1);
        return builder.build();
      }
    }
  }

  public void skip(int count) throws IOException {
    if (read < 0)
      return;
    while (count-- > 0) {
      readNext();
      offset++;
    }
  }

  public boolean eat(char ch) throws IOException {
    if (read < 0)
      return false;
    readNext();
    if (read > 0 && ch == buffer[offset]) {
      offset++;
      return true;
    }
    return false;
  }

  private void readNext() throws IOException {
    if (offset >= read) {
      //noinspection StatementWithEmptyBody
      while ((read = base.read(buffer)) == 0);
      offset = 0;
    }
  }
}