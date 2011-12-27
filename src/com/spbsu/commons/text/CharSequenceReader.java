package com.spbsu.commons.text;

import java.io.Reader;
import java.io.IOException;

/**
 * User: igorkuralenok
 * Date: 09.06.2009
 */
public class CharSequenceReader extends Reader {
  private int currentOffset = 0;
  private final CharSequence seq;

  public CharSequenceReader(CharSequence seq) {
    super(seq);
    this.seq = seq;
  }

  public int read(char[] cbuf, int off, int len) throws IOException {
    int read;
    for(read = 0; read < len && currentOffset < seq.length(); read++){
      cbuf[off + read] = seq.charAt(currentOffset++);
    }
    return read == 0 && currentOffset == seq.length() ? -1 : read;
  }

  @Override
  public int read() throws IOException {
    return seq.charAt(currentOffset++);
  }

  @Override
  public void reset() throws IOException {
    currentOffset = 0;
  }

  public void close() throws IOException {
  }
}