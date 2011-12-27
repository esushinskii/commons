package com.spbsu.commons.xml;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.jdom.Document;
import org.jdom.Element;
import org.jdom.JDOMException;
import org.jdom.input.SAXBuilder;
import org.jdom.output.Format;
import org.jdom.output.XMLOutputter;
import org.jetbrains.annotations.NotNull;

import java.io.*;
import java.util.Collections;
import java.util.List;

/**
 * User: selivanov
 * Date: 28.12.2009 : 2:24:52
 */
public class JDOMUtil {
  private static final Log LOG = LogFactory.getLog(JDOMUtil.class);
  private static final XMLOutputter OUTPUTTER = new XMLOutputter(Format.getPrettyFormat());

  private JDOMUtil() {
  }

  public static Element loadXML(File file) throws IOException {
    return loadXMLDocument(file).detachRootElement();
  }

  public static Element parseXml(String xmlFile) {
    try {
      return loadXML(new ByteArrayInputStream(xmlFile.getBytes("utf-8")));
    } catch (UnsupportedEncodingException e) {
      throw new RuntimeException("i/o while reading xml: ", e);
    }
  }

  public static Element loadXML(InputStream xmlStream) {
    return loadXMLDocument(xmlStream).detachRootElement();
  }

  public static Document loadXMLDocument(File file) throws IOException {
    final FileInputStream stream = new FileInputStream(file);
    try {
      return loadXMLDocument(stream);
    } finally {
      stream.close();
    }
  }

  public static Document loadXMLDocument(InputStream xmlStream) {
    Reader reader = null;
    try {
      reader = new BufferedReader(new InputStreamReader(xmlStream, "UTF-8"));
      return new SAXBuilder().build(reader);
    } catch (IOException e) {
      LOG.fatal("i/o while reading xml: ", e);
      throw new RuntimeException("i/o while reading xml: ", e);
    } catch (JDOMException e) {
      LOG.fatal("jdom exception while parsing xml: ", e);
      throw new RuntimeException("jdom exception while parsing xml: ", e);
    } finally {
      if (reader != null) {
        try {
          reader.close();
        } catch (IOException e) {
          LOG.error("fail to close file", e);
        }
      }
    }
  }

  public static void flushXML(Element element, final File file) throws IOException {
    PrintWriter writer = null;
    try {
      writer = new PrintWriter(new OutputStreamWriter(new FileOutputStream(file), "utf-8"));
      OUTPUTTER.output(element, writer);
    } finally {
      if (writer != null) {
        writer.close();
      }
    }
  }

  @NotNull
  public static List<Element> getChildren(@NotNull final Element element, @NotNull final String name) {
    @SuppressWarnings({"unchecked"}) final List<Element> children = element.getChildren(name);
    if (children != null) {
      return Collections.unmodifiableList(children);
    }
    return Collections.emptyList();
  }

  @NotNull
  public static List<Element> getContent(@NotNull final Element element) {
    @SuppressWarnings({"unchecked"}) final List<Element> content = element.getContent();
    return content != null ? Collections.unmodifiableList(content) : Collections.<Element>emptyList();
  }

}