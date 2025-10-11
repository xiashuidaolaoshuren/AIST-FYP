"""
Unit tests for data processing modules (Wikipedia parser and text chunker).
"""

import pytest
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

from src.data_processing import WikipediaParser, TextChunker


class TestWikipediaParser:
    """Tests for WikipediaParser class."""
    
    def test_create_parser(self, tmp_path):
        """Test creating a WikipediaParser instance."""
        # Create a minimal Wikipedia XML file
        xml_file = tmp_path / "test_wiki.xml"
        xml_content = '''<mediawiki>
  <page>
    <title>Test Article</title>
    <id>1</id>
    <revision>
      <text>This is a test article with some content.</text>
    </revision>
  </page>
</mediawiki>'''
        with open(xml_file, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        
        parser = WikipediaParser(str(xml_file))
        assert parser.dump_path == xml_file
        assert parser.max_articles is None
    
    def test_parser_with_max_articles(self, tmp_path):
        """Test WikipediaParser with max_articles limit."""
        xml_file = tmp_path / "test_wiki.xml"
        xml_content = '''<mediawiki>
  <page><title>Article 1</title><id>1</id><revision><text>Content 1 with enough text to pass the minimum length requirement. This text needs to be longer than 100 characters to avoid being filtered out by the parser.</text></revision></page>
  <page><title>Article 2</title><id>2</id><revision><text>Content 2 with enough text to pass the minimum length requirement. This text needs to be longer than 100 characters to avoid being filtered out by the parser.</text></revision></page>
  <page><title>Article 3</title><id>3</id><revision><text>Content 3 with enough text to pass the minimum length requirement. This text needs to be longer than 100 characters to avoid being filtered out by the parser.</text></revision></page>
</mediawiki>'''
        with open(xml_file, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        
        parser = WikipediaParser(str(xml_file), max_articles=2)
        articles = list(parser.extract_articles())
        
        assert len(articles) == 2
        assert articles[0]['doc_id'] == 'enwiki_1'
        assert articles[1]['doc_id'] == 'enwiki_2'
    
    def test_parser_filters_redirects(self, tmp_path):
        """Test that parser filters out redirect pages."""
        xml_file = tmp_path / "test_wiki.xml"
        xml_content = '''<mediawiki>
  <page>
    <title>Redirect Page</title>
    <id>1</id>
    <revision>
      <text>#REDIRECT [[Target Page]]</text>
    </revision>
  </page>
  <page>
    <title>Normal Page</title>
    <id>2</id>
    <revision>
      <text>This is a normal article with actual content that is long enough to be included in the parser output. It contains multiple sentences and more than 100 characters.</text>
    </revision>
  </page>
</mediawiki>'''
        with open(xml_file, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        
        parser = WikipediaParser(str(xml_file))
        articles = list(parser.extract_articles())
        
        assert len(articles) == 1
        assert articles[0]['title'] == 'Normal Page'
    
    def test_parser_filters_disambiguation(self, tmp_path):
        """Test that parser filters out disambiguation pages."""
        xml_file = tmp_path / "test_wiki.xml"
        xml_content = '''<mediawiki>
  <page>
    <title>Test (disambiguation)</title>
    <id>1</id>
    <revision>
      <text>Test may refer to: Test 1, Test 2, Test 3...</text>
    </revision>
  </page>
  <page>
    <title>Real Article</title>
    <id>2</id>
    <revision>
      <text>This is a real article with actual content that should be included in the output. It has multiple sentences and is longer than 100 characters to pass filtering.</text>
    </revision>
  </page>
</mediawiki>'''
        with open(xml_file, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        
        parser = WikipediaParser(str(xml_file))
        articles = list(parser.extract_articles())
        
        assert len(articles) == 1
        assert articles[0]['title'] == 'Real Article'
    
    def test_parser_article_structure(self, tmp_path):
        """Test that parsed articles have the correct structure."""
        xml_file = tmp_path / "test_wiki.xml"
        xml_content = '''<mediawiki>
  <page>
    <title>Test Article</title>
    <id>123</id>
    <revision>
      <text>This is a test article with some content that is long enough to be included in the output. It needs to have more than 100 characters to pass the minimum length filter.</text>
    </revision>
  </page>
</mediawiki>'''
        with open(xml_file, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        
        parser = WikipediaParser(str(xml_file))
        articles = list(parser.extract_articles())
        
        assert len(articles) == 1
        article = articles[0]
        
        assert 'doc_id' in article
        assert 'title' in article
        assert 'text' in article
        assert 'url' in article
        assert article['doc_id'] == 'enwiki_123'
        assert article['title'] == 'Test Article'
        assert article['url'] == 'https://en.wikipedia.org/wiki/Test_Article'


class TestTextChunker:
    """Tests for TextChunker class."""
    
    def test_create_chunker(self):
        """Test creating a TextChunker instance."""
        chunker = TextChunker()
        assert chunker.min_length == 10
        assert chunker.overlap_sentences == 0
    
    def test_chunk_simple_text(self):
        """Test chunking simple text with multiple sentences."""
        chunker = TextChunker()
        article = {
            'doc_id': 'test_123',
            'title': 'Test',
            'text': 'This is the first sentence. This is the second sentence.'
        }
        
        chunks = chunker.chunk_article(article)
        
        assert len(chunks) == 2
        assert chunks[0]['text'] == 'This is the first sentence.'
        assert chunks[1]['text'] == 'This is the second sentence.'
    
    def test_chunk_fields(self):
        """Test that chunks have all required fields."""
        chunker = TextChunker()
        article = {
            'doc_id': 'test_123',
            'title': 'Test',
            'text': 'This is a test sentence.'
        }
        
        chunks = chunker.chunk_article(article)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        
        assert 'doc_id' in chunk
        assert 'sent_id' in chunk
        assert 'text' in chunk
        assert 'char_start' in chunk
        assert 'char_end' in chunk
        assert 'source' in chunk
        assert 'version' in chunk
        
        assert chunk['doc_id'] == 'test_123'
        assert chunk['sent_id'] == 0
        assert chunk['source'] == 'wikipedia'
        assert chunk['version'] == 'wiki_sent_v1'
    
    def test_chunk_character_offsets(self):
        """Test that character offsets are correct."""
        chunker = TextChunker()
        text = 'First sentence. Second sentence.'
        article = {
            'doc_id': 'test_123',
            'title': 'Test',
            'text': text
        }
        
        chunks = chunker.chunk_article(article)
        
        assert len(chunks) == 2
        
        # Verify offsets match actual text
        for chunk in chunks:
            extracted_text = text[chunk['char_start']:chunk['char_end']]
            assert extracted_text.strip() == chunk['text']
    
    def test_filter_short_sentences(self):
        """Test that short sentences are filtered out."""
        chunker = TextChunker(min_length=10)
        article = {
            'doc_id': 'test_123',
            'title': 'Test',
            'text': 'Hi. This is a longer sentence that should be included.'
        }
        
        chunks = chunker.chunk_article(article)
        
        # 'Hi.' is too short and should be filtered
        assert len(chunks) == 1
        assert chunks[0]['text'] == 'This is a longer sentence that should be included.'
    
    def test_chunk_text_method(self):
        """Test the chunk_text convenience method."""
        chunker = TextChunker()
        chunks = chunker.chunk_text("First sentence. Second sentence.", doc_id="test_456")
        
        assert len(chunks) == 2
        assert chunks[0]['doc_id'] == 'test_456'
        assert chunks[1]['doc_id'] == 'test_456'
    
    def test_sent_id_increments(self):
        """Test that sent_id increments correctly."""
        chunker = TextChunker()
        article = {
            'doc_id': 'test_123',
            'title': 'Test',
            'text': 'Sentence one. Sentence two. Sentence three.'
        }
        
        chunks = chunker.chunk_article(article)
        
        assert len(chunks) == 3
        assert chunks[0]['sent_id'] == 0
        assert chunks[1]['sent_id'] == 1
        assert chunks[2]['sent_id'] == 2
