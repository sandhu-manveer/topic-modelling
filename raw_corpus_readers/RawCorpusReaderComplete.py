"""reader for raw corpus"""
import os
import time
import pickle
import json
import codecs
import bs4
import re

from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.api import CategorizedCorpusReader
from nltk.corpus.reader.plaintext import CategorizedPlaintextCorpusReader
from nltk import sent_tokenize
from nltk import wordpunct_tokenize
from nltk import pos_tag
from nltk import FreqDist

from readability.readability import Unparseable
from readability.readability import Document as Paper

DOC_PATTERN = r'(?!\.)[\w_\s]+/[\w\s\d\-]+\.txt'
CAT_PATTERN = r'([\w_\s]+)/.*'
ENCODING = 'iso-8859-1'

class RawCorpusReader(CategorizedCorpusReader, CorpusReader):
    """
    A corpus reader for raw emails to enable preprocessing.
    """

    def __init__(self, root, **kwargs):
        """
        Initialize the corpus reader.  Categorization arguments
        (``cat_pattern``, ``cat_map``, and ``cat_file``) are passed to
        the ``CategorizedCorpusReader`` constructor.  The remaining
        arguments are passed to the ``CorpusReader`` constructor.
        """

        # Get the CorpusReader specific arguments
        fileids  = kwargs.pop('fileids')
        encoding = kwargs.pop('encoding')

        # Initialize the NLTK corpus reader objects
        CategorizedCorpusReader.__init__(self, kwargs)
        CorpusReader.__init__(self, root, fileids, encoding)

     # In a custom corpus reader class
    def manifest(self):
        """
        Reads and parses the manifest.json file in our corpus if it exists.
        """
        return json.load(self.open("manifest.json"))

    def resolve(self, fileids, categories):
        """
        Returns a list of fileids or categories depending on what is passed
        to each internal corpus reader function. Implemented similarly to
        the NLTK ``CategorizedPlaintextCorpusReader``.
        """
        if fileids is not None and categories is not None:
            raise ValueError("Specify fileids or categories, not both")

        if categories is not None:
            return self.fileids(categories)
        return fileids

    def docs(self, fileids=None, categories=None):
        """
        Returns the complete text of an HTML document, closing the document
        after we are done reading it and yielding it in a memory safe fashion.
        """
        # Resolve the fileids and the categories
        fileids = self.resolve(fileids, categories)

        # Create a generator, loading one document into memory at a time.
        for path, encoding in self.abspaths(fileids, include_encoding=True):
            with codecs.open(path, 'r', encoding=ENCODING) as f:
                yield f.read()

    def paras(self, fileids=None, categories=None):
        """
        Parse the individual paras from the corpus.
        """
        for doc in self.docs(fileids, categories):
            paras = doc.split('\r\n')
            for para in paras:
                yield para 

    def sents(self, fileids=None, categories=None):
        """
        Uses the built in sentence tokenizer to extract sentences from the
        emails.
        """
        for paragraph in self.paras(fileids, categories):
            for sentence in sent_tokenize(paragraph):
                yield sentence

    def words(self, fileids=None, categories=None):
        """
        Uses the built in word tokenizer to extract tokens from sentences.
        """
        for sentence in self.sents(fileids, categories):
            for token in wordpunct_tokenize(sentence):
                yield token

    def tokenize(self, fileids=None, categories=None):
        """
        Segments, tokenizes, and tags a document in the corpus.
        """
        for paragraph in self.paras(fileids=fileids):
            yield [
                pos_tag(wordpunct_tokenize(sent))
                for sent in sent_tokenize(paragraph)
            ]

    def sizes(self, fileids=None, categories=None):
        """
        Returns a list of tuples, the fileid and size on disk of the file.
        This function is used to detect oddly large files in the corpus.
        """
        # Resolve the fileids and the categories
        fileids = self.resolve(fileids, categories)

        # Create a generator, getting every path and computing filesize
        for path in self.abspaths(fileids):
            yield os.path.getsize(path)

    def preprocess(self, fileid, outpath):
        """
        For a single file does the following preprocessing work:
            1. Checks the location on disk to make sure no errors occur.
            2. Gets all emails for the given text.
            3. Segements the emails with the sent_tokenizer
            4. Tokenizes the sentences with the wordpunct_tokenizer
            5. Tags the sentences using the default pos_tagger
            6. Writes the document as a pickle to the target location.
        This method is called multiple times from the transform runner.
        """
        # Must provide outpath
        if not outpath:
            raise ValueError("Specify fileids or categories, not both")

        # Compute the outpath to write the file to.
        target = self.abspath(fileid)
        parent = os.path.dirname(target)

        # Create a data structure for the pickle
        try:
            document = list(self.tokenize(fileids=[target])) # check
        except Exception as ex:
            document = []
            pass

        # Open and serialize the pickle to disk
        with open(outpath, 'wb') as f:
            pickle.dump(document, f, pickle.HIGHEST_PROTOCOL)

        # Clean up the document
        # del document

        # Return the target fileid
        return document

    def preprocess_without_save(self, fileid):
        try:
            document = list(self.tokenize(fileids=[fileid])) # check
        except Exception as ex:
            document = []
            pass

        return document

    def transform(self, htmldir, textdir):
        """
        Pass in a directory containing HTML documents
        and an output directory for the preprocessed
        text and this function transforms the HTML to
        a text corpus that has been tagged in the Brown
        corpus style.
        """
        # List the target HTML directory
        for cat in os.listdir(htmldir):
            catdir = os.path.join(htmldir, cat)
            outCatPath = os.path.join(textdir, cat)
            if not os.path.exists(outCatPath):
                os.mkdir(outCatPath)
            else:
                for name in os.listdir(catdir):
                    # Determine paths of files to transform & write to
                    inpath  = os.path.join(catdir, name)
                    outpath = os.path.join(outCatPath,
                                        os.path.splitext(name)[0]
                                        + ".txt")
                    outpathPickle = os.path.join(outCatPath,
                                        os.path.splitext(name)[0]
                                        + ".pickle")

                    # Open the file for reading UTF-8
                    if os.path.isfile(inpath):
                        with codecs.open(outpath, 'wb', encoding='utf-8') as f:

                            # Paragraphs double newline separated,
                            # sentences separated by a single newline.
                            # Also write token/tag pairs.
                            for paragraph in self.preprocess(inpath, outpathPickle):
                                for sentence in paragraph:
                                    f.write(" ".join("%s/%s"
                                            % (word, tag)
                                            for word, tag in sentence))
                                    f.write("\n")
                                f.write("\n")

    def describe(self, fileids=None, categories=None):
        """
        Performs a single pass of the corpus and
        returns a dictionary with a variety of metrics
        concerning the state of the corpus.
        """
        # Structures to perform counting.
        counts  = FreqDist()
        tokens  = FreqDist()
        started = time.time()

        # Perform single pass over paragraphs, tokenize and count
        for para in self.paras(fileids, categories):
            counts['paras'] += 1

            for sent in self.sents(fileids):
                counts['sents'] += 1

                for word in self.words(fileids):
                    counts['words'] += 1
                    tokens[word] += 1

        # Compute the number of files and categories in the corpus
        n_fileids = len(self.resolve(fileids, categories) or self.fileids())
        n_topics  = len(self.categories(self.resolve(fileids, categories)))

        # Return data structure with information
        return {
            'files':  n_fileids,
            'topics': n_topics,
            'paras':  counts['paras'],
            'sents':  counts['sents'],
            'words':  counts['words'],
            'vocab':  len(tokens),
            'lexdiv': float(counts['words']) / float(len(tokens)),
            'ppdoc':  float(counts['paras']) / float(n_fileids),
            'sppar':  float(counts['sents']) / float(counts['paras']),
            'secs':   time.time() - started,
        }

if __name__ == '__main__':
    cat_pattern = r'([\w_\s]+)/.*'
    fileids = r'(?!\.)[\w_\s]+/[\w\s\d\-]+\.txt'
    
    corpus = RawCorpusReader('path\\to\\raw_corpus\\basic_corpus', fileids=fileids, encoding=None, cat_pattern=cat_pattern)
    corpus.transform('path\\to\\raw_corpus\\basic_corpus',
     'path\\to\\preprocessed_corpus\\basic_corpus')
    # for para in corpus.paras():
    #     print(para)