from raw_corpus_readers.CorpusReaderComplete import RawCorpusReader

if __name__ == '__main__':
    cat_pattern = r'([\w_\s]+)/.*'
    fileids = r'(?!\.)[\w_\s]+/[\w\s\d\-]+\.txt'
    corpus = RawCorpusReader('path\\to\\raw_corpus\\basic_corpus', fileids=fileids, encoding=None, cat_pattern=cat_pattern)
    corpus.transform('path\\to\\App\\raw_corpus\\basic_corpus',
     'path\\to\\preprocessed_corpus\\basic_corpus')