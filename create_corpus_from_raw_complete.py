from raw_to_corpus_converter.RawToCorpusConverter import RawToCorpusConverter
from raw_corpus_readers.RawCorpusReaderComplete import RawCorpusReader

if __name__ == '__main__':
    converter = RawToCorpusConverter('path\\to\\WR\\App\\raw_data\\items_bcgspider_4.csv')
    categories = converter.getCategories()
    converter.setUpCorpus('path\\to\\WR\\App\\raw_corpus\\complete_corpus', categories)