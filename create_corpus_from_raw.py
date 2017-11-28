from raw_to_corpus_converter.RawToCorpusConverter import RawToCorpusConverter
from raw_corpus_readers.RawCorpusReader import RawCorpusReader

if __name__ == '__main__':
    converter = RawToCorpusConverter('path\\to\\raw_data\\items_bcgspider_4.csv')
    categories = converter.getCategories()
    converter.setUpCorpus('path\\to\\preprocessed_data\\basic_corpus', categories)