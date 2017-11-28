import os
import unicodedata
import nltk
import pickle
import codecs

from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import AgglomerativeClustering

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn

from gensim.sklearn_integration import SklLdaModel
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel

from preprocessed_corpus_reader.reader import PickledCorpusReader
from raw_corpus_readers.RawCorpusReader import RawCorpusReader

import agglomerativeClust_basic
from agglomerativeClust_basic import HierarchicalTopics

agglomerative_clust = pickle.load(open('path\\to\\models\\HierarchicalTopics_30.pkl', 'rb'))

topics = agglomerative_clust.labels

# read raw docs
cat_pattern = r'([\w_\s]+)/.*'
fileids = r'(?!\.)[\w_\s]+/[\w\s\d\-]+\.txt'
raw_doc_reader = RawCorpusReader('path\\to\\App\\raw_corpus\\basic_corpus', fileids=fileids, encoding=None, cat_pattern=cat_pattern)
raw_docs = raw_doc_reader.docs()
raw_docs = list(raw_docs)
fileids = list(raw_doc_reader.fileids())

print(len(topics))
print(len(fileids))

def createResultCorpus(topics, raw_docs, results_dir="/results/AgglomerativeClustering_topic_separated_docs"):
    for topic, fileid, doc in zip(topics, fileids , raw_docs):
        if not os.path.exists(os.path.join(results_dir, str(topic))):
            os.makedirs(os.path.join(results_dir, str(topic)))

        with codecs.open(os.path.join(results_dir, str(topic), fileid.split('/')[1]), 'wb', encoding='utf-8') as f:
            f.write(doc)

createResultCorpus(topics, raw_docs, results_dir="path\\to\\results\\AgglomerativeClustering_topic_separated_docs")
