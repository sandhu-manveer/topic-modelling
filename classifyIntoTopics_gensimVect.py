import os
import unicodedata
import nltk
import pickle
import codecs

from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn

from gensim.sklearn_integration import SklLdaModel
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel

from preprocessed_corpus_reader.reader import PickledCorpusReader
from raw_corpus_readers.RawCorpusReader import RawCorpusReader

from topicModelling_gensimVect import GensimTopicModels, GensimTfidfVectorizer, TextNormalizer

def get_topics(vectorized_corpus, model):
    from operator import itemgetter

    topics = [
        max(model[doc], key=itemgetter(1))[0]
        for doc in vectorized_corpus
    ]

    return topics

gensim_lda = pickle.load(open('path\\to\\models\\topics\\LdaModelalpha_auto.pkl', 'rb'))

lda = gensim_lda.model.named_steps['model'].gensim_model

pickled_corpus = PickledCorpusReader('path\\to\\preprocessed_corpus\\basic_corpus')
docs = pickled_corpus.docs()
corpus_docs = list(docs)

corpus = [
    gensim_lda.model.named_steps['vect'].lexicon.doc2bow(doc)
    for doc in gensim_lda.model.named_steps['norm'].transform(corpus_docs)
]

topics = get_topics(corpus,lda)

# read raw docs
cat_pattern = r'([\w_\s]+)/.*'
fileids = r'(?!\.)[\w_\s]+/[\w\s\d\-]+\.txt'
raw_doc_reader = RawCorpusReader('path\\to\\App\\raw_corpus\\basic_corpus', fileids=fileids, encoding=None, cat_pattern=cat_pattern)
raw_docs = raw_doc_reader.docs()
raw_docs = list(raw_docs)
fileids = list(raw_doc_reader.fileids())

print(len(topics))
print(len(fileids))

def createResultCorpus(topics, raw_docs, results_dir="/results/LDA_topic_separated_docs"):
    for topic, fileid, doc in zip(topics, fileids , raw_docs):
        if not os.path.exists(os.path.join(results_dir, str(topic))):
            os.makedirs(os.path.join(results_dir, str(topic)))

        with codecs.open(os.path.join(results_dir, str(topic), fileid.split('/')[1]), 'wb', encoding='utf-8') as f:
            f.write(doc)

createResultCorpus(topics, raw_docs, results_dir="path\\to\\App\\results\\LDA_topic_separated_docs_alpha_31Aug")
