import os
import unicodedata
import nltk
import pickle

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

from topicModelling_gensimVect import GensimTopicModels, GensimTfidfVectorizer, TextNormalizer

def get_topics(vectorized_corpus, model):
    from operator import itemgetter

    topics = [
        max(model[doc], key=itemgetter(1))[0]
        for doc in vectorized_corpus
    ]

    return topics

gensim_lda = pickle.load(open('path\\to\\models\\topics\\LdaModel.pkl', 'rb'))

lda = gensim_lda.model.named_steps['model'].gensim_model

pickled_corpus = PickledCorpusReader('path\\to\\preprocessed_corpus\\basic_corpus')
docs = pickled_corpus.docs()
corpus_docs = list(docs)


corpus = [
    gensim_lda.model.named_steps['vect'].lexicon.doc2bow(doc)
    for doc in gensim_lda.model.named_steps['norm'].transform(corpus_docs)
]

topics = get_topics(corpus,lda)
print(set(topics))
print("topics:", len(topics))
print("docs:", len(corpus_docs))

# for topic, doc in zip(topics, corpus_docs):
#     print("Topic:{}".format(topic))
#     print(doc)