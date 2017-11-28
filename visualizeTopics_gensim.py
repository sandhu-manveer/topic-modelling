import os
import unicodedata
import nltk
import pickle

import pyLDAvis
import pyLDAvis.gensim

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

gensim_lda = pickle.load(open('path\\to\\models\\topics\\LdaModel.pkl', 'rb'))

lda = gensim_lda.model.named_steps['model'].gensim_model

pickled_corpus = PickledCorpusReader('path\\to\\preprocessed_corpus\\basic_corpus')
docs = pickled_corpus.docs()

corpus = [
    gensim_lda.model.named_steps['vect'].lexicon.doc2bow(doc)
    for doc in gensim_lda.model.named_steps['norm'].transform(docs)
]

lexicon = gensim_lda.model.named_steps['vect'].lexicon

data = pyLDAvis.gensim.prepare(lda, corpus, lexicon)
pyLDAvis.display(data)