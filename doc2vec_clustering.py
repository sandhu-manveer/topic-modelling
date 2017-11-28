"""Trains classifiers after doc2vec vectorization"""
import nltk
import unicodedata
import numpy as np
import pickle
import os
import gc

from corpus_loaders.loader import CorpusLoader
from preprocessed_corpus_reader.reader import PickledCorpusReader
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

from random import shuffle

from gensim.corpora import Dictionary
from gensim.matutils import sparse2full
from gensim.models import word2vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

def identity(words):
    return words

class TextNormalizer(BaseEstimator, TransformerMixin):
    """
    Normalizer class
    """
    def __init__(self, language='english'):
        self.stopwords  = set(nltk.corpus.stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def is_punct(self, token):
        """Removes punctuation"""
        return all(
            unicodedata.category(char).startswith('P') for char in token
        )

    def is_stopword(self, token):
        """Removes stopwords"""
        return token.lower() in self.stopwords

    def normalize(self, document):
        """Run normaalizer"""
        return [
            self.lemmatize(token, tag).lower()
            for paragraph in document
            for sentence in paragraph
            for (token, tag) in sentence
            if not self.is_punct(token) and not self.is_stopword(token)
        ]

    def lemmatize(self, token, pos_tag):
        """Lemmatize data"""
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(pos_tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        for document in documents:
            yield self.normalize(document)

class GensimVectorizer(BaseEstimator, TransformerMixin):
    """Gensim vectorization with doc2vec"""

    def __init__(self, path=None):
        self.path = path
        self.model = None
        self.load()

    def load(self):
        """Load doc2vec model if it exists"""
        if os.path.exists(self.path):
            self.model = Doc2Vec.load(self.path)

    def save(self):
        """Save model"""
        self.model.save(self.path)

    def fit(self, documents, labels=None):
        if not self.model:
            trgDocs = list(documents)
            total_examples = len(trgDocs)
            self.model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=8)
            sentences = self.LabeledLineSentence(trgDocs, labels)
            self.model.build_vocab(sentences.to_array())
            self.model.train(sentences.sentences_perm(), total_examples=total_examples, epochs=10)
            self.save()
            self.model = Doc2Vec.load(self.path)
        else:
            pass
        return self

    def transform(self, documents):
        returnDocs = []
        for doc in documents:
            returnDocs.append(self.model.infer_vector(doc))
        return returnDocs

    class LabeledLineSentence(object):
        """Create labelled sentences for training doc2vec model"""
        def __init__(self, documents, labels):
            self.documents = documents
            self.labels = labels
            self.sources = []
            for i in range(0, len(self.documents)):
                self.sources.append( [ self.documents[i], self.labels[i] ] )
        
        def __iter__(self):
            for element, label in self.sources:
                yield LabeledSentence(element, [label])
        
        def to_array(self):
            self.sentences = []
            for line, label in self.sources:
                self.sentences.append(LabeledSentence(line, [label]))
            return self.sentences
        
        def sentences_perm(self):
            shuffle(self.sentences)
            return self.sentences

if __name__ == '__main__':
    labels = ["not_escalated", "BEMS","RRR", "CAP"]
    reader = PickledCorpusReader('path\\to\\preprocessed_data\\preprocessed_min')

    trg_docs = reader.docs(categories=labels)

    # normalize
    normalizer = TextNormalizer()
    normalized_docs = normalizer.transform(trg_docs)

    # vectorize
    vectorizer = GensimVectorizer(path='path\\to\\doc2vec_complete\\doc2vec.d2v')
    vectorizer.fit(normalized_docs)
    vectorized_docs = vectorizer.transform(normalized_docs)

    # vectorized_docs = np.toarray(vectorized_docs)

    # clusterer = KMeans(n_clusters=8)
    # labels = clusterer.fit_predict(vectorized_docs)

    # with open('models/unsupervised/' + 'Kmeans' + '.pkl', 'wb') as fobj:
    #     pickle.dump(clusterer, fobj)

    clusterer = pickle.load(open('models/unsupervised/Kmeans.pkl', 'rb'))
    clusters = clusterer.predict(vectorized_docs)

    outdoc = open('output.txt', 'w')
    refetch_docs = reader.fileids(categories=labels)
    count = 0
    for doc, cluster in zip(list(refetch_docs), clusters):
        outdoc.write('{}: Cluster {}\n'.format(doc, cluster))
        count += 1