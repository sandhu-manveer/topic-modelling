import os
import unicodedata
import nltk
import pickle

import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn

from preprocessed_corpus_reader.reader import PickledCorpusReader

from collections import Counter

class TextNormalizer(BaseEstimator, TransformerMixin):
    """
    Normalizer class
    """
    def __init__(self, language='english', custom_stopwords=[]):
        self.stopwords  = set(nltk.corpus.stopwords.words(language))
        self.stopwords.update(custom_stopwords)
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
        """
        Removes stopwords and punctuation, lowercases, lemmatizes
        """
        for token, tag in document:
            token = token.lower().strip()

            if self.is_punct(token) or self.is_stopword(token):
                continue

            yield self.lemmatize(token, tag)

    def lemmatize(self, token, pos_tag):
        """Lemmatize data"""
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(pos_tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

class HierarchicalTopics(object):

    def __init__(self, corpus, custom_stopwords=[]):
        """
        The corpus is a corpus object,
        e.g. HTMLCorpusReader() or PickledCorpusReader()
        """
        self.model = None
        self.normalizer = TextNormalizer(custom_stopwords=custom_stopwords)
        self.vocab = list(
            set(self.normalizer.normalize(corpus.words()))
        )
    
    def vectorize(self, document):
        features = set(self.normalizer.normalize(document))
        return np.array([
            token in features for token in self.vocab], np.short)

    
    def cluster(self, corpus):
        """
        Fits the AgglomerativeClustering model to the given data.
        """
        self.model = AgglomerativeClustering()

        vectors = [self.vectorize(
                corpus.words(fileid)) for fileid in
                corpus.fileids()]

        self.model.fit_predict(vectors)

        self.labels = self.model.labels_
        self.children = self.model.children_

    def plot_dendrogram(self, **kwargs):
        """
        Compute the distances between each pair of children and
        a position for each child node. Then create a linkage
        matrix, and plot the dendrogram.
        """
        distance = np.arange(self.children.shape[0])
        position = np.arange(2, self.children.shape[0]+2)

        linkage_matrix = np.column_stack([
            self.children, distance, position]
        ).astype(float)

        fig, ax = plt.subplots(figsize=(15, 7))

        ax = dendrogram(linkage_matrix, orientation='left', **kwargs)

        plt.tick_params(axis='x', bottom='off', top='off', labelbottom='off')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    fp = open('stopwords.txt', 'r')
    stopwords = fp.read()
    stopwords = stopwords.split('\n')
    fp.close()

    corpus = PickledCorpusReader('path\\to\\preprocessed_corpus\\basic_corpus')

    agglomerative_clust = HierarchicalTopics(corpus=corpus,custom_stopwords=stopwords)
    agglomerative_clust.cluster(corpus)

    with open('models/topics' + 'HierarchicalTopics.pkl', 'wb') as fobj:
        pickle.dump(agglomerative_clust, fobj)

    # # visualize
    # corpus = PickledCorpusReader('path\\to\\preprocessed_small')
    # normalizer = TextNormalizer(custom_stopwords=stopwords)
    # labels=[]
    # for fileid in corpus.fileids():
    #     terms = []
    #     for term, count in Counter(list(normalizer.normalize(corpus.words(fileid)))).most_common(10):
    #         terms.append(term)
    #     labels.append(terms)

    # with open('models/topics/HierarchicalTopics.pkl', 'rb') as f:
    #     clusterer = pickle.load(f)
    #     clusterer.plot_dendrogram(labels=labels, leaf_font_size=9)
