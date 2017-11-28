from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

import pickle

from preprocessed_corpus_reader.reader import PickledCorpusReader
import nltk
import unicodedata
import numpy as np

from nltk.corpus import wordnet as wn

STOPWORDS   = set(nltk.corpus.stopwords.words('english'))
lemmatizer  = nltk.WordNetLemmatizer()

def is_punct(token):
    # Is every character punctuation?
    return all(
        unicodedata.category(char).startswith('P')
        for char in token
    )
def wnpos(tag):
    # Return the WordNet POS tag from the Penn Treebank tag
    return {
        'N': wn.NOUN,
        'V': wn.VERB,
        'R': wn.ADV,
        'J': wn.ADJ
    }.get(tag[0], wn.NOUN)

def normalize(document, stopwords=STOPWORDS):
    """
    Removes stopwords and punctuation, lowercases, lemmatizes
    """

    for token, tag in document:
        token = token.lower().strip()

        if is_punct(token) or (token in stopwords):
            continue

        yield lemmatizer.lemmatize(token, wnpos(tag))

class HierarchicalTopics(object):

    def __init__(self, corpus):
        """
        The corpus is a corpus object,
        e.g. HTMLCorpusReader() or PickledCorpusReader()
        """
        self.model = None
        self.vocab = list(
            set(normalize(corpus.words(categories=['ArticlesItem'])))
        )
    
    def vectorize(self, document):
        features = set(normalize(document))
        return np.array([
            token in features for token in self.vocab], np.short)

    
    def cluster(self, corpus, n_clusters=2):
        """
        Fits the AgglomerativeClustering model to the given data.
        """
        self.model = AgglomerativeClustering(n_clusters=n_clusters)

        vectors = [self.vectorize(
                corpus.words(fileid)) for fileid in
                corpus.fileids(categories=['ArticlesItem']
            )]

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

    STOPWORDS.update(stopwords)

    # create
    corpus = PickledCorpusReader('path\\to\\preprocessed_corpus\\basic_corpus')

    clusterer = HierarchicalTopics(corpus)
    clusterer.cluster(corpus, n_clusters=30)

    with open('models/' + 'HierarchicalTopics_30.pkl', 'wb') as fobj:
        pickle.dump(clusterer, fobj)

    for idx, fileid in enumerate(corpus.fileids(categories=['ArticlesItem'])):
        print(clusterer.labels[idx], fileid)

    # # visualize
    # from collections import Counter
    # corpus = PickledCorpusReader('path\\to\\preprocessed_corpus\\subset_corpus')

    # labels = []
    # for fileid in corpus.fileids(categories=['ArticlesItem']):
    #     terms = []
    #     for term, count in Counter(list(normalize(corpus.words(fileid)))).most_common(10):
    #         terms.append(term)
    #     labels.append(terms)

    # with open('models/HierarchicalTopics.pkl', 'rb') as f:
    #     clusterer = pickle.load(f)
        # clusterer.plot_dendrogram(labels=labels, leaf_font_size=9)
