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

def identity(words):
    return words

class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, language='english'):
        self.stopwords  = set(nltk.corpus.stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def is_punct(self, token):
        return all(
            unicodedata.category(char).startswith('P') for char in token
        )

    def is_stopword(self, token):
        return token.lower() in self.stopwords

    def normalize(self, document):
        return [
            self.lemmatize(token, tag).lower()
            for paragraph in document
            for sentence in paragraph
            for (token, tag) in sentence
            if not self.is_punct(token) and not self.is_stopword(token)
        ]

    def lemmatize(self, token, pos_tag):
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
            yield self.normalize(document[0])

class GensimVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, path=None):
        self.path = path
        self.model = None
        self.load()

    def load(self):
        if os.path.exists(self.path):
            self.model = Doc2Vec.load(self.path)

    def save(self):
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

def create_pipeline(estimator, reduction=False):

    steps = [
        ('normalize', TextNormalizer()),
        ('vectorize', GensimVectorizer(path="path\\to\\doc2vec_complete\\doc2vec.d2v"))
    ]

    if reduction:
        steps.append((
            'reduction', TruncatedSVD(n_components=1000)
        ))

    # Add the estimator
    steps.append(('classifier', estimator))
    return Pipeline(steps)


labels = ["not_escalated", "BEMS","RRR", "CAP"]
reader = PickledCorpusReader('path\\to\\preprocessed_data\\preprocessed_complete')
loader = CorpusLoader(reader, 12, shuffle=True, categories=labels)

models = []

for form in (LogisticRegression, SGDClassifier):
    models.append(create_pipeline(form(), False))

models.append(create_pipeline(LinearSVC(), False))

import time
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def score_models(models, loader):
    for model in models:

        name = model.named_steps['classifier'].__class__.__name__
        if 'reduction' in model.named_steps:
            name += " (TruncatedSVD)"

        scores = {
            'model': str(model),
            'name': name,
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'time': [],
        }

        for X_train, X_test, y_train, y_test in loader:
            start = time.time()

            model.fit(X_train, y_train)

            with open('models/supervised/classifiers_complete'+ '_'+ model.named_steps['vectorize'].__class__.__name__ +'_' + model.named_steps['classifier'].__class__.__name__ + '.pkl', 'wb') as fobj:
                pickle.dump(model, fobj)
            gc.collect()

            y_pred = model.predict(X_test)

            scores['time'].append(time.time() - start)
            scores['accuracy'].append(accuracy_score(y_test, y_pred))
            scores['precision'].append(precision_score(y_test, y_pred, average='weighted'))
            scores['recall'].append(recall_score(y_test, y_pred, average='weighted'))
            scores['f1'].append(f1_score(y_test, y_pred, average='weighted'))

        yield scores

if __name__ == '__main__':
    for scores in score_models(models, loader):
        with open('results/classifiers/results_d2v_complete.json', 'a') as f:
            f.write(json.dumps(scores) + "\n")