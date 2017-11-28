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
from gensim.sklearn_integration import SklLsiModel
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel

from preprocessed_corpus_reader.reader import PickledCorpusReader

def identity(words):
    return words

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

class GensimTfidfVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, dirpath=".", tofull=False):
        """
        Gensim vectorizer
        """
        self._lexicon_path = os.path.join(dirpath, "corpus.dict")
        self._tfidf_path = os.path.join(dirpath, "tfidf.model")

        self.lexicon = None
        self.tfidf = None
        self.tofull = tofull

        self.load()

    def load(self):
        if os.path.exists(self._lexicon_path):
            self.lexicon = Dictionary.load(self._lexicon_path)

        if os.path.exists(self._tfidf_path):
            self.tfidf = TfidfModel().load(self._tfidf_path)

    def save(self):
        self.lexicon.save(self._lexicon_path)
        self.tfidf.save(self._tfidf_path)

    def fit(self, documents, labels=None):
        if self.lexicon == None or self.tfidf == None:
            inputDocuments = list(documents)
            self.lexicon = Dictionary(inputDocuments)
            self.tfidf = TfidfModel([
                self.lexicon.doc2bow(doc)
                for doc in inputDocuments],
                id2word=self.lexicon)
            self.save()
            return self
        else:
            return self

    def transform(self, documents):
        returnDocs = []
        for document in documents:
            vec = self.tfidf[self.lexicon.doc2bow(document)]
            if self.tofull:
                returnDocs.append(sparse2full(vec))
            else:
                returnDocs.append(vec)
        return returnDocs

class GensimTopicModels(object):

    def __init__(self, n_topics=50, estimator='LSA', custom_stopwords=[]):
        """
        n_topics is the desired number of topics
        """
        self.n_topics = n_topics
        if estimator == 'LSA':
            self.estimator = SklLsiModel(num_topics=self.n_topics)
        else:
            self.estimator = SklLdaModel(num_topics=self.n_topics)

        self.model = Pipeline([
            ('norm', TextNormalizer(custom_stopwords=custom_stopwords)),
            ('vect', GensimTfidfVectorizer()),
            ('model', self.estimator)
        ])

    def fit(self, documents):
        self.model.fit(documents)

        return self.model


if __name__ == '__main__':
    # fp = open('stopwords.txt', 'r')
    # stopwords = fp.read()
    # stopwords = stopwords.split('\n')
    # fp.close()

    # corpus = PickledCorpusReader('path\\to\\preprocessed_corpus\\subset_corpus')

    # gensim_lsa = GensimTopicModels(n_topics=30, custom_stopwords=stopwords)

    # docs = [
    #     list(corpus.docs(fileids=fileid))[0]
    #     for fileid in corpus.fileids()
    # ]

    # gensim_lsa.fit(docs)

    # # with open('models/topics/' + gensim_lsa.model.named_steps['model'].gensim_model.__class__.__name__ + '.pkl', 'wb') as fobj:
    # #     pickle.dump(gensim_lsa, fobj)

    # lsa = gensim_lsa.model.named_steps['model'].gensim_model
    # print(lsa.show_topics())

    # import pyLDAvis
    # import pyLDAvis.gensim

    gensim_lsa = pickle.load(open('path\\to\\models\\topics\\LsiModel.pkl', 'rb'))

    # pickled_corpus = PickledCorpusReader('path\\to\\App\\preprocessed_corpus\\basic_corpus')
    # docs = pickled_corpus.docs()

    # corpus = [
    #     gensim_lsa.model.named_steps['vect'].lexicon.doc2bow(doc)
    #     for doc in gensim_lsa.model.named_steps['norm'].transform(docs)
    # ]

    # lexicon = gensim_lsa.model.named_steps['vect'].lexicon

    # data = pyLDAvis.gensim.prepare(lsa, corpus, lexicon)
    # pyLDAvis.save_html(data, 'results/LSA_topics.html')

    # # visualize
    lsa = gensim_lsa.model.named_steps['model'].gensim_model
    vect = gensim_lsa.model.named_steps['vect'].lexicon
    # print(lsa.show_topic(0, topn=10))
    fp = open('results/lsa_results.txt', 'w', encoding='utf-8')
    for i in range(0,30):
        fp.write('Topic {}:\n'.format(i))
        for word in lsa.show_topic(i, topn=10):
            fp.write(str(vect[int(word[0])] + "  " ))
        fp.write('\n\n')