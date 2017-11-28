import unicodedata
import nltk
import pickle

from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn

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

class SklearnTopicModels(object):

    def __init__(self, n_topics=50, custom_stopwords=[]):
        """
        n_topics is the desired number of topics
        """
        self.n_topics = n_topics
        self.model = Pipeline([
            ('norm', TextNormalizer(custom_stopwords=custom_stopwords)),
            ('tfidf', CountVectorizer(tokenizer=identity,
                                      preprocessor=None, lowercase=False)),
            ('model', LatentDirichletAllocation(n_topics=self.n_topics)),
        ])

    def fit_transform(self, documents):
        self.model.fit_transform(documents)

        return self.model
    
    def get_topics(self, n=25):
        """
        n is the number of top terms to show for each topic
        """
        vectorizer = self.model.named_steps['tfidf']
        model = self.model.steps[-1][1]
        names = vectorizer.get_feature_names()
        topics = dict()

        for idx, topic in enumerate(model.components_):
            features = topic.argsort()[:-(n - 1): -1]
            tokens = [names[i] for i in features]
            topics[idx] = tokens

        return topics

if __name__ == '__main__':
    fp = open('stopwords.txt', 'r')
    stopwords = fp.read()
    stopwords = stopwords.split('\n')
    fp.close()

    corpus = PickledCorpusReader('path\\to\\preprocessed_corpus\\basic_corpus')

    lda = SklearnTopicModels(n_topics=30, custom_stopwords=stopwords)
    documents = corpus.docs()

    lda.fit_transform(documents)
    topics = lda.get_topics()

    with open('models/topics/' + 'LDA' + '.pkl', 'wb') as fobj:
        pickle.dump(lda, fobj)

    for topic, terms in topics.items():
        print("Topic #{}:".format(topic+1))
        print(terms)
