import pickle

from doc2vec_clustering import TextNormalizer
from doc2vec_clustering import GensimVectorizer

from sklearn.decomposition import PCA

from preprocessed_corpus_reader.reader import PickledCorpusReader

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

labels = ["not_escalated", "BEMS","RRR", "CAP"]
reader = PickledCorpusReader('path\\to\\preprocessed_data\\preprocessed_min')

trg_docs = reader.docs(categories=labels)

# normalize
normalizer = TextNormalizer()
normalized_docs = normalizer.transform(trg_docs)

# vectorize
vectorizer = GensimVectorizer(path='path\\to\\models\\doc2vec_complete\\doc2vec.d2v')
vectorizer.fit(normalized_docs)
vectorized_docs = vectorizer.transform(normalized_docs)

clusterer = pickle.load(open('models/unsupervised/Kmeans.pkl', 'rb'))
clusterer.fit(vectorized_docs)

labels = ['c{}'.format(c) for c in clusterer.labels_]

# pca
pca = PCA(n_components=2)
# Fit and transform the data to the model
reduced_data_pca = pca.fit_transform(vectorized_docs)
# print(reduced_data_pca)

# Hardcoded scatterplot
for i in range(0, reduced_data_pca.shape[0]):
    if labels[i] == 'c0':
        c0 = plt.scatter(reduced_data_pca[i,0],reduced_data_pca[i,1],c=['230', '25', '75'], marker='+')
    elif labels[i] == 'c1':
        c1 = plt.scatter(reduced_data_pca[i,0],reduced_data_pca[i,1],c=['60', '180', '75'],marker='o')
    elif labels[i] == 'c2':
        c2 = plt.scatter(reduced_data_pca[i,0],reduced_data_pca[i,1],c=['255', '225', '25'],marker='h')
    elif labels[i] == 'c3':
        c3 = plt.scatter(reduced_data_pca[i,0],reduced_data_pca[i,1],c=['0', '130', '200'],marker='.')
    elif labels[i] == 'c4':
        c4 = plt.scatter(reduced_data_pca[i,0],reduced_data_pca[i,1],c=['245', '130', '48'],marker='D')
    elif labels[i] == 'c5':
        c5 = plt.scatter(reduced_data_pca[i,0],reduced_data_pca[i,1],c=['145', '30', '180'],marker='<')
    elif labels[i] == 'c6':
        c6 = plt.scatter(reduced_data_pca[i,0],reduced_data_pca[i,1],c=['70', '240', '240'],marker='>')
    elif labels[i] == 'c7':
        c7 = plt.scatter(reduced_data_pca[i,0],reduced_data_pca[i,1],c=['240', '50', '230'],marker='|')

plt.legend([c0, c1, c2, c3, c4, c5, c6, c7], ['C0','C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'])
plt.title('Clusters')
plt.show()

# x = np.array([a[0] for a in reduced_data_pca[0:len(reduced_data_pca)]])
# y = np.array([a[1] for a in reduced_data_pca[0:len(reduced_data_pca)]])
# colors = np.random.rand(len(reduced_data_pca))
# area = np.pi * (15 * np.random.rand(len(reduced_data_pca)))

# plt.scatter(x,  y, s=area, c=colors, alpha=0.5)
# plt.show()

# print([a[0] for a in reduced_data_pca[0:len(reduced_data_pca)]])
# print([a[1] for a in reduced_data_pca[0:len(reduced_data_pca)]])


