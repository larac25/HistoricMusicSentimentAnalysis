import logging
import os
import pandas as pd
import gensim
from gensim.models import word2vec
from joblib.numpy_pickle_utils import xrange
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
import numpy as np

label = pd.read_pickle('/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/label/label.pkl')

kv = gensim.models.KeyedVectors.load_word2vec_format('/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP'
                                                     '/anno_corpus/corpus/label/label_ms')

print(label)

vocab = list(kv.key_to_index)
X = kv[vocab]

# gensim build in function for keyed vectors
#centroids = kv.rank_by_centrality(words=vocab)
#print(centroids)

num_clusters = 9  # 9 clusters because we have 9 labels

# Initalize a k-means object and use it to extract centroids

kmeans_clustering = KMeans(n_clusters=num_clusters, random_state=0)
idx = kmeans_clustering.fit_predict(X)

labels = kmeans_clustering.labels_
centroids = kmeans_clustering.cluster_centers_

score = kmeans_clustering.score(X)
print(score)
silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')

print("Silhouette_score: ")
print(silhouette_score)


word_centroid_map = dict(zip(vocab, idx))


for cluster in xrange(0, num_clusters):

    # Print the cluster number

    print("\nCluster %d" % cluster)

    # Find all of the words for that cluster number, and print them out

    words = []

    for x, y in word_centroid_map.items():

        if y == cluster:

            words.append(x)

    print(words)


# find the index and the distance of the closest points from x to each class centroid
close = pairwise_distances_argmin_min(centroids, X, metric='cosine')
index_closest_points = close[0]
distance_closest_points = close[1]

for i in range(0, num_clusters):
    print("The closest word to the centroid of class {0} is {1}, the distance is {2}".format
          (i, vocab[index_closest_points[i]], distance_closest_points[i]))


# todo: plot clusters
