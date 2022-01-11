import pandas as pd
import gensim
from gensim.scripts import word2vec2tensor
from gensim.models import word2vec
from joblib.numpy_pickle_utils import xrange
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances_argmin_min

# load word2vec files
wv_label = pd.read_pickle('/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/label/label_wv.pkl')
wv_path = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/label/label_wv'
wv_keyed_vectors = gensim.models.KeyedVectors.load_word2vec_format(wv_path)

#load fasttext files
ft_label = pd.read_pickle('/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/label/label_ft.pkl')
ft_path = '/Users/Lara/Desktop/Uni/Info_4/Masterarbeit/DATA/HMP/anno_corpus/corpus/label/label_ft'
ft_keyed_vectors = gensim.models.KeyedVectors.load_word2vec_format(ft_path)

tensorflow_input = input('create tensorflow files? y/n')

if tensorflow_input == 'y':

    wv_tensorflow = gensim.scripts.word2vec2tensor.word2vec2tensor(wv_path, tensor_filename='wv')
    ft_tensorflow = gensim.scripts.word2vec2tensor.word2vec2tensor(ft_path, tensor_filename='ft')

elif tensorflow_input == 'n':
    pass

# print labels
print(wv_label)
print(ft_label)

input_switch = input('perform cluster analysis for? wv = word2vec embeddings; ft = fasttext embeddings ')

if input_switch == 'wv':
    keyed_vectors = wv_keyed_vectors

elif input_switch == 'ft':
    keyed_vectors = ft_keyed_vectors

vocab = list(keyed_vectors.key_to_index)
X = keyed_vectors[vocab]

num_clusters = 9  # 9 clusters because we have 9 labels

# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans(n_clusters=num_clusters, random_state=0)
idx = kmeans_clustering.fit_predict(X)

labels = kmeans_clustering.labels_
print(labels)
centroids = kmeans_clustering.cluster_centers_
print('centroids: ')
print(centroids)

score = kmeans_clustering.score(X)
print(score)
silhouette_score = metrics.silhouette_score(X, labels, metric='cosine')

print('Silhouette_score: ')
print(silhouette_score)


word_centroid_map = dict(zip(vocab, idx))


for cluster in xrange(0, num_clusters):

    # Print the cluster number

    print('\nCluster %d' % cluster)

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
    print('The closest word to the centroid of class {0} is {1}, the distance is {2}'.format
          (i, vocab[index_closest_points[i]], distance_closest_points[i]))