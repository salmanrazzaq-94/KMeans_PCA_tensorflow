import numpy as np
import tensorflow as tf
import os

#Loading the data
dfile = "data.npy"
data = np.load(dfile)
print("shape of data:", data.shape)
vectorized = data.reshape((-1,60))
vectorized = np.float32(vectorized)
print(vectorized.shape)

num_points = vectorized[0]
dimensions = vectorized[1]
points = vectorized

def input_fn():
  return tf.compat.v1.train.limit_epochs(
      tf.convert_to_tensor(points, dtype=tf.float32), num_epochs=1)

num_clusters = 7
kmeans = tf.compat.v1.estimator.experimental.KMeans(
    num_clusters=num_clusters, use_mini_batch=False)

# train
num_iterations = 4
previous_centers = None
for _ in range(num_iterations):
  kmeans.train(input_fn)
  cluster_centers = kmeans.cluster_centers()
  if previous_centers is not None:
    print ('delta:', cluster_centers - previous_centers)
  previous_centers = cluster_centers
  print ('score:', kmeans.score(input_fn))
print ('cluster centers shape:', cluster_centers.shape)

# # map the input points to their clusters
cluster_indices = list(kmeans.predict_cluster_index(input_fn)) 
# for i, point in enumerate(points):
#   cluster_index = cluster_indices[i]
#   center = cluster_centers[cluster_index]
#   print ('band:', i+1, 'is in cluster', cluster_index)

image = np.array(cluster_indices).reshape(1800, 1135)
print(image.shape)

