import tensorflow as tf
import numpy as np
import os

#Loading the data
dfile = "data.npy"
data = np.load(dfile)
vectorized = data.reshape((-1,60))
vectorized = np.float32(vectorized)
print(vectorized.shape)

# PCA
def pca(x,dim = 2):
    with tf.name_scope("PCA"):
        m= 2043000
        n= 60
        #print(n)
        mean = tf.reduce_mean(x,axis=1)
        #print('mean',mean)
        x_new = x - tf.reshape(mean,(-1,1))
        #print('x_new', x_new)
        cov = tf.matmul(x_new,x_new,transpose_a=True)/(m - 1) 
        e,v = tf.linalg.eigh(cov,name="eigh")
        e_index_sort = tf.math.top_k(e,sorted=True,k=dim)[1]
        v_new = tf.gather(v,indices=e_index_sort)
        pca = tf.matmul(x_new,v_new,transpose_b=True)
    return pca

pca_data = tf.constant(vectorized)

pca_data = pca(pca_data,dim = 2)

print(pca_data.shape)

reduced_bands = pca_data.numpy().reshape((1800,1135,2))
print(reduced_bands.shape)
