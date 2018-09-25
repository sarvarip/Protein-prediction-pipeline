from __future__ import division
import time
import math
import numpy as np
from random import shuffle
from keras.models import model_from_json
import pickle
import random
import numpy as np
from keras.optimizers import Adam
import inputoutput as IO
import pandas as pd
from sklearn.manifold import TSNE
from keras import backend as K

def num_vector_from_descriptor_vector(descriptor_vector):
    '''
    Given a sequence (map) of ("descriptor_name" -> value), returns a vector of values
    :param descriptor_vector:
    :return:
    '''
    x = []
    for k, v in descriptor_vector.items():
        x.append(v)
    return x



model_name = "tnse_antibac_aac"

std_name = "std_" + model_name + "_model"
mean_name = "mean_" + model_name + "_model"

json_file = open('./models/' + model_name + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
#loaded_model.load_weights("./models/" + model_name + ".h5")
loaded_model.load_weights("./models/" + model_name + ".h5")
print("Loaded model from disk")

optim = Adam(lr=0.01, beta_1=0.95)

loaded_model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])

#absolute mean is zero, since we normalized the data

name = 'neg_' + model_name
neg_dvec = IO.deserialize_descriptor_vector(name)
name = 'pos_' + model_name
pos_dvec = IO.deserialize_descriptor_vector(name)

keys = []
for k in neg_dvec[1]:
    keys.append(k)


pos_nmat = []
for dvec in pos_dvec:
    if dvec is None:
        continue
    pos_nvec = num_vector_from_descriptor_vector(dvec)
    pos_nmat.append(pos_nvec)

neg_nmat = []
for dvec in neg_dvec:
    if dvec is None:
        continue
    neg_nvec = num_vector_from_descriptor_vector(dvec)
    neg_nmat.append(neg_nvec)

path = "./models/" + std_name + ".pickle"
array_file = open(path, 'rb')
std = pickle.load(array_file)

path = "./models/" + mean_name + ".pickle"
array_file = open(path, 'rb')
mean = pickle.load(array_file)

#neg_nmat.extend(pos_nmat)

neg_nmat = np.array(neg_nmat)

neg_nmat -= mean
neg_nmat /= std

numrows = len(neg_nmat)
numcols = len(neg_nmat[0])
#print(numrows)
#print(numcols)

i = 1

for layer in loaded_model.layers:
    weights = layer.get_weights() # list of numpy arrays
    if i == 2:
        w1 = weights
    if i == 5:
        w2 = weights
    if i == 8:
        w3 = weights
    i = i + 1

for k in list(range(numrows)):
    x = neg_nmat[k,]

    #forward pass

    h1 = np.maximum(0, np.dot(x,w1[0]) + w1[1])
    h2 = np.maximum(0, np.dot(h1,w2[0]) + w2[1])
    h3 = np.maximum(0, np.dot(h2,w3[0]) + w3[1])

    if k == 0:
        neg_compressed = [h3]
    else:
        neg_compressed = np.append(neg_compressed, [h3], axis=0)

# get_10th_layer_output = K.function([loaded_model.layers[0].input],
#                                   [loaded_model.layers[10].output])
# layer_output = get_10th_layer_output([neg_nmat])[0]
# layer_output
# does not seem to work in Theano

pos_nmat = np.array(pos_nmat)

pos_nmat -= mean
pos_nmat /= std

numrows = len(pos_nmat)
numcols = len(pos_nmat[0])
#print(numrows)
#print(numcols)

i = 1

for layer in loaded_model.layers:
    weights = layer.get_weights() # list of numpy arrays
    if i == 2:
        w1 = weights
    if i == 5:
        w2 = weights
    if i == 8:
        w3 = weights
    i = i + 1

for k in list(range(numrows)):
    x = pos_nmat[k,]

    #forward pass

    h1 = np.maximum(0, np.dot(x,w1[0]) + w1[1])
    h2 = np.maximum(0, np.dot(h1,w2[0]) + w2[1])
    h3 = np.maximum(0, np.dot(h2,w3[0]) + w3[1])

    if k == 0:
        pos_compressed = [h3]
    else:
        pos_compressed = np.append(pos_compressed, [h3], axis=0)

df = pd.DataFrame(neg_nmat)
name = model_name + '_original_dim_neg.csv'
df.to_csv(name, index=False, header=False)

df = pd.DataFrame(pos_nmat)
name = model_name + '_origial_dim_pos.csv'
df.to_csv(name, index=False, header=False)

df = pd.DataFrame(neg_compressed)
name = model_name + '_reduced_dim_neg.csv'
df.to_csv(name, index=False, header=False)

df = pd.DataFrame(pos_compressed)
name = model_name + '_reduced_dim_pos.csv'
df.to_csv(name, index=False, header=False)
