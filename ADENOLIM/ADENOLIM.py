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
import copy
import pandas as pd

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



model_name = "pipeline_complete_antibac_aac_nobatch"

std_name = "std_" + model_name + "_model"
mean_name = "mean_" + model_name + "_model"

json_file = open('./' + model_name + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
#loaded_model.load_weights("./models/" + model_name + ".h5")
loaded_model.load_weights("./" + model_name + ".h5")
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

path = path = "C:/Users/Peter/Desktop/MIT/code/models/" + std_name + ".pickle"
array_file = open(path, 'rb')
std = pickle.load(array_file)

path = path = "C:/Users/Peter/Desktop/MIT/code/models/" + mean_name + ".pickle"
array_file = open(path, 'rb')
mean = pickle.load(array_file)

neg_nmat.extend(pos_nmat)

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
    if i == 11:
        w4 = weights
    i = i + 1

for k in list(range(numrows)):
    x = neg_nmat[k,]

    #forward pass

    h1 = np.maximum(0, np.dot(x,w1[0]) + w1[1])
    h2 = np.maximum(0, np.dot(h1,w2[0]) + w2[1])
    h3 = np.maximum(0, np.dot(h2,w3[0]) + w3[1])
    out = np.dot(h3,w4[0])
    num = w4[1]
    out = out + num[0]


    #backward pass

    dh3 = copy.deepcopy(w4[0].T)
    dh3 = dh3[0] #because it was a double array
    dh3[h3<=0] = 0

    dh2 = np.dot(dh3,w3[0].T)
    dh2[h2<=0] = 0

    dh1 = np.dot(dh2,w2[0].T)
    dh1[h1<=0] = 0

    dx = np.dot(dh1,w1[0].T)

    if k == 0:
        dx_nmat = [dx]
    else:
        dx_nmat = np.append(dx_nmat, [dx], axis=0)


dx_nmat_abs = np.absolute(dx_nmat)
summed_abs_slopes = np.sum(dx_nmat_abs, axis = 0)
summed_slopes = np.sum(dx_nmat, axis = 0)
summed_slopes_abs = np.absolute(summed_slopes)

dictionary_abs = dict(zip(keys, summed_abs_slopes.T))
dictionary = dict(zip(keys, summed_slopes.T))
print(dictionary)
dictionary_a = dict(zip(keys, summed_slopes_abs.T))
print(dictionary_a)

res_abs = sorted(dictionary_abs, key=dictionary_abs.get)
vals_abs = sorted(dictionary_abs.values())
dic_abs = dict(zip(res_abs, vals_abs))

print('\nsorted summed abs slopes\n')
print(dic_abs)

res = sorted(dictionary, key=dictionary.get)
vals = sorted(dictionary.values())
dic = dict(zip(res, vals))

print('\nsorted summed slopes\n')
print(dic)

res_a = sorted(dictionary_a, key=dictionary_a.get)
vals_a = sorted(dictionary_a.values())
dic_a = dict(zip(res_a, vals_a))

print('\nabs sorted summed slopes\n')
print(dic_a)


df = pd.DataFrame(dic_abs, index=[0])
name = model_name + '_summed_abs_slopes.csv'
df.to_csv(name, index=False)

df = pd.DataFrame(dic, index=[0])
name = model_name + '_summed_slopes.csv'
df.to_csv(name, index=False)

df = pd.DataFrame(dic_a, index=[0])
name = model_name + '_summed_slopes_abs.csv'
df.to_csv(name, index=False)
