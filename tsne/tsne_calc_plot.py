import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

model_name = "tnse_antibac_aac"

name = model_name + '_original_dim_neg.csv'
neg_orig = pd.read_csv(name)

name = model_name + '_origial_dim_pos.csv'
pos_orig = pd.read_csv(name, header=None)

name = model_name + '_reduced_dim_neg.csv'
neg_red = pd.read_csv(name, header=None)

name = model_name + '_reduced_dim_pos.csv'
pos_red = pd.read_csv(name, header=None)

pos_length = pos_red.count()
neg_length = neg_red.count()
original = np.append(pos_orig, neg_orig, axis=0)
reduced = np.append(pos_red, neg_red, axis=0)

tsne = TSNE(n_components=2, verbose=1, perplexity=40)
tsne_results = tsne.fit_transform(reduced)

g1 = tsne_results[:pos_length[0],:].T
g2 = tsne_results[pos_length[0]+1:,:].T
data = (g1, g2)
colors = ("red", "black")
groups = ("positive", "negative")
sizes = (300, 30)
shapes = ("+", "x")

# Create plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for data, color, group, size, shape in zip(data, colors, groups, sizes, shapes):
    x, y = data
    ax.scatter(x, y, alpha=1, c=color, edgecolors='none', s=size, marker=shape, label=group)

plt.title('t-sne for last layer of neural network')
plt.legend(loc=2)
plt.show()

tsne = TSNE(n_components=2, verbose=1, perplexity=40)
tsne_results = tsne.fit_transform(original)

g1 = tsne_results[:pos_length[0],:].T
g2 = tsne_results[pos_length[0]+1:,:].T
data = (g1, g2)
colors = ("red", "black")
groups = ("positive", "negative")
sizes = (300, 30)
shapes = ("+", "x")

# Create plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for data, color, group, size, shape in zip(data, colors, groups, sizes, shapes):
    x, y = data
    ax.scatter(x, y, alpha=1, c=color, edgecolors='none', s=size, marker=shape, label=group)

plt.title('t-sne of the original data')
plt.legend(loc=2)
plt.show()
