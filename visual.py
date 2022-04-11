import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

with open('model.pickle','rb') as file:
    best_model = pickle.load(file)

#visualize W1, b1, W2, b2
W1 = best_model.W1
W2 = best_model.W2
b1 = best_model.b1
b2 = best_model.b2

W1_plot = sns.heatmap(W1)
b1_plot = sns.heatmap(b1, cmap="GnBu")
W2_plot = sns.heatmap(W2)
b2_plot = sns.heatmap(b2, cmap="GnBu")

plt.show()
