import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import defaultdict, Counter


def asymptotic_decay(learning_rate, t, max_iter):
    return learning_rate / (1 + t / (max_iter / 2))


class SOM(object):

    def __init__(self, x, y, input_len, initial_radius, learning_rate=0.5, sigma=1.0,
                 update_function=asymptotic_decay):

        self.radius = initial_radius
        self._learning_rate = learning_rate
        self._sigma = sigma
        self._input_len = input_len

        self.weights = np.random.rand(x, y, input_len) * 2 - 1

        self._activation_map = np.zeros((x, y))
        self._neigx = np.arange(x)
        self._neigy = np.arange(y)

        self._xx, self._yy = np.meshgrid(self._neigx, self._neigy)
        self._xx = self._xx.astype(float)
        self._yy = self._yy.astype(float)

        self._update_function = update_function



    def _activate(self, x):
        self._activation_map = self._distance_function(x, self.weights)

    def _distance_function(self, x, w):
        return np.linalg.norm(np.subtract(x, w), axis=-1)

    def winner_neuron(self, x):
        self._activate(x)
        return np.unravel_index(self._activation_map.argmin(), self._activation_map.shape)

    def neighborhood(self, c, sigma):
        """Campana gaussiana centrada en c."""
        d = 2*np.pi*sigma*sigma
        ax = np.exp(-np.power(self._xx-self._xx.T[c], 2)/d)
        ay = np.exp(-np.power(self._yy-self._yy.T[c], 2)/d)
        return (ax * ay).T

    def update(self, x, w, t, max_iteration):
        eta = self._update_function(self._learning_rate, t, max_iteration)
        sigma = self._update_function(self._sigma, t, max_iteration)
        h = self.neighborhood(w, sigma)
        self.weights += np.einsum('ij, ijk->ijk', eta * h, x - self.weights)
        # ij, ijk son los labels de las dos matrices eta*h es 2D y x-W es 3D

    def train(self, training_data, max_iterations):
        for epoch in range(max_iterations):
            for i in range(len(training_data)):
                self.update(training_data[i], self.winner_neuron(training_data[i]), i, max_iterations)

        print("Training Finished")

    def labels_map(self, training_data, labels):
        winmap = defaultdict(list)
        for x, l in zip(training_data, labels):
            winmap[self.winner_neuron(x)].append(l)
        for position in winmap:
            winmap[position] = Counter(winmap[position])
        return winmap

    def distance_map(self):
        u = np.zeros((self.weights.shape[0], self.weights.shape[1], 8))

        ii = [[0, -1, -1, -1, 0, 1, 1, 1]] * 2
        jj = [[-1, -1, 0, 1, 1, 1, 0, -1]] * 2

        for x in range(self.weights.shape[0]):
            for y in range(self.weights.shape[1]):
                w_2 = self.weights[x, y]
                e = y % 2 == 0  # only used on hexagonal topology
                for k, (i, j) in enumerate(zip(ii[e], jj[e])):
                    if x + i >= 0 and x + i < self.weights.shape[0] and y + j >= 0 and y + j < self.weights.shape[1]:
                        w_1 = self.weights[x + i, y + j]
                        u[x, y, k] = np.linalg.norm(w_2 - w_1)

        u = u.sum(axis=2)
        return u / u.max()


df = pd.read_csv('europe.csv')
X_cols = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']

som = SOM(6, 6, 7, 3)

data = df[X_cols]
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
data = data.values

som.train(data, 1000)
winner_coordinates = np.array([som.winner_neuron(x) for x in data]).T

country_map = som.labels_map(data, df['Country'])

plt.figure(figsize=(14, 14))
for p, countries in country_map.items():
    countries = list(countries)
    x = p[0] + .1
    y = p[1] - .1
    for i, c in enumerate(countries):
        off_set = (i + 1) / len(countries) - 0.1
        plt.text(x, y + off_set, c, fontsize=20)
plt.pcolor(som.distance_map().T, cmap='gray_r', alpha=.5)
plt.xticks(np.arange(6 + 1))
plt.yticks(np.arange(6 + 1))
plt.grid()
plt.show()


W = som.weights
plt.figure(figsize=(14, 14))
for i, f in enumerate(X_cols):
    plt.subplot(3, 3, i+1)
    plt.title(f)
    plt.pcolor(W[:,:,i].T, cmap='coolwarm')
    plt.xticks(np.arange(6+1))
    plt.yticks(np.arange(6+1))
plt.tight_layout()
plt.show()

Z = np.zeros((6, 6))
plt.figure(figsize=(8, 8))
for i in np.arange(som.weights.shape[0]):
    for j in np.arange(som.weights.shape[1]):
        feature = np.argmax(W[i, j, :])
        plt.plot([j + .5], [i + .5], 'o', color='C' + str(feature),
                 marker='s', markersize=24)

legend_elements = [Patch(facecolor='C' + str(i),
                         edgecolor='w',
                         label=f) for i, f in enumerate(X_cols)]

plt.legend(handles=legend_elements,
           loc='center left',
           bbox_to_anchor=(1, .95))

plt.xlim([0, 6])
plt.ylim([0, 6])
plt.show()


