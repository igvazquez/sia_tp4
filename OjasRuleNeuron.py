import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera


class OjasRuleNeuron:

    def __init__(self):

        self.weights = None
        self.weights_history = None

        self.epochs = 0

    def fit(self, learn_factor, entries, limit):

        n_samples = entries.shape[0]
        n_features = entries.shape[1]

        self.weights = np.random.random_sample(n_features) * 2 - 1
        self.epochs = 0

        self.weights_history = [self.weights]
        print("Calculating...")
        print("⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜ 0%")
        for i in range(limit):

            for j in range(n_samples):
                y = np.dot(self.weights, entries[j, :])
                # print("y: ",y)
                delta_w = learn_factor * (y * entries[j, :] - np.power(y, 2) * self.weights)
                # print("delta w",delta_w)
                self.weights += delta_w
                # print("weights: ",self.weights)
                self.weights_history = np.concatenate((self.weights_history, [self.weights]))
            # self.errors_history = np.append(self.errors_history, [error])
            self.epochs += 1
            if i == int(limit / 4):
                print("⬛⬛⬜⬜⬜⬜⬜⬜⬜⬜ 25%")
            elif i == int(limit / 2):
                print("⬛⬛⬛⬛⬛⬜⬜⬜⬜⬜ 50%")
            elif i == int(3 * limit / 4):
                print("⬛⬛⬛⬛⬛⬛⬛⬜⬜⬜ 75%")

        print("⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛ 100%\n")
        print("Epochs: ", self.epochs)
        print("First PC: ", self.weights)
        return self.weights, self.epochs
