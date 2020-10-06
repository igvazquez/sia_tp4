import numpy as np


class HopfieldNetwork:

    def __init__(self, patterns):

        self.patterns = patterns
        self.N = len(patterns[0])
        self.weights = (np.dot(patterns.T, patterns) - (len(patterns) * np.identity(self.N))) / self.N

    def get_output(self, S, epsilon):
        h = np.dot(self.weights, S)
        # print("h:", h)
        output = np.array([])
        for i in range(len(S)):
            val = h[i]
            output = np.append(output, 1 if val > 0 else (S[i] if abs(val) <= epsilon else -1))
        return output

    def classify(self, input_patterns, max_epochs, epsilon):
        outputs = []

        for i in range(len(input_patterns)):
            epoch = 0
            print("Epoch ", epoch, ": ")

            old_S = input_patterns[i]
            printLetter(old_S)
            print("\n")
            epoch += 1
            print("Epoch ", epoch, ": ")
            new_S = self.get_output(old_S, epsilon)
            printLetter(new_S)
            print("\n")

            while (not np.array_equal(old_S, new_S)) and epoch <= max_epochs:
                old_S = np.copy(new_S)

                epoch += 1
                print("Epoch ", epoch, ": ")
                new_S = self.get_output(old_S, epsilon)
                printLetter(new_S)
                print("\n")
            print("------------------------------------------------\n")
            outputs = np.append(outputs, new_S)
        return outputs


def printLetter(pattern):
    for i in range(5):
        s = ""
        for j in range(5):
            val = pattern[i * 5 + j]
            s += "# " if val == 1 else ". "
        print(s)
