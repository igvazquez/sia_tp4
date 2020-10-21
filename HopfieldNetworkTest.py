import numpy as np
from HopfieldNetwork import HopfieldNetwork
import json




def randomizePatterns(to_randomize, randomProb):
    out = []
    for i in range(len(to_randomize)):
        pattern = np.copy(patterns[i])
        rand = np.random.rand(len(pattern))
        for j in range(len(pattern)):
            pattern[j] = pattern[j] * -1 if rand[j] <= randomProb else pattern[j]
        out = np.append(out, pattern, axis=0)
    out = np.reshape(out, (len(to_randomize), len(to_randomize[0])))
    return out


def getPatternsFromFile(file):
    s = file.read()
    ps = []
    letter_count = 0
    letter_length = 0

    for j in range(len(s)):
        c = s[j]
        if c != '\n' and c != ',' and c != ' ':
            ps = np.append(ps, 1 if c == '#' else -1)
            if letter_count == 0:
                letter_length += 1
        elif c == ',':
            letter_count += 1

    return np.reshape(ps, (letter_count, letter_length))


with open('./data/config.json') as json_file:
    data = json.load(json_file)
    for p in data['ej2']:
        print('Patterns file: ' + p['patterns_file'])
        print('Random Probability: ' + p['random_prob'])
        print('Epsilon: ' + p['epsilon'])
        print('Maximum epochs: ' + p['max_epochs'])
        print('')

patterns_file_name = p['patterns_file']
randomProbability = float(p['random_prob'])
epsilon = float(p['epsilon'])
max_epochs = int(p['max_epochs'])
patterns_file = open("data/" + patterns_file_name, "r+")

patterns = getPatternsFromFile(patterns_file)

random_patterns = randomizePatterns(patterns, randomProbability)
#noisy_pattern = np.array([[-1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1]])
#random_patterns = np.append(random_patterns, noisy_pattern)
hpn = HopfieldNetwork(patterns)
outputs = hpn.classify(random_patterns, max_epochs, epsilon)
