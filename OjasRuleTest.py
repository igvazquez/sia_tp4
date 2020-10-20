from OjasRuleNeuron import OjasRuleNeuron
import numpy as np
import pandas as pd
import json

with open('./data/config.json') as json_file:
    data = json.load(json_file)
    for p in data['ej1b']:
        print('Learn Factor: ' + p['learn_factor'])
        print('Maximum epochs: ' + p['max_epochs'])
        print('')

learn_factor = float(p['learn_factor'])
max_epochs = int(p['max_epochs'])
df = pd.read_csv('europe.csv')
X_cols = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']

data = df[X_cols]
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
data = data.values

ORN = OjasRuleNeuron()
FPC,history = ORN.fit(learn_factor,data,max_epochs)

