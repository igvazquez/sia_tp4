from OjasRuleNeuron import OjasRuleNeuron
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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


std_df = StandardScaler().fit_transform(df[X_cols])

ORN = OjasRuleNeuron()
FPC,history = ORN.fit(learn_factor,std_df,max_epochs)

pca = PCA(n_components=7)
components = pca.fit_transform(std_df)
print('Library first principal component:\n',pca.components_[0])