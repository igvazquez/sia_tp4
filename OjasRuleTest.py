from OjasRuleNeuron import OjasRuleNeuron
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
lib_components = pca.fit_transform(std_df)[:,0]
data_components = std_df.dot(FPC)
if lib_components[0]*data_components[0] < 0:
    data_components = data_components*-1
print("Oja's Rule first principal component:\n", FPC)
print('Library first principal component:\n',pca.components_[0])

df_2 = pd.DataFrame({'Pais': df['Country'], 'PCA1': data_components})
plt.figure(figsize=(7, 5))
plt.xticks(rotation=90)
plt.title("Using Oja's Rule First Principal Component")
sorted = df_2.sort_values(by='PCA1', ascending=True)
sns.barplot(x='Pais', y='PCA1', data=sorted)
plt.show()

df_3 = pd.DataFrame({'Pais': df['Country'], 'PCA1': lib_components})
plt.figure(figsize=(7, 5))
plt.xticks(rotation=90)
plt.title("Using Library's First Principal Component")
sorted = df_3.sort_values(by='PCA1', ascending=True)
sns.barplot(x='Pais', y='PCA1', data=sorted)

plt.show()
