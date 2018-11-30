import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from kmeans import KMeans
import utils

if __name__ == "__main__":
    data = pd.read_csv('seed.txt', sep='\t', header=None)
    x = data.drop([7], axis=1)
    print(x.head())
    x = x.values
    x = utils.normalize(x)

    kmean = KMeans(n_cluster=3, init_pp=True)
    kmean.fit(x)

    label = kmean.predict(x)
    # print('predicted', np.unique(label))
    score = silhouette_score(x, label)
    print('Silhouette: ', score)
    print('SSE: ', kmean.SSE)
