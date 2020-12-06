import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris['feature_names'])
data = X[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']]

res = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=500).fit(data)
    data["clusters"] = kmeans.labels_
    res[k] = kmeans.inertia_
plt.figure()
plt.plot(list(res.keys()), list(res.values()))
plt.xlabel("n")
plt.ylabel("distance")
plt.show()

# so population = 3 is the correct answer.

