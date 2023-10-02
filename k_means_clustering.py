import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

dataset = pd.read_csv("./Clustering/Spotify/SpotifyFeatures.csv")

# removing the first 4 columns as they don't seem to provide any huge assist for the clustering
dataset = dataset.drop(columns=dataset.columns[0:4])

numerical_columns = dataset._get_numeric_data().columns
categorical_columns = []
for i in dataset.columns:
    if i not in numerical_columns:
        categorical_columns.append(i)

categorical_columns_label = []
categorical_columns_onehot = []

for i in categorical_columns:
    if len(dataset[i].value_counts()) > 2:
        categorical_columns_onehot.append(i)
    else:
        categorical_columns_label.append(i)

labelencoder = LabelEncoder()
for i in categorical_columns_label:
    dataset[i] = labelencoder.fit_transform(dataset[i])

a = 0
b = 0
dummy_variable_indices = []
ct = ColumnTransformer([("encoder", OneHotEncoder(), categorical_columns_onehot)], remainder="passthrough")

for i in categorical_columns_onehot:
    c = a+b
    dummy_variable_indices.append(c)
    a = c
    b = len(dataset[i].value_counts())

X = ct.fit_transform(dataset)
# removing the dummy variable
include_indices = []
for i in range(X.shape[1]):
    if i not in dummy_variable_indices:
        include_indices.append(i)

X = X[: , include_indices]

# now data preprocessing is done; let's move to the clustering step

wcss = []
for i in range(1,X.shape[1]):
    kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=300, random_state=0)
    kmeans.fit(X)
    # inertia method gives the squared distance between the points and the centroid
    wcss.append(kmeans.inertia_)

print(wcss)


# Plotting the elbow graph
plt.plot(range(1,X.shape[1]), wcss)
plt.title("The Elbow Graph")
plt.xlabel("No of Clusters")
plt.ylabel("WCSS")
plt.show()

# The elbow graph is ambugious in this case
# the elbow point is not clear in [6,7,8]

# So we have another method called Silhouette Score
# It is a better alternative for the elbow method to find the optimal number of clusters

from sklearn.metrics import silhouette_score

silhouette_scores = []

for i in range(2,X.shape[1]+1):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    print("The Silhoutte score for K=" , i , "is" , silhouette_score(X , y_kmeans))
    silhouette_scores.append(silhouette_score(X , y_kmeans))





