import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd

# Load the dataset
df = pd.read_csv("D:/machine learning project/diabetes.csv")

# Select all features except 'Outcome'
X = df.drop(columns=['Outcome']).dropna().values  # Drop rows with missing values

# Set random seed for reproducibility
np.random.seed(42)

# Fit the KMeans model
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

# Predict the clusters
y_kmeans = kmeans.predict(X)

# Reduce dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the data points with cluster labels
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, s=50, cmap="viridis")

# Plot the cluster centers in reduced dimensions
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c="black", s=200, alpha=0.5)

# Add labels and title
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("KMeans Clustering on Diabetes Data (PCA Visualization)")

# Show the plot
plt.show()
