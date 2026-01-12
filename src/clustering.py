import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import umap
import matplotlib.pyplot as plt

def get_latents(model, dataset):
    latents = []
    for x, _ in dataset:
        x_tensor = torch.tensor(x).float().unsqueeze(0)
        _, mu, _ = model(x_tensor)
        latents.append(mu.detach().numpy().flatten())
    return np.array(latents)

def kmeans_and_metrics(Z, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(Z)
    sil = silhouette_score(Z, labels)
    ch = calinski_harabasz_score(Z, labels)
    return labels, sil, ch

def visualize_umap(Z, labels, save_path):
    reducer = umap.UMAP(random_state=42)
    X2 = reducer.fit_transform(Z)
    plt.figure(figsize=(8,6))
    plt.scatter(X2[:,0], X2[:,1], c=labels, cmap='Spectral', s=10)
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()
    return X2
