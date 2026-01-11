import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import umap
import matplotlib.pyplot as plt

def get_latents(model, dataset):
    model.eval()
    latents = []
    with torch.no_grad():
        for x, _ in dataset:
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            mu, logvar = model.encode(x_tensor)
            z = model.reparameterize(mu, logvar)
            latents.append(z.squeeze(0).numpy())
    return np.array(latents)

def kmeans_and_metrics(Z, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(Z)
    sil = silhouette_score(Z, labels)
    ch = calinski_harabasz_score(Z, labels)
    return labels, sil, ch

def visualize_umap(Z, labels, save_path):
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(Z)
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', s=10)
    plt.title("UMAP of VAE Latents")
    plt.savefig(save_path)
    plt.close()
    return embedding
