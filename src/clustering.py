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


import numpy as np
import torch
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
import umap
import matplotlib.pyplot as plt

def get_latents(model, dataset, device='cpu', is_conv=False):
    model.eval()
    latents = []
    with torch.no_grad():
        for x, _ in dataset:
            if is_conv:
                x_tensor = torch.tensor(x).float().unsqueeze(0).unsqueeze(0).to(device)  # (1,1,64,128)
                _, mu, _ = model(x_tensor)
            else:
                x_tensor = torch.tensor(x).float().unsqueeze(0).to(device)  # (1,D)
                _, mu, _ = model(x_tensor)
            latents.append(mu.cpu().numpy().flatten())
    return np.array(latents)

def kmeans_and_metrics(Z, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(Z)
    sil = silhouette_score(Z, labels)
    ch = calinski_harabasz_score(Z, labels)
    db = davies_bouldin_score(Z, labels)
    return labels, sil, ch, db

def agglomerative_and_metrics(Z, n_clusters=10):
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agg.fit_predict(Z)
    sil = silhouette_score(Z, labels)
    ch = calinski_harabasz_score(Z, labels)
    db = davies_bouldin_score(Z, labels)
    return labels, sil, ch, db

def dbscan_and_metrics(Z, eps=0.5, min_samples=5):
    dbs = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbs.fit_predict(Z)
    # DBSCAN may assign -1 for noise; metrics require >=2 clusters
    valid = len(set(labels)) > 1 and (set(labels) - {-1})
    if valid:
        sil = silhouette_score(Z, labels)
        ch = calinski_harabasz_score(Z, labels)
        db = davies_bouldin_score(Z, labels)
    else:
        sil = ch = db = np.nan
    return labels, sil, ch, db

def adjusted_rand(labels_pred, labels_true):
    return adjusted_rand_score(labels_true, labels_pred)

def visualize_umap(Z, labels, save_path):
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(Z)
    plt.figure(figsize=(8,6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', s=10)
    plt.title("UMAP of VAE Latents")
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()
    return embedding
