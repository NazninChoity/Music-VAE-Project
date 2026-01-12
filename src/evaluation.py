import pandas as pd

def save_metrics(method, silhouette, calinski_harabasz, path):
    df = pd.DataFrame([{
        'method': method,
        'silhouette': silhouette,
        'calinski_harabasz': calinski_harabasz
    }])
    df.to_csv(path, index=False)
