import os
import librosa
import numpy as np
from torch.utils.data import Dataset

class AudioFeatureDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_paths = []
        self.labels = []

        genres_path = os.path.join(root_dir, 'genres')
        for genre in os.listdir(genres_path):
            genre_path = os.path.join(genres_path, genre)
            for file in os.listdir(genre_path):
                if file.endswith('.wav') or file.endswith('.au'):
                    self.file_paths.append(os.path.join(genre_path, file))
                    self.labels.append(genre)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean, label
