import os
import librosa
import numpy as np
from torch.utils.data import Dataset

class AudioFeatureDataset(Dataset):
    def __init__(self, root_dir, feature_type='mfcc', n_mfcc=20, sr=22050, n_fft=1024, hop_length=512):
        """
        feature_type: 'mfcc' or 'mel_spec'
        """
        self.root_dir = root_dir
        self.feature_type = feature_type
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.file_paths = []
        self.labels = []
        genres_path = os.path.join(root_dir, 'genres')
        for genre in sorted(os.listdir(genres_path)):
            genre_path = os.path.join(genres_path, genre)
            if not os.path.isdir(genre_path):
                continue
            for file in os.listdir(genre_path):
                if file.endswith(('.wav', '.au')):
                    self.file_paths.append(os.path.join(genre_path, file))
                    self.labels.append(genre)

    def __len__(self):
        return len(self.file_paths)

    def _mfcc(self, y):
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc)
        return np.mean(mfcc.T, axis=0)  # shape: (n_mfcc,)

    def _mel_spec(self, y):
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # Normalize and resize to fixed shape (64 x 128) by trimming/padding time axis
        T = mel_db.shape[1]
        target_T = 128
        if T >= target_T:
            mel_db = mel_db[:, :target_T]
        else:
            pad = np.zeros((mel_db.shape[0], target_T - T))
            mel_db = np.concatenate([mel_db, pad], axis=1)
        # Scale to [0,1]
        mel_min, mel_max = mel_db.min(), mel_db.max()
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-8)
        return mel_norm.astype(np.float32)  # shape: (64, 128)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        y, _ = librosa.load(file_path, sr=self.sr)
        if self.feature_type == 'mfcc':
            x = self._mfcc(y)  # (n_mfcc,)
        elif self.feature_type == 'mel_spec':
            x = self._mel_spec(y)  # (64, 128)
        else:
            raise ValueError("feature_type must be 'mfcc' or 'mel_spec'")
        return x, label
