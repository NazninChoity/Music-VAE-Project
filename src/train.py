import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.vae import VAE

def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(model, dataset, epochs=10, batch_size=32, lr=1e-3):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for batch, labels in dataloader:
            batch = batch.float()
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss = loss_function(recon_batch, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")



import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def vae_loss(recon_x, x, mu, logvar, recon_type='mse'):
    if recon_type == 'mse':
        recon = F.mse_loss(recon_x, x, reduction='sum')
    else:
        recon = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kld

def train_conv_vae(model, dataset, epochs=10, batch_size=32, lr=1e-3, device='cpu'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0.0
        for x, _ in dataloader:
            # x: (B, 64, 128) -> (B,1,64,128)
            x = torch.tensor(x).float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            x_hat, mu, logvar = model(x)
            loss = vae_loss(x_hat, x, mu, logvar, recon_type='mse')
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.2f}")

