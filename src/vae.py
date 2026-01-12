import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # define encoder and decoder here

    def encode(self, x):
        # return mu, logvar

    def reparameterize(self, mu, logvar):
        # return z

    def decode(self, z):
        # return reconstruction

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
