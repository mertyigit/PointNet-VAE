import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define the VAE model
class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, device, latent_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.device = device


    def forward(self, x):
        # Encode the input to get mu and logvar
        #embeddings = self.encoder(x)    
        embeddings, _, _ = self.encoder(x)  ## If PointNet Encoder is used      
        reconstructed_x = self.decoder(embeddings)
        
        return reconstructed_x