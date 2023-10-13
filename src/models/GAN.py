import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, encoder, decoder, device, latent_dim):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.device = device

        # Define the layers for the mean and log-variance vectors
        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, self.latent_dim)
        
#        self.fc_decode = nn.Linear(self.latent_dim, self.encoder.num_global_feats)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        
        eps = torch.randn_like(std, device=self.device)
            
        z = mu + eps * std
        return z

    def forward(self, x):
        # Encode the input to get mu and logvar
        global_features, _, _ = self.encoder(x) ## If PointNet Encoder
        #global_features = self.encoder(x) ## If Convolution Encoder
        mu = self.fc_mu(global_features)
        logvar = self.fc_logvar(global_features)
            
        # Reparameterize and sample from the latent space
        z = self.reparameterize(mu, logvar)
        z.to(self.device)
        # Decode the sampled z to generate output
        
        reconstructed_x = self.decoder(z)
        
        return reconstructed_x, mu, logvar