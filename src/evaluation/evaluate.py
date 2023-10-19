'''
Developed from scrtach by Mert Sengul.
Please cite the repo if you readapt.
'''

import torch
from tqdm import tqdm
import numpy as np
import os

class Evaluater:
    '''
    Evaluater object.
    '''
    def __init__(
        self,
        model,
        criterion,
        encoder_type,
        model_type,
        checkpoint,
        device,
    ):

        super().__init__()

        self.model = model
        self.criterion = criterion
        self.encoder_type = encoder_type
        self.model_type = model_type
        self.checkpoint = checkpoint
        self.device = device
    
    
    def evaluate(self, holdout_loader):
        # evaluate
        eval_loss, eval_rc_loss, eval_kl_loss = self._evaluate(holdout_loader)
        print('Loss: {} - Reconst Loss: {} - KL Loss: {}'.format(eval_loss, eval_rc_loss, eval_kl_loss))

    def evaluate_data(self, data):
        _loss = []
        _rc_loss = []
        _kl_loss = []
        kl_divergence = torch.zeros(1)

        self.model.load_state_dict(torch.load(self.checkpoint, map_location=self.device))
        
        # put model in evaluation mode
        self.model.eval()

        with torch.no_grad():
            
            points, target, batch_size = self._sanitizer(data) # No need to return
            
            points = points.to(self.device)
            target = target.to(self.device)
            
            if self.model_type == 'VAE':
                reconstructed_x, mu, logvar = self.model(points)
                kl_divergence = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
            
            elif self.model_type == 'AutoEncoder':
                reconstructed_x, _, _ = self.model(points)
            
            if self.encoder_type == 'ConvolutionEncoder':
                loss_reconstruction = self.criterion(reconstructed_x, points.squeeze(1))
            
            elif self.encoder_type == 'PointNetEncoder':
                loss_reconstruction = self.criterion(reconstructed_x, points.transpose(2, 1))
            
            loss = loss_reconstruction + kl_divergence
            
            epoch_loss = loss.item()
            rc_loss = loss_reconstruction.item()
            kl_loss = kl_divergence.item()

        print('Loss: {} - Reconst Loss: {} - KL Loss: {}'.format(epoch_loss, rc_loss, kl_loss))
        return points, reconstructed_x

    def _evaluate(self, loader):
        _loss = []
        _rc_loss = []
        _kl_loss = []
        kl_divergence = torch.zeros(1)

        self.model.load_state_dict(torch.load(self.checkpoint, map_location=self.device))
        
        # put model in evaluation mode
        self.model.eval()

        with torch.no_grad():
            for i, data in tqdm(enumerate(loader)):   
                points, target, batch_size = self._sanitizer(data) # No need to return
                
                points = points.to(self.device)
                target = target.to(self.device)
                
                if self.model_type == 'VAE':
                    reconstructed_x, mu, logvar = self.model(points)
                    kl_divergence = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)

                elif self.model_type == 'AutoEncoder':
                    reconstructed_x, _, _ = self.model(points)

                if self.encoder_type == 'ConvolutionEncoder':
                    loss_reconstruction = self.criterion(reconstructed_x, points.squeeze(1))
                elif self.encoder_type == 'PointNetEncoder':
                    loss_reconstruction = self.criterion(reconstructed_x, points.transpose(2, 1))

                loss = loss_reconstruction + kl_divergence

                _loss.append(loss.item())
                _rc_loss.append(loss_reconstruction.item())
                _kl_loss.append(kl_divergence.item())

        epoch_loss = np.mean(_loss)
        rc_loss = np.mean(_rc_loss)
        kl_loss = np.mean(_kl_loss)

        return epoch_loss, rc_loss, kl_loss



    def _sanitizer(self, data):
        ### Preparate the 3D cloud for encoder ###
        
        batch_size = data.y.shape[0]
        
        if self.encoder_type == 'ConvolutionEncoder':
            points = torch.stack([data[idx].pos for idx in range(batch_size)]).unsqueeze(1) ## If Convolution Encoder
        elif self.encoder_type == 'PointNetEncoder':
            points = torch.stack([data[idx].pos for idx in range(batch_size)]).transpose(2, 1) ## If PointNet Encoder

        targets = data.y

        return points, targets, batch_size