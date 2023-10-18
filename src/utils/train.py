'''
Developed from scrtach by Mert Sengul.
Please cite the repo if you readapt.
'''

import torch
from tqdm import tqdm
import numpy as np
import os

class Trainer:
    '''
    Trainer object.
    '''
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        encoder_type,
        model_type,
        checkpoint,
        experiment,
        device,
        kl_loss_weight=None,
    ):

        super().__init__()

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.encoder_type = encoder_type
        self.model_type = model_type
        self.checkpoint = checkpoint
        self.experiment = experiment
        self.device = device
        self.kl_loss_weight = kl_loss_weight
    
    def fit(self, train_loader, val_loader, epochs):
        for epoch in tqdm(range(epochs)):
            # train
            train_loss, train_rc_loss, train_kl_loss = self._train(train_loader)
            print('Epoch: {} - Loss: {} - Reconst Loss: {} - KL Loss: {}'.format(epoch, train_loss, train_rc_loss, train_kl_loss))
            
            # validate
            val_loss, val_rc_loss, val_kl_loss = self._validate(val_loader)
            print('Epoch: {} - Loss: {} - Reconst Loss: {} - KL Loss: {}'.format(epoch, val_loss, val_rc_loss, val_kl_loss))

            #save model state
            self._save_checkpoint(train_loss, val_loss, epoch)

    def _save_checkpoint(self, train_loss, val_loss, epoch):
        path = '{}/{}'.format(self.checkpoint, self.experiment)
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(self.model.state_dict(), '{}/checkpoint_{}.pth'.format(path, epoch))

    def _sanitizer(self, data):
        ### Preparate the 3D cloud for encoder ###
        
        batch_size = data.y.shape[0]
        
        if self.encoder_type == 'ConvolutionEncoder':
            points = torch.stack([data[idx].pos for idx in range(batch_size)]).unsqueeze(1) ## If Convolution Encoder
        elif self.encoder_type == 'PointNetEncoder':
            points = torch.stack([data[idx].pos for idx in range(batch_size)]).transpose(2, 1) ## If PointNet Encoder

        targets = data.y

        return points, targets, batch_size


    def _train(self, loader):
        # put model in train mode
        _loss = []
        _rc_loss = []
        _kl_loss = []
        kl_divergence = torch.zeros(1).to(self.device)
        self.model.to(self.device)
        self.model.train()

        for i, data in tqdm(enumerate(loader)):            
            self.optimizer.zero_grad()

            points, target, batch_size = self._sanitizer(data) # No need to return

            points = points.to(self.device)
            target = target.to(self.device)
            
            if self.model_type == 'VAE':
                reconstructed_x, mu, logvar = self.model(points)
                kl_divergence = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
            
            elif self.model_type == 'AutoEncoder':
                reconstructed_x, _, _ = self.model(points)
            
            else:
                print('The model stype is not known!')

            if self.encoder_type == 'ConvolutionEncoder':
                loss_reconstruction = self.criterion(reconstructed_x, points.squeeze(1))
            elif self.encoder_type == 'PointNetEncoder':
                loss_reconstruction = self.criterion(reconstructed_x, points.transpose(2, 1))

            loss = loss_reconstruction + self.kl_loss_weight * kl_divergence
            loss.backward()
            self.optimizer.step()

            _loss.append(loss.item())
            _rc_loss.append(loss_reconstruction.item())
            _kl_loss.append(kl_divergence.item())

        epoch_loss = np.mean(_loss)
        rc_loss = np.mean(_rc_loss)
        kl_loss = np.mean(_kl_loss)
        
        return epoch_loss, rc_loss, kl_loss

    def _validate(self, loader):
        _loss = []
        _rc_loss = []
        _kl_loss = []
        kl_divergence = torch.zeros(1)
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
        

    