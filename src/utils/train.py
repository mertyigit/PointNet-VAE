'''
Developed from scrtach by Mert Sengul.
Please cite the repo if you readapt.
'''

import torch


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
    ):

        super().__init__()

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.encoder_type = encoder_type
        self.model_type = model_type,
        self.checkpoint = checkpoint,
        self.experiment = experiment,
        self.device = device
    
    def fit(self, train_loader, val_loader, epochs):
        for epoch in tqdm(range(epochs)):
            # train
            train_loss, train_rc_loss, train_kl_loss = self._train(train_loader)
            print('Epoch: {} - Loss: {} - Reconst Loss: {} - KL Loss: {}'.format(epoch, train_loss, train_rc_loss, train_kl_loss))
            
            # validate
            val_loss, val_rc_loss, val_kl_loss = self._validate(val_loader)
            print('Epoch: {} - Loss: {} - Reconst Loss: {} - KL Loss: {}'.format(epoch, val_loss, val_rc_loss, val_kl_loss))

            #save model state
            _save_checkpoint(train_loss, val_loss, epoch)

    def _save_checkpoint(self, train_loss, val_loss, epoch):
            torch.save(model.state_dict(), '{}/{}/checkpoint_{}.pth'.format(self.checkpoint, self.experiment, epoch))

    def _sanitizer(self, data):
        ### Preparate the 3D cloud for encoder ###
        
        self.batch_size = data.y.shape[0]
        
        if self.encoder_type == 'ConvolutionEncoder':
            self.points = torch.stack([data[idx].pos for idx in range(self.batch_size)]).unsqueeze(1) ## If Convolution Encoder
        elif self.encoder_type == 'PointNetEncoder':
            self.points = torch.stack([data[idx].pos for idx in range(self.batch_size)]).transpose(2, 1) ## If PointNet Encoder

        self.targets = data.y

        return self.points, self.target, self.batch_size


    def _train(self, loader):
        # put model in train mode
        _loss = []
        _rc_loss = []
        _kl_loss = []
        kl_divergence = torch.zeros(1)
        self.model.train()

        for i, data in tqdm(enumerate(loader)):            
            optimizer.zero_grad()

            points, target, batch_size = sanitizer(data) # No need to return


            if model_type == 'VAE':
                reconstructed_x, mu, logvar = self.model(points)
                kl_divergence = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
            
            elif model_type == 'AutoEncoder':
                reconstructed_x, _, _ = model(points)


            if encoder_type == 'ConvolutionEncoder':
                loss_reconstruction = self.criterion(reconstructed_x, points.squeeze(1))
            elif encoder_type == 'PointNetEncoder':
                loss_reconstruction = self.criterion(reconstructed_x, points.transpose(2, 1))

            loss = loss_reconstruction + kl_divergence
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
            points, target, batch_size = sanitizer(data) # No need to return

            if model_type == 'VAE':
                reconstructed_x, mu, logvar = self.model(points)
                kl_divergence = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
            
            elif model_type == 'AutoEncoder':
                reconstructed_x, _, _ = model(points)

            if encoder_type == 'ConvolutionEncoder':
                loss_reconstruction = self.criterion(reconstructed_x, points.squeeze(1))
            elif encoder_type == 'PointNetEncoder':
                loss_reconstruction = self.criterion(reconstructed_x, points.transpose(2, 1))

            loss = loss_reconstruction + kl_divergence

            _loss.append(loss.item())
            _rc_loss.append(loss_reconstruction.item())
            _kl_loss.append(kl_divergence.item())

        epoch_loss = np.mean(_loss)
        rc_loss = np.mean(_rc_loss)
        kl_loss = np.mean(_kl_loss)

        return epoch_loss, rc_loss, kl_loss
        

    