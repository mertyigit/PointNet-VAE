'''
Contains classed comprising Point Net Architecture. Usage for each class can 
be found in main() at the bottom.

TO use: Import Classification and Segmentation classes into desired script



NOTE:
This architecture does not cover Part Segmentation. Per the Point Net paper 
that is a different architecture and is not implemented here.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
#class PointCloudDecoder(nn.Module):
#    def __init__(self, latent_dim, num_hidden, num_point=2500, point_dim=3, bn_decay=0.5):
#        self.num_point = num_point
#        self.point_dim = point_dim 
#        self.latent_dim = latent_dim
#
#        super(PointCloudDecoder, self).__init__()
#
#        self.upconv1 = nn.ConvTranspose2d(int(self.latent_dim/2), 512, kernel_size=(2, 2), stride=(2, 2), padding=0)
#        self.bn_upconv1 = nn.BatchNorm2d(512)
#
#        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=0)
#        self.bn_upconv2 = nn.BatchNorm2d(256)
#
#        self.upconv = nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
#        self.bn_upconv = nn.BatchNorm2d(256)
#
#        self.upconv3 = nn.ConvTranspose2d(256, 256, kernel_size=(4, 5), stride=(2, 3), padding=0)
#        self.bn_upconv3 = nn.BatchNorm2d(256)
#
#        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=(5, 7), stride=(3, 3), padding=0)
#        self.bn_upconv4 = nn.BatchNorm2d(128)
#
#        self.upconv5 = nn.ConvTranspose2d(128, 3, kernel_size=(1, 1), stride=(1, 1), padding=0)
#
#    def forward(self, x):
#
#        # UPCONV Decoder
#        x = x.view(x.size(0), -1, 1, 2)
#        x = self.bn_upconv1(nn.functional.relu(self.upconv1(x)))
#        x = self.bn_upconv2(nn.functional.relu(self.upconv2(x)))
#        x = self.bn_upconv(nn.functional.relu(self.upconv(x)))
#        x = self.bn_upconv(nn.functional.relu(self.upconv(x)))
#        x = self.bn_upconv(nn.functional.relu(self.upconv(x)))
#        x = self.bn_upconv(nn.functional.relu(self.upconv(x)))
#        x = self.bn_upconv(nn.functional.relu(self.upconv(x)))
#        x = self.bn_upconv(nn.functional.relu(self.upconv(x)))
#        x = self.bn_upconv(nn.functional.relu(self.upconv(x)))
#        x = self.bn_upconv(nn.functional.relu(self.upconv(x)))
#        x = self.bn_upconv3(nn.functional.relu(self.upconv3(x)))
#        x = self.bn_upconv4(nn.functional.relu(self.upconv4(x)))
#        x = self.upconv5(x)
#        x = x.view(x.size(0), -1, 3)
#
#        return x

class PointCloudDecoderMLP(nn.Module):
    def __init__(self, latent_dim, num_hidden, num_point=2500, point_dim=3, bn_decay=0.5):
        self.num_point = num_point
        self.point_dim = point_dim 
        self.latent_dim = latent_dim

        super(PointCloudDecoderMLP, self).__init__()

        self.fc1 = nn.Linear(self.latent_dim, self.latent_dim*2)
        self.fc2 = nn.Linear(self.latent_dim*2, self.latent_dim*4)
        self.fc3 = nn.Linear(self.latent_dim*4, self.latent_dim*8)
        self.fc4 = nn.Linear(self.latent_dim*8, self.latent_dim*16)
        self.fc5 = nn.Linear(self.latent_dim*16, self.latent_dim*32)
        self.fc6 = nn.Linear(self.latent_dim*32, self.latent_dim*32)
        self.fcend = nn.Linear(self.latent_dim*32, int(self.point_dim*self.num_point))

    def forward(self, x):
        # UPCONV Decoder
        #x = x.view(x.size(0), self.latent_dim, 1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = nn.functional.relu(self.fc4(x))
        x = nn.functional.relu(self.fc5(x))
        x = nn.functional.relu(self.fc6(x))
        x = self.fcend(x)
        x = x.reshape(-1, self.num_point, self.point_dim)
        return x
    

class PointCloudDecoderSelf(nn.Module):
    def __init__(self, latent_dim, num_hidden, num_point=2500, point_dim=3, bn_decay=0.5):
        self.num_point = num_point
        self.point_dim = point_dim 
        self.latent_dim = latent_dim

        super(PointCloudDecoderSelf, self).__init__()

        self.upconv1 = nn.ConvTranspose1d(self.latent_dim, self.latent_dim, kernel_size=3, stride=3, padding=1)
        self.bn_upconv1 = nn.BatchNorm1d(int(self.latent_dim))
        self.upconv2 = nn.ConvTranspose1d(self.latent_dim, self.latent_dim*2, kernel_size=3, stride=3, padding=1)
        self.bn_upconv2 = nn.BatchNorm1d(int(self.latent_dim*2))
        self.upconv3 = nn.ConvTranspose1d(self.latent_dim*2, self.latent_dim*4, kernel_size=3, stride=3, padding=1)
        self.bn_upconv3 = nn.BatchNorm1d(int(self.latent_dim*4))
        self.upconv4 = nn.ConvTranspose1d(self.latent_dim*4, self.latent_dim*8, kernel_size=3, stride=3, padding=1)
        self.bn_upconv4 = nn.BatchNorm1d(int(self.latent_dim*8))
        self.upconv5 = nn.ConvTranspose1d(self.latent_dim*8, self.latent_dim*16, kernel_size=3, stride=3, padding=1)
        self.bn_upconv5 = nn.BatchNorm1d(int(self.latent_dim*16))
        self.upconv6 = nn.ConvTranspose1d(self.latent_dim*16, self.latent_dim*32, kernel_size=3, stride=3, padding=1)
        self.bn_upconv6 = nn.BatchNorm1d(int(self.latent_dim*32))
        self.upconv7 = nn.ConvTranspose1d(self.latent_dim*32, self.latent_dim*32, kernel_size=3, stride=3, padding=0)

    def forward(self, x):
        # UPCONV Decoder
        x = x.view(x.size(0), self.latent_dim, 1)
        x = (nn.functional.relu(self.upconv1(x)))
        x = (nn.functional.relu(self.upconv2(x)))
        x = (nn.functional.relu(self.upconv3(x)))
        x = (nn.functional.relu(self.upconv4(x)))
        x = (nn.functional.relu(self.upconv5(x)))
        x = (nn.functional.relu(self.upconv6(x)))
        x = self.upconv7(x)

        return x

class PointCloudDecoder(nn.Module):
    def __init__(self, latent_dim, num_hidden, num_point=2500, point_dim=3, bn_decay=0.5):
        self.num_point = num_point
        self.point_dim = point_dim 
        self.latent_dim = latent_dim

        super(PointCloudDecoder, self).__init__()

        self.upconv1 = nn.ConvTranspose2d(int(self.latent_dim/2), int(self.latent_dim), kernel_size=(3, 3), stride=(2, 2), padding=0)
        self.bn_upconv1 = nn.BatchNorm2d(int(self.latent_dim))

        self.upconv2 = nn.ConvTranspose2d(int(self.latent_dim), int(self.latent_dim*4), kernel_size=(3, 3), stride=(2, 2), padding=0)
        self.bn_upconv2 = nn.BatchNorm2d(int(self.latent_dim*4))

        self.upconv3 = nn.ConvTranspose2d(int(self.latent_dim*4), int(self.latent_dim*8), kernel_size=(3, 3), stride=(2, 2), padding=0)
        self.bn_upconv3 = nn.BatchNorm2d(int(self.latent_dim*8))

        self.upconv4 = nn.ConvTranspose2d(int(self.latent_dim*8), 512, kernel_size=(3, 3), stride=(2, 2), padding=0)
        self.bn_upconv4 = nn.BatchNorm2d(512)
        
        self.fcconv = nn.ConvTranspose2d(512, 512, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn_fcconv = nn.BatchNorm2d(512)

        self.upconv5 = nn.ConvTranspose2d(512, 3, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn_upconv5 = nn.BatchNorm2d(3)

    def forward(self, x):

        # UPCONV Decoder
        x = x.view(x.size(0), -1, 1, 2)
        x = self.bn_upconv1(nn.functional.relu(self.upconv1(x)))
        x = self.bn_upconv2(nn.functional.relu(self.upconv2(x)))
        x = self.bn_upconv3(nn.functional.relu(self.upconv3(x)))
        x = self.bn_upconv4(nn.functional.relu(self.upconv4(x)))
        x = self.bn_fcconv(nn.functional.relu(self.fcconv(x)))
        x = self.bn_fcconv(nn.functional.relu(self.fcconv(x)))
        x = self.bn_fcconv(nn.functional.relu(self.fcconv(x)))
        x = self.bn_fcconv(nn.functional.relu(self.fcconv(x)))
        x = self.bn_fcconv(nn.functional.relu(self.fcconv(x)))
        x = self.bn_fcconv(nn.functional.relu(self.fcconv(x)))        
        x = self.upconv5(x)
        x = x.view(x.size(0), -1, 3)

        return x




# Test 
def main():
    test_data = torch.rand(32, 3, 2500)

    ## test T-net
    tnet = Tnet(dim=3)
    transform = tnet(test_data)
    print(f'T-net output shape: {transform.shape}')

    ## test backbone
    pointfeat = PointNetBackbone(local_feat=False)
    out, _, _ = pointfeat(test_data)
    print(f'Global Features shape: {out.shape}')

    pointfeat = PointNetBackbone(local_feat=True)
    out, _, _ = pointfeat(test_data)
    print(f'Combined Features shape: {out.shape}')

    # test on single batch (should throw error if there is an issue)
    pointfeat = PointNetBackbone(local_feat=True).eval()
    out, _, _ = pointfeat(test_data[0, :, :].unsqueeze(0))


if __name__ == '__main__':
    main()

