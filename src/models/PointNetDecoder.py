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
# T-net (Spatial Transformer Network)
class Tnet(nn.Module):
    ''' T-Net learns a Transformation matrix with a specified dimension '''
    def __init__(self, dim, num_points=2500):
        super(Tnet, self).__init__()

        # dimensions for transform matrix
        self.dim = dim 

        self.conv1 = nn.Conv1d(dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, dim**2)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.max_pool = nn.MaxPool1d(kernel_size=num_points)
        

    def forward(self, x):
        bs = x.shape[0]

        # pass through shared MLP layers (conv1d)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))

        # max pool over num points
        x = self.max_pool(x).view(bs, -1)
        
        # pass through MLP
        x = self.bn4(F.relu(self.linear1(x)))
        x = self.bn5(F.relu(self.linear2(x)))
        x = self.linear3(x)

        # initialize identity matrix
        iden = torch.eye(self.dim, requires_grad=True).repeat(bs, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()
        elif x.is_mps:
            iden = iden.to(torch.device('mps'))
        x = x.view(-1, self.dim, self.dim) + iden

        return x


# ============================================================================
# Point Net Backbone (main Architecture)
class PointNetDecoder(nn.Module):
    def __init__(self, num_points=2500, num_global_feats=1024, local_feat=True):
        super(PointNetDecoder, self).__init__()

        self.num_points = num_points
        self.num_global_feats = num_global_feats
        self.local_feat = local_feat

        # Transposed convolution layers
        self.deconv1 = nn.ConvTranspose1d(self.num_global_feats, 128, kernel_size=1)
        self.deconv2 = nn.ConvTranspose1d(128, 64, kernel_size=1)
        self.deconv3 = nn.ConvTranspose1d(64, 64, kernel_size=1)
        self.deconv4 = nn.ConvTranspose1d(64, 64, kernel_size=1)
        self.deconv5 = nn.ConvTranspose1d(64, 3, kernel_size=1)

        # Transformation layer in the decoder (similar to T-Net)
        self.tnet_decoder = Tnet(dim=128, num_points=num_points)

        # Batch normalization for deconv layers
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(64)

    def forward(self, x):
        bs = x.shape[0]

        # Depending on whether local features were concatenated or not
        if self.local_feat:
            global_features = x[:, :self.num_global_feats, :]
        else:
            global_features = x

        # Expand the global features to match the shape of the local features
        global_features = global_features.unsqueeze(-1).repeat(1, 1, self.num_points)

        # Pass through the transformation layer (T-Net) in the decoder
        transformation_matrix = self.tnet_decoder(global_features)

        # Apply the transformation matrix to the input (similar to your encoder)
        x = torch.bmm(x.transpose(2, 1), transformation_matrix).transpose(2, 1)

        # Continue with the decoder layers
        x = self.bn1(F.relu(self.deconv1(x)))
        x = self.bn2(F.relu(self.deconv2(x)))
        x = self.bn3(F.relu(self.deconv3(x)))
        x = self.bn4(F.relu(self.deconv4(x)))

        # Final layer to generate point coordinates
        x = self.deconv5(x)

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

