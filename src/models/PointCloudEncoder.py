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
class PointCloudEncoder(nn.Module):
    def __init__(self, latent_dim, num_point=2500, point_dim=3, bn_decay=0.5):
        self.num_point = num_point
        self.point_dim = point_dim 
        self.latent_dim = latent_dim
        super(PointCloudEncoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, point_dim), padding=0)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 1), padding=0)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=(1, 1), padding=0)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=(1, 1), padding=0)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, self.latent_dim, kernel_size=(1, 1), padding=0)
        self.bn5 = nn.BatchNorm2d(self.latent_dim)

        self.fc1 = nn.Linear(self.latent_dim, self.latent_dim)
        self.bn_fc1 = nn.BatchNorm1d(self.latent_dim)


    def forward(self, x):
        # Encoder
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        point_feat = nn.functional.relu(self.bn3(self.conv3(x)))
        x = nn.functional.relu(self.bn4(self.conv4(point_feat)))
        x = nn.functional.relu(self.bn5(self.conv5(x)))
        x = F.max_pool2d(x, kernel_size=(self.num_point, 1), padding=0)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.bn_fc1(self.fc1(x)))

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

