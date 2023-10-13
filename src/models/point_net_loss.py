''' Point Net Loss function which is essentially a regularized Focal Loss.
    Code was adapted from this repo:
        https://github.com/clcarwin/focal_loss_pytorch
    '''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# special loss for Classification: Focal Loss + regularization
class PointNetLoss(nn.Module):
    def __init__(self, alpha=None, gamma=0, reg_weight=0, size_average=True):
        super(PointNetLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reg_weight = reg_weight
        self.size_average = size_average

        # sanitize inputs
        if isinstance(alpha,(float, int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,(list, np.ndarray)): self.alpha = torch.Tensor(alpha)

        # get Balanced Cross Entropy Loss
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=self.alpha)
        

    def forward(self, predictions, targets, A=None):

        # get batch size
        bs = predictions.size(0)

        # get Balanced Cross Entropy Loss
        ce_loss = self.cross_entropy_loss(predictions, targets)

        # get predicted class probabilities for the true class
        pn = F.softmax(predictions)
        pn = pn.gather(1, targets.view(-1, 1)).view(-1)

        # get regularization term
        if self.reg_weight > 0:
            I = torch.eye(64).unsqueeze(0).repeat(A.shape[0], 1, 1) # .to(device)
            if A.is_cuda: I = I.cuda()
            elif A.is_mps: I = I.to(torch.device('mps'))
            reg = torch.linalg.norm(I - torch.bmm(A, A.transpose(2, 1)))
            reg = self.reg_weight*reg/bs
        else:
            reg = 0

        # compute loss (negative sign is included in ce_loss)
        loss = ((1 - pn)**self.gamma * ce_loss)
        if self.size_average: return loss.mean() + reg
        else: return loss.sum() + reg


# special loss for segmentation Focal Loss + Dice Loss
class PointNetSegLoss(nn.Module):
    def __init__(self, alpha=None, gamma=0, size_average=True, dice=False):
        super(PointNetSegLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.dice = dice

        # sanitize inputs
        if isinstance(alpha,(float, int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,(list, np.ndarray)): self.alpha = torch.Tensor(alpha)

        # get Balanced Cross Entropy Loss
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=self.alpha)
        

    def forward(self, predictions, targets, pred_choice=None):

        # get Balanced Cross Entropy Loss
        ce_loss = self.cross_entropy_loss(predictions.transpose(2, 1), targets)

        # reformat predictions (b, n, c) -> (b*n, c)
        predictions = predictions.contiguous() \
                                 .view(-1, predictions.size(2)) 
        # get predicted class probabilities for the true class
        pn = F.softmax(predictions)
        pn = pn.gather(1, targets.view(-1, 1)).view(-1)

        # compute loss (negative sign is included in ce_loss)
        loss = ((1 - pn)**self.gamma * ce_loss)
        if self.size_average: loss = loss.mean() 
        else: loss = loss.sum()

        # add dice coefficient if necessary
        if self.dice: return loss + self.dice_loss(targets, pred_choice, eps=1)
        else: return loss


    @staticmethod
    def dice_loss(predictions, targets, eps=1):
        ''' Compute Dice loss, directly compare predictions with truth '''

        targets = targets.reshape(-1)
        predictions = predictions.reshape(-1)

        cats = torch.unique(targets)

        top = 0
        bot = 0
        for c in cats:
            locs = targets == c

            # get truth and predictions for each class
            y_tru = targets[locs]
            y_hat = predictions[locs]

            top += torch.sum(y_hat == y_tru)
            bot += len(y_tru) + len(y_hat)


        return 1 - 2*((top + eps)/(bot + eps)) 

def pairwise_distances(a: torch.Tensor, b: torch.Tensor, p=2):
    """
    Compute the pairwise distance_tensor matrix between a and b which both have size [m, n, d]. The result is a tensor of
    size [m, n, n] whose entry [m, i, j] contains the distance_tensor between a[m, i, :] and b[m, j, :].
    :param a: A tensor containing m batches of n points of dimension d. i.e. of size [m, n, d]
    :param b: A tensor containing m batches of n points of dimension d. i.e. of size [m, n, d]
    :param p: Norm to use for the distance_tensor
    :return: A tensor containing the pairwise distance_tensor between each pair of inputs in a batch.
    """

    if len(a.shape) != 3:
        raise ValueError("Invalid shape for a. Must be [m, n, d] but got", a.shape)
    if len(b.shape) != 3:
        raise ValueError("Invalid shape for a. Must be [m, n, d] but got", b.shape)
    return (a.unsqueeze(2) - b.unsqueeze(1)).abs().pow(p).sum(3)

def chamfer(a, b):
    """
    Compute the chamfer distance between two sets of vectors, a, and b
    :param a: A m-sized minibatch of point sets in R^d. i.e. shape [m, n_a, d]
    :param b: A m-sized minibatch of point sets in R^d. i.e. shape [m, n_b, d]
    :return: A [m] shaped tensor storing the Chamfer distance between each minibatch entry
    """
    M = pairwise_distances(a, b)
    dist1 = torch.mean(torch.sqrt(M.min(1)[0]))
    dist2 = torch.mean(torch.sqrt(M.min(2)[0]))
    return (dist1 + dist2) / 2.0


def chamfer_distance(template: torch.Tensor, source: torch.Tensor):
	try:
		from .cuda.chamfer_distance import ChamferDistance
		cost_p0_p1, cost_p1_p0 = ChamferDistance()(template, source)
		cost_p0_p1 = torch.mean(torch.sqrt(cost_p0_p1))
		cost_p1_p0 = torch.mean(torch.sqrt(cost_p1_p0))
		chamfer_loss = (cost_p0_p1 + cost_p1_p0)/2.0
	except:
		chamfer_loss = chamfer(template, source)
	return chamfer_loss


class ChamferDistanceLoss(nn.Module):
	def __init__(self):
		super(ChamferDistanceLoss, self).__init__()

	def forward(self, template, source):
		return chamfer_distance(template, source)