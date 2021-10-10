# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable

# class FocalLoss(nn.Module):
#     """
#         This criterion is a implemenation of Focal Loss, which is proposed in
#         Focal Loss for Dense Object Detection.
#             Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
#         The losses are averaged across observations for each minibatch.
#         Args:
#             alpha(1D Tensor, Variable) : the scalar factor for this criterion
#             gamma(float, double) : gamma > 0; reduces the relative loss for well-clasified examples (p > .5),
#                                    putting more focus on hard, misclassified examples
#             size_average(bool): By default, the losses are averaged over observations for each minibatch.
#                                 However, if the field size_average is set to False, the losses are
#                                 instead summed for each minibatch.
#     """
#     def __init__(self, alpha, gamma=2, class_num=5,size_average=False):
#         super(FocalLoss, self).__init__()
#         if alpha is None:
#             self.alpha = Variable(torch.ones(class_num, 1))
#         else:
#             if isinstance(alpha, Variable):
#                 self.alpha = alpha
#             else:
#                 self.alpha = Variable(alpha)
#
#         self.gamma = gamma
#
#         # self.class_num = class_num
#         self.size_average = size_average
#
#     def forward(self, inputs, targets):
#         N = inputs.size(0)  # batch_size
#         C = inputs.size(1)  # channels
#         P = F.softmax(inputs, dim=1)
#
#         class_mask = inputs.data.new(N, C).fill_(0)
#         class_mask = Variable(class_mask)
#         ids = targets.view(-1, 1)
#         class_mask.scatter_(1, ids.data, 1.)
#         # print(class_mask)
#
#         if inputs.is_cuda and not self.alpha.is_cuda:
#             self.alpha = self.alpha.cuda()
#         alpha = self.alpha[ids.data.view(-1)]
#
#         probs = (P*class_mask).sum(1).view(-1, 1)
#
#         log_p = probs.log()
#         # print('probs size= {}'.format(probs.size()))
#         # print(probs)
#
#         batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
#
#         # print('-----bacth_loss------')
#         # print(batch_loss)
#
#         if self.size_average:
#             loss = batch_loss.mean()
#         else:
#             loss = batch_loss.sum()
#         return loss
#


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# I refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py

class FocalLoss2d(nn.Module):

    def __init__(self, gamma=0, weight=None, size_average=True):
        super(FocalLoss2d, self).__init__()

        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1,2)
            # input = input.contiguous().view(-1, input.size(2)).squeeze()
            input = input.contiguous().view(input.size(0),-1)
            # input = input.unsqueeze(0)
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        # compute the negative likelyhood
        weight = Variable(self.weight)
        print(input.size())
        print(target.size())
        logpt = -F.cross_entropy(input, target)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1-pt)**self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


# import numpy as np
# input = np.array([[1.1,5,8],[2.1,9,20]])
# input = torch.from_numpy(input)
# target = np.array([0,1])
# target = torch.from_numpy(target)
# logpt = -F.cross_entropy(input, target)

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=False):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        # print ("FOCAL LOSS", gamma, alpha)

    def forward(self, input, target):
        target = target.float()
        if input.dim() == 1:
            input = input.unsqueeze(1)
        if target.dim() == 1:
            target = target.unsqueeze(1)

        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C

        target = target.view(-1,1)
        target = target.float()
        pt = input * target + (1 - input) * (1 - target)

        logpt = (pt+0.0000001).log()
        at = (1 - self.alpha) * target + (self.alpha) * (1 - target)

        logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        #loss = torch.clamp(loss,min= -10.0,max=10.0)
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()