import torch
from torch.nn.modules.loss import _WeightedLoss


class BCECustomLoss(_WeightedLoss):

    def __init__(self, weight=None, mode=0, size_average=True, reduce=True):
        super(BCECustomLoss, self).__init__(weight, size_average, reduce)
        self.mode = mode

    def forward(self, input, target):
        assert len(self.weight) == 2

        loss = self.weight[0] * (target * torch.log(input + 1e-10)) + \
               self.weight[1] * ((1 - target) * torch.log(1 - input + 1e-10))

        if self.mode == 0:
            loss = -torch.mean(loss)
        else:
            loss = -torch.mean(torch.sum(loss, 1))
        return loss
