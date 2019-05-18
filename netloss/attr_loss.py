import torch
import numpy as np
import json
from netloss.bce_loss import BCECustomLoss
from netmodels.netmar.resnet import resnet101_mar


class AttributeLoss(torch.nn.Module):

    def __init__(self, weights_path, info_path, classes, requires_grad=True, mode=0):
        super(AttributeLoss, self).__init__()
        self.resmar = resnet101_mar(num_classes=classes)

        checkpoint = torch.load(weights_path)
        self.resmar.load_state_dict(checkpoint['marnet'])
        self.resmar = self.resmar.eval()

        with open(info_path, 'r') as tfile:
            a = json.load(tfile)

        weights = torch.zeros(2, classes)
        for i in range(classes):
            weights[0, i] = np.exp(1 - a[0][i])
            weights[1, i] = np.exp(a[0][i])

        self.w_bce = BCECustomLoss(weights, mode)

        if not requires_grad:
            for param in self.resmar.parameters():
                param.requires_grad = False

    def forward(self, y_pred, y_true):
        pred = self.resmar(y_pred)
        loss = self.w_bce(pred, y_true)
        return loss
