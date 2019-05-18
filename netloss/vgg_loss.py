import torch
from netmodels.vgg.vgg_model import Vgg16
from other_utils.tensor_utils import normalize_batch


class VggLoss(torch.nn.Module):

    def __init__(self, size_average=True, reduce=True, gpu_id=0):
        super(VggLoss, self).__init__()
        self.reduce = reduce
        self.size_average = size_average
        self.vgg = Vgg16(requires_grad=False).cuda(device=gpu_id).eval()
        self.l = torch.nn.MSELoss().cuda(device=gpu_id)
        self.require_grad = False

    def forward(self, y_pred, y_true):
        y_pred = normalize_batch(y_pred)
        y_true = normalize_batch(y_true)
        feat_p = self.vgg(y_pred)
        feat_t = self.vgg(y_true)

        loss4 = self.l(feat_p.relu1_2, feat_t.relu1_2) + self.l(feat_p.relu2_2, feat_t.relu2_2) + \
                self.l(feat_p.relu3_3, feat_t.relu3_3) + self.l(feat_p.relu4_3, feat_t.relu4_3)

        loss4 /= 4
        return loss4
