import json
import os
from torch.utils.data import DataLoader
from datasets.aic import AICDataset
from other_utils.path_config import PathMng
from other_utils.class_utils import Toperation
from netmodels.netmar.resnet import resnet101_mar
from other_utils.tensor_utils import *
from sklearn.metrics import roc_curve


def check_single_image(pm):
    data_test = AICDataset(pm.naic_dataset, comp_path=None, train_data=True,
                           type_data=Toperation.classification,
                           dst_size=[320, 128], tran=True, random_flip=0)
    test_loader = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=1)

    marnet = resnet101_mar(num_classes=24)

    resmar_weights_path = os.path.join(pm.naic_dataset, 'lastmar.checkpoint')
    checkpoint = torch.load(resmar_weights_path)
    marnet.load_state_dict(checkpoint['marnet'])
    marnet = marnet.cuda()
    elem_list = [[] for _ in range(24)]
    gt_list = [[] for _ in range(24)]
    ft = []
    st = [0.5] * 51
    for step, elem in enumerate(test_loader):

        img, att, _ = elem

        img = img.cuda()
        outatt = marnet(img)
        outatt = outatt.data.cpu().numpy().squeeze()
        att = att.cpu().numpy().squeeze()

        for i in range(24):
            elem_list[i].append(outatt[i])
            gt_list[i].append(att[i])
        if step % 1000 == 0:
            print(step)

    tmp_json = {}
    for i in range(24):
        fpr, tpr, thresholds = roc_curve(np.array(gt_list[i], int), np.array(elem_list[i]))
        tmp_json[i] = {'fpr': list(fpr), 'tpr': list(tpr), 'ths': thresholds}
        print(i, fpr, tpr, thresholds)
        optimal_idx = np.argmax(np.abs(tpr - fpr))
        optimal_threshold = thresholds[optimal_idx]
        print(i, optimal_threshold)
        ft.append(float(optimal_threshold))

    with open('./th.json', 'w') as fn:
        json.dump([ft, st], fn)


if __name__ == '__main__':
    pm = PathMng('')
    check_single_image(pm)
